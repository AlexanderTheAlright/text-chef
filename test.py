from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import re
import networkx as nx
from itertools import combinations
import pandas as pd
import traceback
import base64
import streamlit as st
import nltk
from nltk.corpus import stopwords
import os
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from io import BytesIO
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use('Agg')

# Set page config
st.set_page_config(
    page_title='Text Analysis Dashboard',
    page_icon='📊',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Only load NLTK data once
@st.cache_resource
def load_nltk_resources():
    for lang in ['english', 'french']:
        try:
            nltk.data.find(f'corpora/stopwords/{lang}')
        except LookupError:
            nltk.download('stopwords')

# Make sure to call it (e.g., in your main or early on)
load_nltk_resources()

def load_custom_stopwords_file():
    """Load custom stopwords from CSV file, returning a set (or empty set if not found)."""
    try:
        if os.path.exists('custom_stopwords.csv'):
            custom_stopwords_df = pd.read_csv('custom_stopwords.csv')
            return set(
                word.lower().strip()
                for word in custom_stopwords_df['word'].tolist()
                if isinstance(word, str)
            )
        return set()
    except Exception as e:
        st.error(f"Error loading custom stopwords file: {str(e)}")
        return set()

def normalize_stopword_set(words_set):
    """
    Convert each stopword to lowercase, remove punctuation/HTML/spaces, etc.,
    and split it the same way process_text() does so it lines up with your final tokenizer.
    """
    normalized = set()
    for w in words_set:
        w = str(w).lower().strip()
        w = re.sub(r'<[^>]+>', '', w)
        w = re.sub(r'[^\w\s]', ' ', w)
        w = re.sub(r'\s+', ' ', w).strip()
        if w:
            tokens = w.split()
            for t in tokens:
                if t:
                    normalized.add(t)
    return normalized

def load_cached_groups():
    """Load groups from cache file"""
    try:
        if os.path.exists('cached_groups.csv'):
            groups_df = pd.read_csv('cached_groups.csv')
            return groups_df.to_dict('records')
    except Exception as e:
        print(f"Error loading cached groups: {e}")
    return None

def load_cached_assignments():
    """Load assignments from cache file"""
    try:
        if os.path.exists('cached_assignments.csv'):
            assignments_df = pd.read_csv('cached_assignments.csv')
            return dict(zip(assignments_df.text, assignments_df.group))
    except Exception as e:
        print(f"Error loading cached assignments: {e}")
    return None

def initialize_stopwords():
    """Initialize stopwords from NLTK (English and French) plus any saved custom CSV entries."""
    # Only do this once if needed
    if 'custom_stopwords' not in st.session_state:
        try:
            # Get both English and French stopwords from NLTK
            english_stops = set(stopwords.words('english'))
            french_stops = set(stopwords.words('french'))

            # Load custom stopwords from CSV if available
            custom_stops = load_custom_stopwords_file()

            # Merge all stopwords
            merged_stops = english_stops.union(french_stops).union(custom_stops)

            # Normalize them so they match your text preprocessing
            st.session_state.custom_stopwords = normalize_stopword_set(merged_stops)
        except Exception as e:
            st.error(f"Error initializing stopwords: {str(e)}")
            st.session_state.custom_stopwords = set()

# Prevent automatic rerunning
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Disable automatic script re-running
if not st.session_state.initialized:
    st.session_state.initialized = True
    st.session_state.clicked = False
    st.session_state.file_processed = False
    st.session_state.current_viz = None
    st.session_state.current_groups = None
    initialize_stopwords()

# Initialize synonyms mapping as an empty dictionary
if 'synonyms' not in st.session_state:
    st.session_state.synonyms = {}

# Initialize synonym groups for advanced functionality
if 'synonym_groups' not in st.session_state:
    st.session_state.synonym_groups = defaultdict(set)

# Add initialization for additional session state variables used in the app
if 'sample_seed' not in st.session_state:
    st.session_state.sample_seed = None  # For consistent random sampling

if 'uploaded_file_processed' not in st.session_state:
    st.session_state.uploaded_file_processed = False  # Track file processing status

if 'grouping_options' not in st.session_state:
    st.session_state.grouping_options = []  # Store unique columns for grouping options

if 'var_open_columns' not in st.session_state:
    st.session_state.var_open_columns = {}  # Store *_open columns from all sheets

grouping_columns = []

# Data Loading
@st.cache_data
def load_excel_file(file, chosen_survey="All"):
    """
    Load only the question_mapping sheet plus either a single chosen survey
    or all survey sheets (if chosen_survey='All').
    """
    try:
        excel_file = pd.ExcelFile(file)
        sheets = excel_file.sheet_names

        if 'question_mapping' not in sheets:
            return None, None, None, None

        question_mapping = pd.read_excel(excel_file, 'question_mapping')
        if not all(col in question_mapping.columns for col in ['variable', 'question', 'surveyid']):
            return None, None, None, None

        # Prepare to store the chosen (or all) survey data
        responses_dict = {}
        available_open_vars = set()
        all_columns = set()

        # Exclude 'question_mapping' from surveys
        valid_sheets = [s for s in sheets if s != 'question_mapping']

        # Figure out which sheets to load
        if chosen_survey == "All":
            sheets_to_load = valid_sheets
        else:
            # Make sure user's choice is valid
            sheets_to_load = [sheet for sheet in valid_sheets if sheet == chosen_survey]

        # Define invalid response patterns
        invalid_responses = {
            'dk', 'dk.', 'd/k', 'd.k.', 'dont know', "don't know", "don't know",
            'na', 'n/a', 'n.a.', 'n/a.', 'not applicable', 'none', 'nil',
            'no response', 'no answer', '.', '-', 'x', 'refused', 'ref',
            'dk/ref', '', 'nan', 'NaN', 'NAN', '_dk_', '_na_', '___dk___', '___na___',
            '__dk__', '__na__', '_____dk_____', '_____na_____'
        }

        for sheet in sheets_to_load:
            # Read with na_values to catch Excel's missing value formats
            df = pd.read_excel(
                excel_file,
                sheet_name=sheet,
                na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
                           '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
                           'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'none']
            )

            base_columns = {col.split('.')[0] for col in df.columns}
            all_columns.update(base_columns)

            # Track columns ending in '_open'
            sheet_open_vars = {col for col in base_columns if col.endswith('_open')}
            available_open_vars.update(sheet_open_vars)

            # Clean all '_open' columns before storing
            for col in df.columns:
                if col.split('.')[0] in sheet_open_vars:
                    # First replace any remaining missing value patterns with NaN
                    df[col] = df[col].replace({r'^\s*$': pd.NA}, regex=True)  # Empty or whitespace

                    # Handle dk/na patterns with regex, case-insensitive
                    def clean_response(x):
                        if pd.isna(x):
                            return pd.NA
                        if not isinstance(x, str):
                            return pd.NA
                        x = str(x).lower().strip()
                        if any(x.startswith('_') and x.endswith('_') and pat in x for pat in ['dk', 'na']):
                            return pd.NA
                        if x in invalid_responses:
                            return pd.NA
                        return x.strip() if x.strip() else pd.NA

                    df[col] = df[col].apply(clean_response)

            # Store this DataFrame
            responses_dict[sheet] = df

        # Derive grouping columns
        grouping_columns = sorted(
            col for col in all_columns
            if not col.endswith('_open') and not col.endswith('.1')
        )

        # Build open_var_options mapping
        open_var_options = {
            var: f"{var} - {question_mapping[question_mapping['variable'] == var].iloc[0]['question']}"
            if not question_mapping[question_mapping['variable'] == var].empty
            else var
            for var in sorted(available_open_vars)
        }

        return question_mapping, responses_dict, open_var_options, grouping_columns

    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        return None, None, None, None


def get_available_variables(responses_dict, question_mapping):
    """
    Get available open-ended variables that exist in both question mapping and actual data.

    Parameters:
    - responses_dict: Dict of DataFrames containing survey responses
    - question_mapping: DataFrame containing variable mappings

    Returns:
    - open_var_options: Dict mapping variable names to their descriptions
    - grouping_columns: List of available grouping columns
    """
    # Track all columns that actually exist in the data
    existing_columns = set()
    for df in responses_dict.values():
        # Get base column names (remove .1 suffixes)
        base_columns = {col.split('.')[0] for col in df.columns}
        existing_columns.update(base_columns)

    print(f"Found columns in data: {existing_columns}")  # Debug

    # Find open-ended variables that exist in both mapping and data
    open_vars = set()
    for col in existing_columns:
        if col.endswith('_open'):
            # Check if variable exists in question mapping
            if not question_mapping[question_mapping['variable'] == col].empty:
                open_vars.add(col)

    print(f"Found open variables: {open_vars}")  # Debug

    # Build mapping of variables to descriptions
    open_var_options = {}
    for var in sorted(open_vars):
        question_row = question_mapping[question_mapping['variable'] == var]
        if not question_row.empty:
            description = question_row.iloc[0]['question']
            open_var_options[var] = f"{var} - {description}"
        else:
            open_var_options[var] = var

    # Get grouping columns (non-open-ended columns that exist in data)
    grouping_columns = sorted(
        col for col in existing_columns
        if not col.endswith('_open')
        and not col.endswith('.1')
        and col in existing_columns  # Ensure column actually exists
    )

    print(f"Found grouping columns: {grouping_columns}")  # Debug

    return open_var_options, grouping_columns

# Settings
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'file_processed': False,
        'chosen_survey': "All",
        'variable': None,  # Add variable to settings
        'group_by': 'None',
        'colormap': 'viridis',
        'highlight_words': set(),
        'search_word': '',
        'num_topics': 4,
        'min_topic_size': 2
    }


# Text processing
def apply_stopwords_to_texts(texts, stopwords):
    """
    Apply stopwords filtering to a list of texts

    Parameters:
    - texts: list - List of texts to process
    - stopwords: set - Set of stopwords to remove

    Returns:
    - list: Processed texts
    """
    processed_texts = []
    for text in texts:
        processed = process_text(text, stopwords=stopwords)
        if processed:  # Only add non-empty processed texts
            processed_texts.append(processed)
    return processed_texts


def display_word_search_results(texts_by_group, search_word):
    """Display all responses containing a specific word."""
    if not search_word:
        return

    matching_responses = find_word_in_responses(texts_by_group, search_word)
    total_matches = sum(len(responses) for responses in matching_responses.values())

    # Display summary statistics
    st.markdown(f"### Responses Containing: '{search_word}'")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Matching Responses", total_matches)

    with col2:
        total_responses = sum(len(texts) for texts in texts_by_group.values())
        match_percentage = (total_matches / total_responses * 100) if total_responses > 0 else 0
        st.metric("Match Percentage", f"{match_percentage:.1f}%")

    # Display matches by group
    if matching_responses:
        for group, responses in matching_responses.items():
            if responses:  # Only show groups with matches
                st.markdown(f"#### {group} ({len(responses)} matches)")
                for i, response in enumerate(responses, 1):
                    with st.expander(f"Response {i}", expanded=True):
                        # Highlight the search word
                        pattern = re.compile(f"({re.escape(search_word)})", re.IGNORECASE)
                        highlighted_text = pattern.sub(r"**:red[\1]**", response)
                        st.markdown(highlighted_text)
    else:
        st.warning(f"No responses found containing '{search_word}'.")

def get_text_columns(responses_df, question_mapping, survey_id):
    """Get all text-based columns that exist in the question mapping for the given survey"""
    # Get all variables for this survey from question mapping
    survey_vars = question_mapping[question_mapping['surveyid'] == survey_id]['variable'].tolist()

    # Filter for variables ending with '_open'
    open_vars = [var for var in survey_vars if str(var).endswith('_open')]

    # Get base column names from responses
    response_cols = set()
    for col in responses_df.columns:
        base_col = col.split('.')[0]  # Remove .1 suffix if present
        response_cols.add(base_col)

    # Return only variables that exist in both mapping and responses
    valid_vars = [var for var in open_vars if var in response_cols]

    return sorted(valid_vars)


def process_text(text, stopwords=None, synonym_groups=None):
    """
    Clean and process text with improved stopword handling and synonym group support.

    Parameters:
    - text: str - Input text to process
    - stopwords: set - Set of stopwords to remove (optional)
    - synonym_groups: dict - Dictionary mapping group names to sets of synonyms (optional)

    Returns:
    - str - Processed text
    """
    if pd.isna(text) or not isinstance(text, (str, bytes)):
        return ""

    # Convert to string and lowercase
    text = str(text).lower().strip()

    # Check for invalid responses
    invalid_responses = {'dk', 'dk.', 'd/k', 'd.k.', 'dont know', "don't know",
                         'na', 'n/a', 'n.a.', 'n/a.', 'not applicable',
                         'none', 'nil', 'no response', 'no answer', '.', '-', 'x'}

    if text in invalid_responses:
        return ""

    # Remove HTML tags and clean text
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    # Split into words
    words = text.split()

    # Apply stopwords if provided, otherwise use session state stopwords
    if stopwords is None and 'custom_stopwords' in st.session_state:
        stopwords = st.session_state.custom_stopwords

    if stopwords:
        words = [word for word in words if word not in stopwords]

    # Apply synonym groups if provided
    if synonym_groups:
        processed_words = []
        for word in words:
            replaced = False
            for group_name, synonyms in synonym_groups.items():
                if word in synonyms:
                    processed_words.append(group_name)
                    replaced = True
                    break
            if not replaced:
                processed_words.append(word)
        words = processed_words

    # Join and return
    return ' '.join(word for word in words if word)


def get_responses_for_variable(dfs_dict, var, group_by=None):
    """Get responses for a variable across all surveys"""
    responses_by_survey = {}

    for survey_id, df in dfs_dict.items():
        # Find matching columns for this variable
        var_pattern = f"^{re.escape(var)}(?:\.1)?$"
        matching_cols = [col for col in df.columns if re.match(var_pattern, col, re.IGNORECASE)]

        if not matching_cols:
            continue  # Skip if variable not in this survey

        if group_by and group_by in df.columns:
            grouped_responses = defaultdict(list)

            for col in matching_cols:
                temp_df = df[[col, group_by]].copy()
                temp_df[col] = temp_df[col].astype(str)
                temp_df[col] = temp_df[col].replace({'nan': '', 'None': '', 'NaN': ''})

                for group_val, group_df in temp_df.groupby(group_by):
                    responses = [
                        resp for resp in group_df[col].tolist()
                        if isinstance(resp, str) and resp.strip() and
                           resp.lower() not in {'nan', 'none', 'n/a', 'na'}
                    ]
                    if responses:
                        grouped_responses[str(group_val)].extend(responses)

            # Remove duplicates while preserving order
            for group_val in grouped_responses:
                seen = set()
                unique_responses = []
                for resp in grouped_responses[group_val]:
                    resp_clean = resp.strip()
                    if resp_clean and resp_clean not in seen:
                        seen.add(resp_clean)
                        unique_responses.append(resp_clean)
                if unique_responses:
                    responses_by_survey[f"{survey_id}_{group_val}"] = unique_responses
        else:
            # Original ungrouped logic
            responses = []
            for col in matching_cols:
                series = df[col].astype(str)
                series = series.replace({'nan': '', 'None': '', 'NaN': ''})

                valid_responses = [
                    resp for resp in series.tolist()
                    if isinstance(resp, str) and resp.strip() and
                       resp.lower() not in {'nan', 'none', 'n/a', 'na'}
                ]
                responses.extend(valid_responses)

            # Remove duplicates while preserving order
            seen = set()
            unique_responses = []
            for resp in responses:
                resp_clean = resp.strip()
                if resp_clean and resp_clean not in seen:
                    seen.add(resp_clean)
                    unique_responses.append(resp_clean)

            if unique_responses:
                responses_by_survey[survey_id] = unique_responses

    return responses_by_survey

# Stopwords
for lang in ['english', 'french']:
    try:
        nltk.data.find(f'corpora/stopwords/{lang}')
    except LookupError:
        nltk.download('stopwords')


def load_custom_stopwords_file():
    """Load custom stopwords from CSV file"""
    try:
        if os.path.exists('custom_stopwords.csv'):
            custom_stopwords_df = pd.read_csv('custom_stopwords.csv')
            return set(word.lower().strip()
                       for word in custom_stopwords_df['word'].tolist()
                       if isinstance(word, str))
        return set()
    except Exception as e:
        st.error(f"Error loading custom stopwords file: {str(e)}")
        return set()


def initialize_stopwords():
    """Initialize stopwords from NLTK (English and French) and custom additions"""
    if 'custom_stopwords' not in st.session_state:
        try:
            # Get both English and French stopwords
            english_stops = set(stopwords.words('english'))
            french_stops = set(stopwords.words('french'))

            # Combine NLTK stopwords
            nltk_stops = english_stops.union(french_stops)

            # Load custom stopwords from file
            custom_stops = load_custom_stopwords_file()

            # Merge all stopwords
            merged_stops = nltk_stops.union(custom_stops)

            ##################################################################
            # NEW: Normalize the final set so they match your text preprocessing
            ##################################################################
            st.session_state.custom_stopwords = normalize_stopword_set(merged_stops)

        except Exception as e:
            st.error(f"Error initializing stopwords: {str(e)}")
            st.session_state.custom_stopwords = set()

def normalize_stopword_set(words_set):
    """
    Convert each stopword to lowercase, remove punctuation/HTML/spaces, etc.,
    and *split* it the same way process_text() does.
    This ensures 'quelqu'un' becomes 'quelqu' and 'un'.
    """
    normalized = set()
    for w in words_set:
        # 1) Lowercase & strip
        w = str(w).lower().strip()
        # 2) Remove HTML
        w = re.sub(r'<[^>]+>', '', w)
        # 3) Replace punctuation with spaces
        w = re.sub(r'[^\w\s]', ' ', w)
        # 4) Collapse whitespace
        w = re.sub(r'\s+', ' ', w).strip()

        # 5) Split the cleaned stopword into tokens
        #    e.g., "quelqu'un" -> "quelqu un" -> ["quelqu","un"]
        if w:
            tokens = w.split()
            for t in tokens:
                if t:
                    normalized.add(t)
    return normalized

def update_stopwords_batch(new_words):
    """Add multiple new stopwords at once and save to CSV"""
    try:
        valid_words = {
            str(word).lower().strip()
            for word in new_words
            if word and isinstance(word, str) and not str(word).replace(".", "").replace("-", "").isnumeric()
        }

        if not valid_words:
            return False, "No valid stopwords provided"

        # Add new words to session state
        original_count = len(st.session_state.custom_stopwords)
        st.session_state.custom_stopwords.update(valid_words)

        # Normalize again to match your tokenizer
        st.session_state.custom_stopwords = normalize_stopword_set(st.session_state.custom_stopwords)

        # Read existing custom stopwords from CSV if it exists
        existing_custom_words = set()
        if os.path.exists('custom_stopwords.csv'):
            try:
                custom_df = pd.read_csv('custom_stopwords.csv')
                existing_custom_words = set(word.lower().strip()
                                            for word in custom_df['word'].tolist()
                                            if isinstance(word, str))
            except Exception:
                pass

        # Combine existing and new custom words
        all_custom_words = existing_custom_words.union(st.session_state.custom_stopwords)

        # Save updated custom stopwords to CSV
        custom_df = pd.DataFrame(sorted(all_custom_words), columns=['word'])
        custom_df.to_csv('custom_stopwords.csv', index=False)

        added_count = len(st.session_state.custom_stopwords) - original_count
        return True, f"Added {added_count} new stopwords and updated custom_stopwords.csv"
    except Exception as e:
        return False, f"Error updating stopwords: {str(e)}"


def remove_stopword(word):
    """Remove a stopword from the set and update CSV"""
    try:
        if word in st.session_state.custom_stopwords:
            st.session_state.custom_stopwords.remove(word)

            # Save updated stopwords to CSV
            stopwords_df = pd.DataFrame(
                sorted(st.session_state.custom_stopwords),
                columns=['word']
            )
            stopwords_df.to_csv('custom_stopwords.csv', index=False)
            return True, f"Removed '{word}' from stopwords"
        return False, f"'{word}' not found in stopwords"
    except Exception as e:
        return False, f"Error removing stopword: {str(e)}"


def reset_to_nltk_defaults():
    """Reset stopwords to NLTK defaults (English and French) plus custom defaults"""
    try:
        # Get both English and French stopwords
        english_stops = set(stopwords.words('english'))
        french_stops = set(stopwords.words('french'))

        # Load custom defaults
        custom_stops = load_custom_stopwords_file()

        # Combine them all
        merged_stops = english_stops.union(french_stops).union(custom_stops)

        ##################################################################
        # Normalize again to keep consistent with your text preprocessing
        ##################################################################
        st.session_state.custom_stopwords = normalize_stopword_set(merged_stops)

        # Save to CSV
        stopwords_df = pd.DataFrame(
            sorted(st.session_state.custom_stopwords),
            columns=['word']
        )
        stopwords_df.to_csv('custom_stopwords.csv', index=False)

        return True, (f"Reset to defaults ({len(english_stops)} English + "
                      f"{len(french_stops)} French + {len(custom_stops)} custom words)")
    except Exception as e:
        return False, f"Error resetting stopwords: {str(e)}"


def render_stopwords_management():
    with st.sidebar:
        st.markdown("⚙️ Stopwords Management")

        # Add multiple stopwords
        new_stopwords = st.text_area(
            "Add stopwords (one per line)",
            help="Enter multiple words, one per line",
            key="new_stopwords_input"
        )

        # View/Remove current stopwords
        view_stopwords = st.checkbox("View/Edit Current Stopwords", key="view_stopwords")

        if view_stopwords:
            stopwords_list = sorted(st.session_state.custom_stopwords)
            st.write(f"Total stopwords: {len(stopwords_list)}")

            # Add a search filter
            search_term = st.text_input("Filter stopwords", "")

            # Filter stopwords based on search term
            if search_term:
                filtered_stopwords = [word for word in stopwords_list
                                      if search_term.lower() in word.lower()]
            else:
                filtered_stopwords = stopwords_list

            # Display stopwords in columns with remove buttons
            cols = st.columns(2)
            for i, word in enumerate(filtered_stopwords):
                col = cols[i % 2]
                if col.button(f"❌ {word}", key=f"remove_{word}"):
                    success, message = remove_stopword(word)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        if st.button("Add Stopwords"):
            if new_stopwords:
                # Split by newlines and filter empty strings
                words_list = [w.strip() for w in new_stopwords.split('\n') if w.strip()]
                if words_list:
                    success, message = update_stopwords_batch(words_list)
                    if success:
                        st.success(message)
                    else:
                        st.warning(message)
                else:
                    st.warning("Please enter at least one word")

        # Reset to defaults
        if st.button("Reset to Defaults", type="secondary"):
            success, message = reset_to_nltk_defaults()
            if success:
                st.success(message)
            else:
                st.error(message)

# Table functions
def initialize_coding_state():
    """Initialize session state variables for open coding"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0

    if 'open_coding_groups' not in st.session_state:
        # Try to load from cache first
        cached_groups = load_cached_groups()
        st.session_state.open_coding_groups = cached_groups if cached_groups else []

    if 'open_coding_assignments' not in st.session_state:
        # Try to load from cache first
        cached_assignments = load_cached_assignments()
        st.session_state.open_coding_assignments = cached_assignments if cached_assignments else {}

    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = time.time()


def save_coding_state():
    """Save current coding state to CSV files"""
    try:
        # Save groups
        if st.session_state.open_coding_groups:
            groups_df = pd.DataFrame(st.session_state.open_coding_groups)
            groups_df.to_csv('cached_groups.csv', index=False)

        # Save assignments
        if st.session_state.open_coding_assignments:
            assignments_df = pd.DataFrame([
                {"text": k, "group": v}
                for k, v in st.session_state.open_coding_assignments.items()
            ])
            assignments_df.to_csv('cached_assignments.csv', index=False)

        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = "coding_backups"
        os.makedirs(backup_dir, exist_ok=True)

        if st.session_state.open_coding_groups:
            groups_df.to_csv(f'{backup_dir}/groups_{timestamp}.csv', index=False)
        if st.session_state.open_coding_assignments:
            assignments_df.to_csv(f'{backup_dir}/assignments_{timestamp}.csv', index=False)

        st.session_state.last_save_time = time.time()
        return True
    except Exception as e:
        print(f"Error saving coding state: {e}")
        return False


def load_cached_groups():
    """Load groups from cache file"""
    try:
        if os.path.exists('cached_groups.csv'):
            groups_df = pd.read_csv('cached_groups.csv')
            return groups_df.to_dict('records')
    except Exception as e:
        print(f"Error loading cached groups: {e}")
    return None


def load_cached_assignments():
    """Load assignments from cache file"""
    try:
        if os.path.exists('cached_assignments.csv'):
            assignments_df = pd.read_csv('cached_assignments.csv')
            return dict(zip(assignments_df.text, assignments_df.group))
    except Exception as e:
        print(f"Error loading cached assignments: {e}")
    return None


def auto_save_check():
    """Check if it's time to auto-save (every 5 minutes)"""
    if (time.time() - st.session_state.last_save_time) > 300:  # 5 minutes
        return save_coding_state()
    return False


def get_default_columns(df):
    """Get default columns (id, age, jobtitle, province/state) from dataframe"""
    default_cols = ['id', 'age', 'jobtitle']
    location_col = 'province' if 'province' in df.columns else 'state' if 'state' in df.columns else None

    available_cols = []
    for col in default_cols:
        if col in df.columns:
            available_cols.append(col)

    if location_col:
        available_cols.append(location_col)

    return available_cols


def get_filtered_table(df, filters):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()

    for col, value in filters.items():
        if value:
            filtered_df = filtered_df[
                filtered_df[col].astype(str).str.contains(value, case=False, na=False)
            ]

    return filtered_df


def update_table_with_filters(table, search_term=None, column_filters=None):
    """Update table based on search term and column filters"""
    if search_term or column_filters:
        filtered = table.copy()

        if search_term:
            mask = pd.Series(False, index=filtered.index)
            for col in filtered.columns:
                mask |= filtered[col].astype(str).str.contains(search_term, case=False, na=False)
            filtered = filtered[mask]

        if column_filters:
            filtered = get_filtered_table(filtered, column_filters)

        return filtered
    return table


def render_open_coding_tab(variable, responses_dict, open_var_options, grouping_columns):
    st.markdown("## 🗂️ Open Coding (All Results, Inline Group Editing)")

    # Initialize session state
    initialize_coding_state()

    # Display question box
    st.markdown("""
    <style>
        .question-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            background-color: #f8f9fa;
            margin: 10px 0;
        }
        .question-label {
            color: #2E7D32;
            font-size: 1.2em;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .question-text {
            color: #1a1a1a;
            font-size: 1.1em;
            line-height: 1.5;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    open_var_dict = st.session_state.data.get('open_var_options', {})

    st.markdown(f"""
    <div class="question-box">
        <div class="question-label">Primary Question</div>
        <div class="question-text">{open_var_dict.get(variable, "No question text")}</div>
    </div>
    """, unsafe_allow_html=True)

    ################################################################
    # Manage Groups Section
    ################################################################
    st.markdown("### Manage Groups")
    with st.expander("Create or Edit Groups", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_group_name = st.text_input("Group Name:")
            new_group_desc = st.text_input("Group Description (optional):")

            if st.button("Add / Update Group"):
                gname = new_group_name.strip()
                if gname:
                    existing = next((g for g in st.session_state.open_coding_groups if g["name"] == gname), None)
                    if existing:
                        existing["desc"] = new_group_desc.strip()
                        st.success(f"Updated description for group '{gname}'.")
                    else:
                        st.session_state.open_coding_groups.append({
                            "name": gname,
                            "desc": new_group_desc.strip()
                        })
                        st.success(f"Group '{gname}' created.")
                    save_coding_state()
                else:
                    st.warning("Please enter a valid group name.")

        with col2:
            if st.button("💾 Save Changes", type="primary"):
                if save_coding_state():
                    st.success("Changes saved successfully!")
                else:
                    st.error("Error saving changes")

        # Group removal
        remove_group = st.selectbox(
            "Delete a group?",
            options=["(None)"] + [g["name"] for g in st.session_state.open_coding_groups]
        )
        if remove_group != "(None)":
            if st.button("Remove Selected Group"):
                st.session_state.open_coding_groups = [
                    g for g in st.session_state.open_coding_groups
                    if g["name"] != remove_group
                ]
                # Unassign affected rows
                for txt_key, assigned_grp in list(st.session_state.open_coding_assignments.items()):
                    if assigned_grp == remove_group:
                        st.session_state.open_coding_assignments[txt_key] = "Unassigned"
                save_coding_state()
                st.success(f"Removed group '{remove_group}'.")

    ################################################################
    # Data Collection and Processing
    ################################################################
    st.markdown("### Data Selection")

    # Get all available columns
    all_dfs = []
    for survey_id, df in responses_dict.items():
        if variable in df.columns:
            # Add survey_id and process region before appending
            df_with_survey = df.copy()
            df_with_survey['surveyid'] = survey_id

            # Handle region (combine province and state)
            if 'province' in df_with_survey.columns:
                df_with_survey['region'] = df_with_survey['province']
                df_with_survey.drop('province', axis=1, inplace=True)
            elif 'state' in df_with_survey.columns:
                df_with_survey['region'] = df_with_survey['state']
                df_with_survey.drop('state', axis=1, inplace=True)

            all_dfs.append(df_with_survey)

    if not all_dfs:
        st.warning("No valid data found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Add region to default columns if it exists
    if 'region' in combined_df.columns:
        default_cols = get_default_columns(combined_df)
        default_cols.append('region')
    else:
        default_cols = get_default_columns(combined_df)

    # Get base variable name (removing _open suffix)
    base_var = variable.replace('_open', '')

    # Additional columns selection
    remaining_cols = [col for col in grouping_columns if col not in default_cols]
    user_vars = st.multiselect(
        f"Choose additional columns (besides {', '.join(default_cols)}):",
        options=remaining_cols,
        default=[]
    )

    # Build selected columns list ensuring surveyid is last
    selected_cols = []
    selected_cols.extend(default_cols)
    selected_cols.extend(user_vars)

    # Only add base variable if it exists
    if base_var in combined_df.columns:
        if base_var not in selected_cols:
            selected_cols.append(base_var)

    selected_cols.append(variable)
    selected_cols.append('surveyid')  # Add surveyid at the end

    # Remove any duplicates while preserving order
    selected_cols = list(dict.fromkeys(selected_cols))

    # Order
    selected_cols = list(dict.fromkeys(selected_cols))
    if 'id' in selected_cols and 'surveyid' in selected_cols:
        selected_cols.remove('surveyid')
        idx_of_id = selected_cols.index('id')
        selected_cols.insert(idx_of_id + 1, 'surveyid')

    ################################################################
    # Build and Display Enhanced Table
    ################################################################
    st.markdown("### Coding Table")

    # Search and filter controls
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search across all columns:", key="global_search")
    with col2:
        show_filters = st.checkbox("Show column filters", value=False)

    # Column filters
    column_filters = {}
    if show_filters:
        st.markdown("#### Column Filters")
        filter_cols = st.columns(3)
        for i, col in enumerate(selected_cols):
            with filter_cols[i % 3]:
                column_filters[col] = st.text_input(f"Filter {col}:", key=f"filter_{col}")

    # Get the data with all selected columns
    table_data = combined_df[selected_cols].copy()

    # >>> Add these lines to filter out blank or NaN responses for the _open variable <<<
    open_var_name = variable  # We'll refer to this again below, so define it now
    table_data = table_data.dropna(subset=[open_var_name])                       # drop rows where open var is NaN
    table_data = table_data[table_data[open_var_name].astype(str).str.strip() != ""]  # drop rows with blank strings

    # Create column mappings but keep original names
    closed_var_name = base_var if base_var in table_data.columns else None

    # Build rows
    table_rows = []
    for idx, row in table_data.iterrows():
        resp_text = row[open_var_name]
        assigned_grp = st.session_state.open_coding_assignments.get(resp_text, "Unassigned")

        # Get group description
        group_desc = ""
        if assigned_grp != "Unassigned":
            grp_obj = next((g for g in st.session_state.open_coding_groups if g["name"] == assigned_grp), None)
            if grp_obj:
                group_desc = grp_obj["desc"]

        # Build row data - keep original column names
        one_row = {col: row[col] for col in table_data.columns}
        one_row.update({
            "coded_group": assigned_grp,
            "group_description": group_desc
        })
        table_rows.append(one_row)

    coded_table_df = pd.DataFrame(table_rows)
    filtered_df = update_table_with_filters(coded_table_df, search_term, column_filters)

    # Display table with enhanced configuration
    group_options = ["Unassigned"] + [g["name"] for g in st.session_state.open_coding_groups]

    column_config = {
        "coded_group": st.column_config.SelectboxColumn(
            "Coded Group",
            options=group_options,
            help="Pick a group from dropdown"
        ),
        "group_description": st.column_config.TextColumn(
            "Group Description",
            help="Auto-filled after picking a group",
            disabled=True,
        ),
        "surveyid": st.column_config.TextColumn(
            "Survey ID",
            help="Source survey identifier",
            disabled=True
        )
    }

    # Add original column display names
    for col in filtered_df.columns:
        if col not in column_config:
            column_config[col] = st.column_config.TextColumn(
                col,
                help="Response data",
                disabled=True if col != open_var_name else False
            )

    edited_df = st.data_editor(
        filtered_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        key="coding_table"
    )

    # Update assignments
    for row_idx in edited_df.index:
        resp_text = edited_df.loc[row_idx, open_var_name]
        new_grp = edited_df.loc[row_idx, "coded_group"]
        st.session_state.open_coding_assignments[resp_text] = new_grp

    auto_save_check()

    ################################################################
    # Download Options
    ################################################################
    st.markdown("### Export Options")
    col1, col2 = st.columns(2)

    with col1:
        # Download current view
        csv_data = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Current View as CSV",
            data=csv_data,
            file_name=f"coded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Download all data
        full_csv = coded_table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Dataset",
            data=full_csv,
            file_name=f"complete_coded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# Sampling
def get_standard_samples(texts_by_group, n_samples=5, seed=None):
    """Get standard random samples with improved validation"""
    if seed is not None:
        np.random.seed(seed)

    samples_by_group = {}

    for group, texts in texts_by_group.items():
        # Print debug info
        print(f"Processing group {group}: {len(texts)} texts")

        # Filter valid texts
        valid_texts = [
            text for text in texts
            if isinstance(text, str) and text.strip() and
               not pd.isna(text) and text.lower() not in {'nan', 'none', 'n/a', 'na'}
        ]

        print(f"Valid texts for group {group}: {len(valid_texts)}")

        if valid_texts:
            n = min(n_samples, len(valid_texts))
            samples = np.random.choice(valid_texts, size=n, replace=False)
            samples_by_group[group] = list(samples)
        else:
            samples_by_group[group] = []
            print(f"No valid texts found for group {group}")

    return samples_by_group

def display_standard_samples(texts_by_group, n_samples=5):
    """Display standard random samples for each group with improved handling"""
    st.markdown("### Sample Responses")

    if not texts_by_group:
        st.warning("No responses available to display.")
        return

    # Create tabs for each group
    group_tabs = st.tabs(list(texts_by_group.keys()))

    for tab, group_name in zip(group_tabs, texts_by_group.keys()):
        with tab:
            texts = texts_by_group[group_name]
            if texts:
                # Filter valid texts
                valid_texts = [
                    text for text in texts
                    if isinstance(text, str) and text.strip() and
                       not pd.isna(text) and text.lower() not in {'nan', 'none', 'n/a', 'na'}
                ]

                if valid_texts:
                    # Get random samples
                    n = min(n_samples, len(valid_texts))
                    if st.session_state.get('sample_seed') is not None:
                        np.random.seed(st.session_state.sample_seed)
                    samples = np.random.choice(valid_texts, size=n, replace=False)

                    # Display samples
                    st.write(f"Showing {n} of {len(valid_texts)} responses")
                    for i, sample in enumerate(samples, 1):
                        with st.expander(f"Response {i}", expanded=True):
                            st.write(sample)
                else:
                    st.warning("No valid responses available for this group.")
            else:
                st.warning(f"No responses available for {group_name}")

# Charting
def safe_plotly_chart(figure, container, message="Unable to display visualization"):
    """Safely display a Plotly figure with error handling"""
    if figure is not None:
        try:
            if not isinstance(figure, go.Figure):
                container.warning("Invalid visualization data")
                return
            container.plotly_chart(figure, use_container_width=True)
        except Exception as e:
            container.warning(message)
    else:
        container.warning("No data available for visualization")

def find_word_in_responses(texts_by_group, search_word):
    """
    Find all responses containing a specific word across all groups.

    Parameters:
    - texts_by_group: dict -> Dictionary with group names as keys and lists of texts as values
    - search_word: str -> Word to search for

    Returns:
    - dict -> Dictionary with group names as keys and lists of matching responses as values
    """
    matching_responses = defaultdict(list)
    search_word = search_word.lower()

    for group, texts in texts_by_group.items():
        for text in texts:
            if isinstance(text, str) and search_word in text.lower():
                matching_responses[group].append(text)

    return dict(matching_responses)

# Synonyms
def add_synonym_management_to_sidebar():
    """Add synonym management controls to the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Synonym Groups Management")

    # Initialize synonym groups in session state if not present
    if 'synonym_groups' not in st.session_state:
        st.session_state.synonym_groups = defaultdict(set)

    # Add new synonym group
    new_group = st.sidebar.text_input("Enter new group name (e.g., 'money')")
    new_synonyms = st.sidebar.text_area(
        "Enter synonyms (one per line)",
        help="These terms will be treated as equivalent to the group name"
    )

    if st.sidebar.button("Add Synonym Group") and new_group and new_synonyms:
        group_name = new_group.lower().strip()
        synonyms = {word.lower().strip() for word in new_synonyms.split('\n') if word.strip()}
        st.session_state.synonym_groups[group_name] = synonyms
        st.sidebar.success(f"Added synonym group '{group_name}' with {len(synonyms)} terms")

    # Display and edit existing groups
    if st.session_state.synonym_groups:
        st.sidebar.markdown("### Existing Synonym Groups")
        for group_name, synonyms in dict(st.session_state.synonym_groups).items():
            with st.sidebar.expander(f"Group: {group_name}"):
                st.write("Synonyms:", ", ".join(sorted(synonyms)))
                if st.button(f"Delete {group_name}", key=f"del_{group_name}"):
                    del st.session_state.synonym_groups[group_name]
                    st.rerun()

def save_synonym_groups():
    """Save synonym groups to CSV."""
    try:
        rows = []
        for group_name, synonyms in st.session_state.synonym_groups.items():
            for synonym in synonyms:
                rows.append({
                    'group_name': group_name,
                    'synonym': synonym
                })

        df = pd.DataFrame(rows)
        df.to_csv('synonym_groups.csv', index=False)
        st.success("Saved synonym groups to synonym_groups.csv")
    except Exception as e:
        st.error(f"Error saving synonym groups: {e}")

def load_synonym_groups(file):
    """Load synonym groups from CSV."""
    try:
        df = pd.read_csv(file)
        new_groups = defaultdict(set)

        for _, row in df.iterrows():
            if pd.notna(row['group_name']) and pd.notna(row['synonym']):
                group_name = str(row['group_name']).lower().strip()
                synonym = str(row['synonym']).lower().strip()
                if group_name and synonym:
                    new_groups[group_name].add(synonym)

        st.session_state.synonym_groups = new_groups
        st.success(f"Loaded {len(new_groups)} synonym groups")
    except Exception as e:
        st.error(f"Error loading synonym groups: {e}")

def process_text_with_synonyms(text, stopwords=None, synonym_groups=None):
    """Process text with synonym group support."""
    if pd.isna(text) or not isinstance(text, (str, bytes)):
        return ""

    # Remove HTML tags and clean text
    text = re.sub(r'<[^>]+>', '', str(text))
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into words
    words = [word.strip() for word in text.split() if word.strip()]

    # Replace synonyms with their group names
    if synonym_groups:
        processed_words = []
        for word in words:
            replaced = False
            for group_name, synonyms in synonym_groups.items():
                if word in synonyms:
                    processed_words.append(group_name)
                    replaced = True
                    break
            if not replaced:
                processed_words.append(word)
        words = processed_words

    # Remove stopwords
    if stopwords:
        words = [word for word in words if word not in stopwords]

    return ' '.join(words)


# Word Clouds
def generate_wordcloud_data(texts, stopwords=None, synonyms=None, max_words=200):
    """Generate word frequency data for both static and interactive wordclouds"""
    if not texts:
        return None, None

    processed_texts = []
    for text in texts:
        if not isinstance(text, str):
            continue
        processed = process_text(text, stopwords, synonyms)
        if processed:
            processed_texts.append(processed)

    if not processed_texts:
        return None, None

    text = ' '.join(processed_texts)

    # Generate word frequencies
    words = text.split()
    word_freq = Counter(words)

    # Sort and limit to max_words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words]

    return text, sorted_words


def create_interactive_wordcloud(word_freq, colormap='viridis', title='', highlight_words=None):
    """
    Create an interactive Plotly wordcloud visualization with draggable words arranged in a network-like layout.
    Now starts more spread out and includes a 'hover' feature showing frequency and closest terms in a fancy table format.
    """
    if not word_freq:
        return None

    words, freqs = zip(*word_freq)
    max_freq = max(freqs)
    n_words = len(words)

    # Calculate sizes for words (normalized)
    sizes = [20 + (f / max_freq) * 60 for f in freqs]  # Slightly smaller size range

    # Create color scale
    if highlight_words:
        colors = [
            'red' if w.lower() in highlight_words else
            px.colors.sample_colorscale(colormap, f / max_freq)[0]
            for w, f in zip(words, freqs)
        ]
    else:
        colors = [px.colors.sample_colorscale(colormap, f / max_freq)[0] for f in freqs]

    # Calculate initial positions using polar coordinates
    # Increase spread by multiplying radius by a factor
    positions = []
    golden_ratio = (1 + 5 ** 0.5) / 2
    for i, freq in enumerate(freqs):
        # Radius increases as frequency decreases; multiply by 1.5 for extra spacing
        radius = 1.5 * (1 - freq / max_freq) * min(800, 100 * (n_words / 10))
        radius *= (0.8 + 0.4 * np.random.random())  # Add some randomness

        # Angle based on golden ratio for better distribution
        theta = i * 2 * np.pi * golden_ratio

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        positions.append((x, y))

    x_pos, y_pos = zip(*positions)

    # Identify the nearest terms for each word (by Euclidean distance in the 2D layout)
    def find_closest_words(idx, k=3):
        distances = []
        x_i, y_i = positions[idx]
        for j, (x_j, y_j) in enumerate(positions):
            if j != idx:
                dist = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
                distances.append((dist, j))
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        nearest_indices = [d[1] for d in distances[:k]]
        # Build a string of top k nearest word names
        return ", ".join([words[nid] for nid in nearest_indices])

    # Build customdata so hover can display more info
    # customdata[i] = (freq, word, closest_words)
    custom_data = []
    for i, w in enumerate(words):
        closest_str = find_closest_words(i, k=3)
        custom_data.append((freqs[i], w, closest_str))

    # Create figure with dragmode
    fig = go.Figure()

    # Scatter trace for words
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        text=words,
        textposition="middle center",
        mode='text+markers',
        marker=dict(
            size=1,
            color='rgba(0,0,0,0)'  # Invisible markers, needed for dragging
        ),
        textfont=dict(
            size=sizes,
            color=colors
        ),
        # customdata: [ (frequency, word, closest_words), ... ]
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Frequency: %{customdata[0]:.0f}<br>"
            "Closest Terms: %{customdata[2]}<extra></extra>"
        ),
        name=''
    ))

    # Update layout for better interactivity
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        dragmode='pan',
        width=1000,
        height=800,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1000, 1000]  # Adjust based on your needs
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1000, 1000],
            scaleanchor='x',
            scaleratio=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=50, l=50, r=50),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Reset View',
                        method='relayout',
                        args=[{
                            'xaxis.range': [-1000, 1000],
                            'yaxis.range': [-1000, 1000]
                        }]
                    )
                ],
                x=0.05,
                y=1.05,
            )
        ]
    )

    return fig


def generate_wordcloud(texts, stopwords=None, synonyms=None, colormap='viridis',
                       highlight_words=None, return_freq=False):
    """Generate both static and interactive wordclouds"""
    # Generate word frequency data
    text, word_freq = generate_wordcloud_data(texts, stopwords, synonyms)
    if not text or not word_freq:
        return None, None, None

    try:
        def color_func(word, *args, **kwargs):
            if highlight_words and word.lower() in highlight_words:
                return "hsl(0, 100%, 50%)"
            return None

        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap=colormap if not highlight_words else None,
            color_func=color_func if highlight_words else None,
            stopwords=stopwords if stopwords else set(),
            collocations=False,
            min_word_length=2,
            prefer_horizontal=0.7,
            max_words=200
        ).generate(text)

        # Create interactive version
        interactive_wc = create_interactive_wordcloud(
            word_freq,
            colormap,
            highlight_words=highlight_words
        )

        if return_freq:
            return wc, interactive_wc, word_freq
        return wc, interactive_wc, None

    except Exception as e:
        print(f"Error generating wordcloud: {str(e)}")
        return None, None, None


def generate_synonym_group_wordclouds(texts, stopwords=None, synonym_groups=None, colormap='viridis'):
    """Generate separate wordclouds for each synonym group"""
    if not texts or not synonym_groups:
        return None, None

    n_groups = len(synonym_groups)
    if n_groups == 0:
        return None, None

    # Create static figure
    rows = int(np.ceil(np.sqrt(n_groups)))
    cols = int(np.ceil(n_groups / rows))
    fig_static = plt.figure(figsize=(16, 16))
    gs = fig_static.add_gridspec(rows, cols)

    # Create interactive figure
    fig_interactive = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(synonym_groups.keys())
    )

    for idx, (group_name, synonyms) in enumerate(synonym_groups.items()):
        row = idx // cols + 1
        col = idx % cols + 1

        # Process texts for this group
        text, word_freq = generate_wordcloud_data(
            texts,
            stopwords,
            {group_name: synonyms}
        )

        if text and word_freq:
            # Static wordcloud
            ax = fig_static.add_subplot(gs[idx])
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=colormap,
                stopwords=stopwords if stopwords else set(),
                collocations=False,
                min_word_length=2
            ).generate(text)

            ax.imshow(wc)
            ax.axis('off')
            ax.set_title(f"Group: {group_name}\nSynonyms: {', '.join(synonyms)}")

            # Interactive wordcloud
            interactive_wc = create_interactive_wordcloud(
                word_freq,
                colormap,
                title=f"Group: {group_name}"
            )
            for trace in interactive_wc.data:
                fig_interactive.add_trace(trace, row=row, col=col)

            # Update subplot layout
            fig_interactive.update_xaxes(showgrid=False, showticklabels=False, row=row, col=col)
            fig_interactive.update_yaxes(showgrid=False, showticklabels=False, row=row, col=col)

    # Update layouts
    fig_static.tight_layout()

    fig_interactive.update_layout(
        showlegend=False,
        height=300 * rows,
        width=400 * cols,
        title="Synonym Group Wordclouds"
    )

    return fig_static, fig_interactive


def generate_comparison_wordcloud(texts1, texts2, stopwords=None, synonyms=None, colormap='viridis',
                                  highlight_words=None, main_title="", subtitle="", source_text="",
                                  label1="Group 1", label2="Group 2"):
    """Generate both static and interactive comparison wordclouds"""
    # Generate word frequencies for both groups
    text1, word_freq1 = generate_wordcloud_data(texts1, stopwords, synonyms)
    text2, word_freq2 = generate_wordcloud_data(texts2, stopwords, synonyms)

    if not text1 or not text2:
        return None, None

    # Create static comparison
    fig_static = plt.figure(figsize=(20, 10))
    gs = fig_static.add_gridspec(1, 2)
    fig_static.patch.set_facecolor('white')

    def color_func(word, *args, **kwargs):
        if highlight_words and word.lower() in highlight_words:
            return "hsl(0, 100%, 50%)"
        return None

    # Generate static wordclouds
    for i, (text, label) in enumerate([(text1, label1), (text2, label2)]):
        ax = fig_static.add_subplot(gs[i])
        wc = WordCloud(
            width=1500,
            height=1000,
            background_color="white",
            colormap=colormap if not highlight_words else None,
            color_func=color_func if highlight_words else None,
            stopwords=stopwords if stopwords else set(),
            collocations=False,
            max_words=150,
            max_font_size=200,
            prefer_horizontal=1,
            scale=2
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label, fontsize=15, fontweight='bold')

    # Create interactive comparison
    fig_interactive = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[label1, label2]
    )

    # Add interactive wordclouds
    interactive_wc1 = create_interactive_wordcloud(word_freq1, colormap, highlight_words=highlight_words)
    interactive_wc2 = create_interactive_wordcloud(word_freq2, colormap, highlight_words=highlight_words)

    for trace in interactive_wc1.data:
        fig_interactive.add_trace(trace, row=1, col=1)
    for trace in interactive_wc2.data:
        fig_interactive.add_trace(trace, row=1, col=2)

    # Update layouts
    if subtitle:
        fig_static.suptitle(f"{main_title}\n{subtitle}", y=0.95)
    elif main_title:
        fig_static.suptitle(main_title, y=0.95)

    fig_interactive.update_layout(
        title={
            'text': f"{main_title}<br><sup>{subtitle}</sup>" if subtitle else main_title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        width=1600,
        height=800
    )

    # Add source text if provided
    if source_text:
        fig_static.text(0.05, 0.02, f"Source: {source_text}", fontsize=10, color='gray')
        fig_interactive.add_annotation(
            text=f"Source: {source_text}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=-0.05,
            showarrow=False,
            font=dict(size=10, color='gray'),
            align="left"
        )

    return fig_static, fig_interactive


def generate_combined_comparison_wordcloud(texts_by_group, categories_side1, categories_side2,
                                           stopwords=None, synonyms=None, colormap='viridis',
                                           highlight_words=None):
    """
    Generate larger, cleaner wordclouds comparing combined categories.
    Filters out invalid responses and removes labels for cleaner visualization.
    """
    # Combine texts for each side
    texts1 = []
    texts2 = []

    # Invalid response patterns
    invalid_patterns = {
        'dk', 'dk.', 'd/k', 'd.k.', 'dont know', "don't know",
        'na', 'n/a', 'n.a.', 'n/a.', 'not applicable',
        'none', 'nil', 'no response', 'no answer', '.', '-', 'x'
    }

    def is_valid_response(text):
        if not isinstance(text, str):
            return False
        text = text.lower().strip()
        return text and text not in invalid_patterns

    # Collect valid responses for each side
    for category in categories_side1:
        if category in texts_by_group:
            texts1.extend([t for t in texts_by_group[category] if is_valid_response(t)])

    for category in categories_side2:
        if category in texts_by_group:
            texts2.extend([t for t in texts_by_group[category] if is_valid_response(t)])

    if not texts1 or not texts2:
        return None, None

    # Process texts and generate word frequency data
    text1, word_freq1 = generate_wordcloud_data(texts1, stopwords, synonyms)
    text2, word_freq2 = generate_wordcloud_data(texts2, stopwords, synonyms)

    if not text1 or not text2:
        return None, None

    # Create larger static comparison
    fig_static = plt.figure(figsize=(24, 12))  # Increased figure size
    gs = fig_static.add_gridspec(1, 2, wspace=0.1)  # Reduced spacing
    fig_static.patch.set_facecolor('white')

    def color_func(word, *args, **kwargs):
        if highlight_words and word.lower() in highlight_words:
            return "hsl(0, 100%, 50%)"
        return None

    # Generate static wordclouds with larger dimensions
    for i, text in enumerate([text1, text2]):
        ax = fig_static.add_subplot(gs[i])
        wc = WordCloud(
            width=2000,  # Increased width
            height=1200,  # Increased height
            background_color="white",
            colormap=colormap if not highlight_words else None,
            color_func=color_func if highlight_words else None,
            stopwords=stopwords if stopwords else set(),
            collocations=False,
            max_words=150,
            max_font_size=250,  # Increased font size
            prefer_horizontal=0.7,
            scale=2
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis('off')

    # Create interactive comparison with larger dimensions
    fig_interactive = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.05  # Reduced spacing
    )

    # Add interactive wordclouds
    interactive_wc1 = create_interactive_wordcloud(word_freq1, colormap,
                                                   highlight_words=highlight_words)
    interactive_wc2 = create_interactive_wordcloud(word_freq2, colormap,
                                                   highlight_words=highlight_words)

    for trace in interactive_wc1.data:
        fig_interactive.add_trace(trace, row=1, col=1)
    for trace in interactive_wc2.data:
        fig_interactive.add_trace(trace, row=1, col=2)

    # Update layout with larger size
    fig_interactive.update_layout(
        showlegend=False,
        width=2000,
        height=1000,
        margin=dict(t=20, b=20, l=20, r=20),  # Reduced margins
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Remove axes for cleaner look
    fig_interactive.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig_interactive.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig_static, fig_interactive

def generate_multi_group_wordcloud(texts_by_group, stopwords=None, synonyms=None, colormap='viridis',
                                   highlight_words=None, main_title="", subtitle="", source_text=""):
    """Generate both static and interactive multi-group wordclouds"""
    if len(texts_by_group) > 4:
        raise ValueError("Maximum 4 groups supported")

    # Generate static version
    fig_static = plt.figure(figsize=(16, 14))
    gs = fig_static.add_gridspec(2, 2)
    fig_static.patch.set_facecolor('white')

    # Create interactive figure
    fig_interactive = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(texts_by_group.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    def color_func(word, *args, **kwargs):
        if highlight_words and word.lower() in highlight_words:
            return "hsl(0, 100%, 50%)"
        return None

    for i, (group_name, texts) in enumerate(texts_by_group.items()):
        row = i // 2 + 1
        col = i % 2 + 1

        # Generate word frequency data
        text, word_freq = generate_wordcloud_data(texts, stopwords, synonyms)

        if text and word_freq:
            # Static wordcloud
            ax = fig_static.add_subplot(gs[i])
            wc = WordCloud(
                width=1500,
                height=1000,
                background_color="white",
                colormap=colormap if not highlight_words else None,
                color_func=color_func if highlight_words else None,
                stopwords=stopwords if stopwords else set(),
                collocations=False,
                max_words=100,
                max_font_size=200,
                prefer_horizontal=1,
                scale=2
            ).generate(text)

            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{group_name}\nTotal responses: {len(texts)}")

            # Interactive wordcloud
            interactive_wc = create_interactive_wordcloud(
                word_freq,
                colormap=colormap,
                title=group_name,
                highlight_words=highlight_words
            )

            for trace in interactive_wc.data:
                fig_interactive.add_trace(trace, row=row, col=col)

            # Update subplot layout
            fig_interactive.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
            fig_interactive.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)

    # Update layouts
    fig_static.tight_layout()
    if subtitle:
        fig_static.suptitle(f"{main_title}\n{subtitle}", y=0.95)
    elif main_title:
        fig_static.suptitle(main_title, y=0.95)

    fig_interactive.update_layout(
        title={
            'text': f"{main_title}<br><sup>{subtitle}</sup>" if subtitle else main_title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        width=1600,
        height=1400,
        template="plotly_white",
        margin=dict(t=100, b=50, l=50, r=50)
    )

    # Add source text if provided
    if source_text:
        fig_static.text(0.05, 0.02, f"Source: {source_text}", fontsize=10, color='gray')
        fig_interactive.add_annotation(
            text=f"Source: {source_text}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=-0.05,
            showarrow=False,
            font=dict(size=10, color='gray'),
            align="left"
        )

    return fig_static, fig_interactive


def generate_synonym_group_wordclouds(texts, stopwords=None, synonym_groups=None, colormap='viridis'):
    """Generate separate wordclouds for each synonym group with both static and interactive versions"""
    if not texts or not synonym_groups:
        return None, None

    n_groups = len(synonym_groups)
    if n_groups == 0:
        return None, None

    # Create static figure
    rows = int(np.ceil(np.sqrt(n_groups)))
    cols = int(np.ceil(n_groups / rows))
    fig_static = plt.figure(figsize=(16, 16))
    gs = fig_static.add_gridspec(rows, cols)
    fig_static.patch.set_facecolor('white')

    # Create interactive figure
    fig_interactive = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Group: {group}" for group in synonym_groups.keys()],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )

    for idx, (group_name, synonyms) in enumerate(synonym_groups.items()):
        row = idx // cols + 1
        col = idx % cols + 1

        # Process texts for this group
        text, word_freq = generate_wordcloud_data(
            texts,
            stopwords,
            {group_name: synonyms}
        )

        if text and word_freq:
            # Static wordcloud
            ax = fig_static.add_subplot(gs[idx])
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=colormap,
                stopwords=stopwords if stopwords else set(),
                collocations=False,
                min_word_length=2,
                max_words=100
            ).generate(text)

            ax.imshow(wc)
            ax.axis('off')
            ax.set_title(f"Group: {group_name}\nSynonyms: {', '.join(synonyms)}")

            # Interactive wordcloud
            interactive_wc = create_interactive_wordcloud(
                word_freq,
                colormap=colormap,
                title=f"Group: {group_name}"
            )

            for trace in interactive_wc.data:
                fig_interactive.add_trace(trace, row=row, col=col)

            # Update subplot layout
            fig_interactive.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
            fig_interactive.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)

    # Update layouts
    fig_static.tight_layout()

    fig_interactive.update_layout(
        showlegend=False,
        width=400 * cols,
        height=300 * rows,
        template="plotly_white",
        title={
            'text': "Synonym Group Wordclouds",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    return fig_static, fig_interactive

# Helper function to convert PIL Image to base64 for download
def get_image_download_link(image, filename, text):
    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Network Graphs
def create_word_cooccurrence_network(texts, min_edge_weight=2, max_words=50, stopwords=None):
    """
    Create a network visualization of word co-occurrences.

    Parameters:
    - texts: list of strings
    - min_edge_weight: minimum number of co-occurrences to include edge
    - max_words: maximum number of words to include in the network
    - stopwords: set of stopwords to exclude

    Returns:
    - plotly figure object
    """
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(
        max_features=max_words,
        stop_words=list(stopwords) if stopwords else None,
        token_pattern=r'\b\w+\b'  # Only match whole words
    )

    # Fit and transform the texts
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    # Calculate word co-occurrence matrix
    word_cooc = (X.T @ X).toarray()
    np.fill_diagonal(word_cooc, 0)  # Remove self-connections

    # Create network graph
    G = nx.Graph()

    # Add nodes (words)
    word_freq = X.sum(axis=0).A1
    for i, word in enumerate(words):
        G.add_node(word, frequency=int(word_freq[i]))

    # Add edges (co-occurrences)
    for i, j in combinations(range(len(words)), 2):
        weight = word_cooc[i, j]
        if weight >= min_edge_weight:
            G.add_edge(words[i], words[j], weight=weight)

    # Calculate node positions using a force-directed layout
    pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

    # Create edge trace
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.5
    )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        freq = G.nodes[node]['frequency']
        node_text.append(f"{node}<br>Frequency: {freq}")
        node_sizes.append(np.sqrt(freq) * 10)  # Scale node sizes by sqrt of frequency

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            color=[G.nodes[node]['frequency'] for node in G.nodes()],
            size=node_sizes,
            line_width=2
        )
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Word Co-occurrence Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800,
            plot_bgcolor='black'
        )
    )

    return fig

def get_top_cooccurrences(texts, n_words=20, n_pairs=10, stopwords=None):
    """
    Get the top co-occurring word pairs and their frequencies.

    Parameters:
    - texts: list of strings
    - n_words: number of words to consider
    - n_pairs: number of top pairs to return
    - stopwords: set of stopwords to exclude

    Returns:
    - pandas DataFrame with word pairs and their co-occurrence counts
    """
    vectorizer = CountVectorizer(
        max_features=n_words,
        stop_words=list(stopwords) if stopwords else None
    )

    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    # Calculate co-occurrence matrix
    cooc_matrix = (X.T @ X).toarray()
    np.fill_diagonal(cooc_matrix, 0)

    # Get top co-occurring pairs
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if cooc_matrix[i, j] > 0:
                pairs.append({
                    'word1': words[i],
                    'word2': words[j],
                    'cooccurrences': int(cooc_matrix[i, j])
                })

    # Convert to DataFrame and sort
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('cooccurrences', ascending=False).head(n_pairs)

    return pairs_df

# Topic Model
def create_topic_distance_map(topic_model, processed_texts):
    """Create interactive topic distance visualization."""
    # Get topic embeddings
    embeddings = topic_model.topic_embeddings_

    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_sizes = topic_info['Count'].values

    # Normalize sizes for visualization
    size_scale = 50
    normalized_sizes = np.sqrt(topic_sizes) * size_scale

    # Create hover text
    hover_text = []
    for topic_id, size in zip(topic_info['Topic'], topic_sizes):
        if topic_id != -1:  # Skip outlier topic
            words = topic_model.get_topic(topic_id)
            top_words = ", ".join([word for word, _ in words[:5]])
            hover_text.append(
                f"Topic {topic_id}<br>"
                f"Size: {size} documents<br>"
                f"Top words: {top_words}"
            )
        else:
            hover_text.append(f"Outlier Topic<br>Size: {size} documents")

    # Create scatter plot
    fig = go.Figure()

    # Add topics
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode='markers+text',
        marker=dict(
            size=normalized_sizes,
            color=topic_info['Topic'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Topic ID")
        ),
        text=topic_info['Topic'],
        hovertext=hover_text,
        hoverinfo='text',
        textposition="middle center",
    ))

    # Update layout
    fig.update_layout(
        title="Topic Distance Map",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        showlegend=False,
        width=800,
        height=600
    )

    return fig

def create_topic_hierarchy(topic_model, topics, topic_info):
    """Create hierarchical topic visualization."""
    # Create hierarchical structure
    fig = go.Figure()

    # Add sunburst chart
    labels = []
    parents = []
    values = []
    text = []

    # Add root
    labels.append('All Topics')
    parents.append('')
    values.append(sum(topic_info['Count']))
    text.append('All Topics')

    # Add topics
    for topic_id in topics:
        if topic_id != -1:  # Skip outlier topic
            # Get topic words and their scores
            topic_words = topic_model.get_topic(topic_id)

            # Add topic level
            topic_label = f'Topic {topic_id}'
            labels.append(topic_label)
            parents.append('All Topics')
            values.append(topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0])

            # Add words for this topic
            for word, score in topic_words[:5]:  # Top 5 words
                word_label = f'{topic_label}_{word}'
                labels.append(word_label)
                parents.append(topic_label)
                values.append(int(score * 100))  # Scale up the scores for better visualization
                text.append(f'{word}<br>Score: {score:.3f}')

    fig.add_trace(go.Sunburst(
        ids=labels,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        hovertext=text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title="Topic Hierarchy and Key Terms",
        width=800,
        height=800
    )

    return fig

def analyze_topic_diversity(topic_model, processed_texts):
    """Analyze topic diversity and coherence."""
    # Get topic distribution for each document
    topic_distr, _ = topic_model.transform(processed_texts)

    # Calculate diversity metrics
    topic_entropy = -np.sum(topic_distr * np.log2(topic_distr + 1e-10), axis=1)
    avg_entropy = np.mean(topic_entropy)

    # Calculate document-topic concentration
    max_topic_probs = np.max(topic_distr, axis=1)
    avg_concentration = np.mean(max_topic_probs)

    return {
        'avg_entropy': avg_entropy,
        'avg_concentration': avg_concentration,
        'topic_entropy': topic_entropy,
        'max_topic_probs': max_topic_probs
    }

def create_topic_evolution(topic_model, texts_by_group):
    """Create topic evolution visualization across groups."""
    # Process each group
    group_topic_distributions = {}
    for group, texts in texts_by_group.items():
        if texts:
            # Transform texts to get topic distributions
            topic_distr, _ = topic_model.transform(texts)
            # Calculate average distribution for the group
            avg_distr = np.mean(topic_distr, axis=0)
            group_topic_distributions[group] = avg_distr

    # Create heatmap
    if group_topic_distributions:
        groups = list(group_topic_distributions.keys())
        topic_distrib = np.array([group_topic_distributions[g] for g in groups])

        fig = go.Figure(data=go.Heatmap(
            z=topic_distrib,
            x=[f'Topic {i}' for i in range(topic_distrib.shape[1])],
            y=groups,
            colorscale='Viridis',
            hoverongaps=False
        ))

        fig.update_layout(
            title='Topic Distribution Across Groups',
            xaxis_title='Topics',
            yaxis_title='Groups',
            height=400
        )

        return fig

    return None

# Sentiment Analyses
def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER.
    Returns compound score and sentiment category.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    # Categorize sentiment based on compound score
    if scores['compound'] >= 0.05:
        category = 'Positive'
    elif scores['compound'] <= -0.05:
        category = 'Negative'
    else:
        category = 'Neutral'

    return {
        'compound': scores['compound'],
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu'],
        'category': category
    }

def analyze_group_sentiment(texts_by_group):
    """
    Analyze sentiment for each group's texts.
    Returns sentiment statistics by group.
    """
    sentiment_stats = defaultdict(lambda: {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'avg_compound': 0,
        'scores': []
    })

    for group, texts in texts_by_group.items():
        for text in texts:
            if pd.isna(text) or not str(text).strip():
                continue

            sentiment = analyze_sentiment(str(text))
            sentiment_stats[group]['total'] += 1
            sentiment_stats[group]['scores'].append(sentiment['compound'])

            if sentiment['category'] == 'Positive':
                sentiment_stats[group]['positive'] += 1
            elif sentiment['category'] == 'Negative':
                sentiment_stats[group]['negative'] += 1
            else:
                sentiment_stats[group]['neutral'] += 1

    # Calculate averages and percentages
    for group in sentiment_stats:
        total = sentiment_stats[group]['total']
        if total > 0:
            sentiment_stats[group]['avg_compound'] = sum(sentiment_stats[group]['scores']) / total
            sentiment_stats[group]['pos_pct'] = (sentiment_stats[group]['positive'] / total) * 100
            sentiment_stats[group]['neg_pct'] = (sentiment_stats[group]['negative'] / total) * 100
            sentiment_stats[group]['neu_pct'] = (sentiment_stats[group]['neutral'] / total) * 100

    return sentiment_stats

def create_sentiment_sunburst(sentiment_stats):
    """Create a sunburst chart showing sentiment distribution by group."""
    if not sentiment_stats:
        return None

    try:
        # Prepare data for sunburst chart
        labels = []
        parents = []
        values = []
        colors = []

        # Color scheme
        color_map = {
            'Positive': '#2ecc71',
            'Neutral': '#f1c40f',
            'Negative': '#e74c3c'
        }

        # Add root
        total_responses = sum(stats['total'] for stats in sentiment_stats.values())
        if total_responses == 0:
            return None

        labels.append('All Responses')
        parents.append('')
        values.append(total_responses)
        colors.append('#3498db')

        # Add groups and their sentiments
        for group, stats in sentiment_stats.items():
            if stats['total'] > 0:  # Only add groups with responses
                # Add group
                labels.append(str(group))
                parents.append('All Responses')
                values.append(stats['total'])
                colors.append('#3498db')

                # Add sentiments for this group
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    labels.append(f'{group} {sentiment}')
                    parents.append(str(group))
                    values.append(stats[sentiment.lower()])
                    colors.append(color_map[sentiment])

        if len(labels) <= 1:  # Check if we have enough data
            return None

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Responses: %{value}<br>Percentage: %{percentParent:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            width=800,
            height=800,
            title={
                'text': 'Sentiment Distribution by Group',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            }
        )

        return fig
    except Exception as e:
        print(f"Error creating sunburst chart: {str(e)}")
        return None

def create_sentiment_radar(sentiment_stats):
    """
    Create a radar chart comparing sentiment metrics across groups.
    """
    categories = ['Positive %', 'Neutral %', 'Negative %', 'Sentiment Score']

    fig = go.Figure()

    for group, stats in sentiment_stats.items():
        # Scale compound score to percentage for visualization
        sentiment_score = (stats['avg_compound'] + 1) * 50  # Convert [-1,1] to [0,100]

        fig.add_trace(go.Scatterpolar(
            r=[stats['pos_pct'],
               stats['neu_pct'],
               stats['neg_pct'],
               sentiment_score],
            theta=categories,
            name=str(group),
            fill='toself',
            hovertemplate='<b>%{theta}</b><br>Value: %{r:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title={
            'text': 'Sentiment Metrics Comparison',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        width=800,
        height=800
    )

    return fig

def create_sentiment_distribution(sentiment_stats):
    """Create a violin plot showing sentiment score distribution by group with graceful error handling"""
    if not sentiment_stats:
        return None

    try:
        # Prepare data for violin plot
        data = []
        for group, stats in sentiment_stats.items():
            if 'scores' in stats and stats['scores']:
                for score in stats['scores']:
                    data.append({
                        'Group': str(group),
                        'Sentiment Score': float(score)
                    })

        if not data:
            return None

        df = pd.DataFrame(data)

        try:
            fig = px.violin(
                df,
                x='Group',
                y='Sentiment Score',
                box=True,
                points="outliers",
                title='Sentiment Score Distribution by Group',
                template='plotly_white'
            )

            fig.update_layout(
                showlegend=False,
                title={
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24}
                },
                yaxis_title='Sentiment Score (Negative → Positive)',
                xaxis_title='Group',
                width=800,
                height=600
            )

            return fig

        except Exception as e:
            logging.error(f"Error creating violin plot: {str(e)}")
            return None

    except Exception as e:
        logging.error(f"Error processing sentiment data: {str(e)}")
        return None

def analyze_group_sentiment(texts_by_group):
    """Analyze sentiment for each group's texts with improved error handling"""
    try:
        sentiment_stats = defaultdict(lambda: {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'avg_compound': 0,
            'scores': []
        })

        analyzer = SentimentIntensityAnalyzer()

        for group, texts in texts_by_group.items():
            valid_texts = [str(text) for text in texts if pd.notna(text) and str(text).strip()]

            if not valid_texts:
                continue

            for text in valid_texts:
                try:
                    scores = analyzer.polarity_scores(text)
                    sentiment_stats[group]['total'] += 1
                    sentiment_stats[group]['scores'].append(scores['compound'])

                    if scores['compound'] >= 0.05:
                        sentiment_stats[group]['positive'] += 1
                    elif scores['compound'] <= -0.05:
                        sentiment_stats[group]['negative'] += 1
                    else:
                        sentiment_stats[group]['neutral'] += 1
                except Exception as e:
                    logging.error(f"Error analyzing individual text: {str(e)}")
                    continue

            # Calculate averages and percentages
            total = sentiment_stats[group]['total']
            if total > 0:
                sentiment_stats[group]['avg_compound'] = sum(sentiment_stats[group]['scores']) / total
                sentiment_stats[group]['pos_pct'] = (sentiment_stats[group]['positive'] / total) * 100
                sentiment_stats[group]['neg_pct'] = (sentiment_stats[group]['negative'] / total) * 100
                sentiment_stats[group]['neu_pct'] = (sentiment_stats[group]['neutral'] / total) * 100

        return dict(sentiment_stats)  # Convert defaultdict to regular dict

    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return {}

# Theme Evolution
def calculate_theme_evolution(texts_by_group, num_themes=5, min_freq=3):
    """Calculate the evolution of themes across groups."""
    groups = sorted(texts_by_group.keys())

    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        stop_words='english',
        min_df=min_freq
    )

    themes_by_group = {}
    all_themes = set()

    for group in groups:
        texts = texts_by_group[group]
        if not texts:
            continue

        try:
            # Clean and prepare texts
            cleaned_texts = [
                process_text(text, st.session_state.custom_stopwords, st.session_state.synonyms)
                for text in texts
                if isinstance(text, str) and text.strip()
            ]

            if not cleaned_texts:
                continue

            X = vectorizer.fit_transform(cleaned_texts)
            features = vectorizer.get_feature_names_out()

            frequencies = X.sum(axis=0).A1
            top_indices = np.argsort(-frequencies)[:num_themes]

            group_themes = {
                features[i]: int(frequencies[i])
                for i in top_indices
                if frequencies[i] >= min_freq
            }

            themes_by_group[group] = group_themes
            all_themes.update(group_themes.keys())

        except Exception as e:
            st.warning(f"Error processing themes for group {group}: {str(e)}")
            continue

    evolution_data = {
        'groups': groups,
        'themes': sorted(all_themes),
        'values': []
    }

    for theme in evolution_data['themes']:
        theme_values = []
        for group in groups:
            value = themes_by_group.get(group, {}).get(theme, 0)
            theme_values.append(value)
        evolution_data['values'].append(theme_values)

    return evolution_data

def create_theme_flow_diagram(evolution_data):
    """Create a Sankey diagram showing theme evolution."""
    if not evolution_data['groups'] or not evolution_data['themes']:
        return None

    groups = evolution_data['groups']
    themes = evolution_data['themes']
    values = evolution_data['values']

    nodes = []
    node_labels = []

    for group in groups:
        nodes.append(dict(name=group))
        node_labels.append(group)

    links = []
    colors = px.colors.qualitative.Set3

    for i in range(len(groups) - 1):
        source_group = i
        target_group = i + 1

        for theme_idx, theme in enumerate(themes):
            value = min(values[theme_idx][i], values[theme_idx][i + 1])
            if value > 0:
                links.append(dict(
                    source=source_group,
                    target=target_group,
                    value=value,
                    label=theme,
                    color=colors[theme_idx % len(colors)]
                ))

    if not links:
        return None

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            label=[link['label'] for link in links],
            color=[link['color'] for link in links]
        )
    )])

    fig.update_layout(
        title_text="Theme Evolution Flow",
        font_size=12,
        height=600
    )

    return fig

def create_theme_heatmap(evolution_data):
    """Create a heatmap showing theme intensity across groups."""
    if not evolution_data['groups'] or not evolution_data['themes']:
        return None

    groups = evolution_data['groups']
    themes = evolution_data['themes']
    values = evolution_data['values']

    values_array = np.array(values)
    if values_array.size == 0 or values_array.max() == 0:
        return None

    values_normalized = values_array / values_array.max()

    fig = go.Figure(data=go.Heatmap(
        z=values_normalized,
        x=groups,
        y=themes,
        colorscale='Viridis',
        hoverongaps=False
    ))

    fig.update_layout(
        title='Theme Intensity Across Groups',
        xaxis_title='Groups',
        yaxis_title='Themes',
        height=600
    )

    return fig

# Main app
st.title('📊 Text Analysis Dashboard')
with st.expander("ℹ️ About this dashboard", expanded=False):
    st.write('Analyze open-text responses across all surveys')
    st.markdown("""
    This dashboard provides tools for:
    - Visualizing text data through word clouds
    - Analyzing word frequencies and relationships
    - Discovering topics and themes
    - Analyzing sentiment
    - Tracking theme evolution
    """)

# Initialize settings if needed
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'file_processed': False,
        'chosen_survey': "All",
        'variable': None,
        'group_by': 'None',
        'colormap': 'viridis',
        'highlight_words': set(),
        'search_word': '',
        'num_topics': 4,
        'min_topic_size': 2
    }

# Initialize data containers if needed
if 'data' not in st.session_state:
    st.session_state.data = {
        'question_mapping': None,
        'responses_dict': None,
        'open_var_options': None,
        'grouping_columns': None
    }

# Sidebar configuration
with st.sidebar:
    st.header("Analysis Settings")

    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

    # Process file ONCE if needed
    if uploaded_file and not st.session_state.settings['file_processed']:
        with st.spinner("Processing file..."):
            question_mapping, responses_dict, open_var_options, grouping_columns = load_excel_file(
                uploaded_file,
                st.session_state.settings['chosen_survey']
            )

            if question_mapping is not None and responses_dict is not None:
                # Store data in session state
                st.session_state.data.update({
                    'question_mapping': question_mapping,
                    'responses_dict': responses_dict,
                    'open_var_options': open_var_options,
                    'grouping_columns': grouping_columns
                })
                st.session_state.settings['file_processed'] = True
                st.rerun()

    # Only show settings if file is processed
    if st.session_state.settings['file_processed']:
        st.markdown("---")
        st.subheader("📊 Data Selection")

        # Variable selection
        if st.session_state.data['open_var_options']:
            variable = st.selectbox(
                "Select Variable to Analyze",
                options=list(st.session_state.data['open_var_options'].keys()),
                format_func=lambda x: st.session_state.data['open_var_options'][x],
                help="Variables ending with _open",
                key='variable_select'
            )
            st.session_state.settings['variable'] = variable
        else:
            st.warning("No open-ended variables found")

        # Grouping selection
        if st.session_state.data['grouping_columns']:
            group_by = st.selectbox(
                "Group Responses By",
                options=['None'] + st.session_state.data['grouping_columns'],
                help="Select column to group by",
                key='group_select'
            )
            st.session_state.settings['group_by'] = group_by
        else:
            st.warning("No grouping columns available")

        # Search functionality
        st.markdown("---")
        st.subheader("🔍 Search")
        search_word = st.text_input(
            "Search for a specific word",
            value=st.session_state.settings.get('search_word', ''),
            help="Enter a word to find sample responses containing it",
            key='search_input'
        )
        st.session_state.settings['search_word'] = search_word

        # Stopword management
        st.markdown("---")
        st.subheader("⚙️ Text Processing")
        render_stopwords_management()

        # Synonym management
        add_synonym_management_to_sidebar()

        # Visualization settings
        st.markdown("---")
        st.subheader("🎨 Visualization")

        # Colormap
        colormap = st.selectbox(
            "Color scheme",
            ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            key='colormap_select'
        )
        st.session_state.settings['colormap'] = colormap

        # Highlight words
        highlight_words = st.text_area(
            "Highlight Words (one per line)",
            help="Enter words to highlight in red",
            key='highlight_words_input'
        )
        st.session_state.settings['highlight_words'] = set(
            word.strip().lower()
            for word in highlight_words.split('\n')
            if word.strip()
        )

        # Topic analysis settings
        st.markdown("---")
        st.subheader("🔍 Topic Analysis")

        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=8,
            value=st.session_state.settings['num_topics'],
            key='num_topics_slider'
        )
        st.session_state.settings['num_topics'] = num_topics

        min_topic_size = st.slider(
            "Minimum Topic Size",
            min_value=2,
            max_value=5,
            value=st.session_state.settings['min_topic_size'],
            key='min_topic_size_slider'
        )
        st.session_state.settings['min_topic_size'] = min_topic_size

        # Update button
        st.markdown("---")
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.session_state.settings['file_processed'] = False
            st.rerun()

# Main content area
if st.session_state.settings['file_processed']:
    # Access stored data
    question_mapping = st.session_state.data['question_mapping']
    responses_dict = st.session_state.data['responses_dict']
    variable = st.session_state.settings['variable']
    group_by = st.session_state.settings['group_by']

    # Check for valid data and variable selection
    if (st.session_state.data['question_mapping'] is not None
            and st.session_state.data['responses_dict'] is not None
            and st.session_state.data['open_var_options']
            and variable):

        # Show response counts by survey/group
        st.markdown("### Response Counts")

        # Get responses for the selected variable
        responses_by_survey = get_responses_for_variable(responses_dict, variable, group_by)

        if not responses_by_survey:
            st.warning("No responses found for this variable in any survey")
            st.stop()

        # Display response counts
        total_responses = 0
        for survey_id, responses in responses_by_survey.items():
            response_count = len(responses)
            total_responses += response_count
            st.write(f"{survey_id}: {response_count:,} responses")

        # Show total if multiple surveys
        if len(responses_by_survey) > 1:
            st.write(f"**Total: {total_responses:,} responses**")

        # Initialize session state for tab persistence
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 0

        # Create tabs
        tab_names = [
            "🗂️ Open Coding",
            "🎨 Word Cloud",
            "📊 Word Analysis",
            "🔍 Topic Discovery",
            "❤️ Sentiment Analysis",
            "🌊 Theme Evolution"
        ]

        tabs = st.tabs(tab_names)

        # Open Coding
        with tabs[0]:
            render_open_coding_tab(
                variable=variable,
                responses_dict=responses_dict,
                open_var_options = st.session_state.data['open_var_options'],
                grouping_columns=grouping_columns
            )

        # Word Cloud Tab
        with tabs[1]:
            with st.expander("🎨 About Word Cloud Visualization", expanded=False):
                st.markdown("""
                Word clouds provide an intuitive visual representation of your text data where:
                - **Larger words** appear more frequently in your responses
                - **Colors** help distinguish between different words
                - **Positioning** is optimized for visual appeal

                **Key Features:**
                - 🔄 Interactive and static visualization options
                - 📊 Frequency analysis and hover details
                - 🎯 Word highlighting capabilities
                - 🔍 Multiple comparison views
                - 💾 Downloadable high-quality images
                """)

            # Display full question
            st.markdown("""
            <style>
                .question-box {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    margin: 10px 0;
                }
                .question-label {
                    color: #2E7D32;
                    font-size: 1.2em;
                    margin-bottom: 10px;
                    font-weight: bold;
                }
                .question-text {
                    color: #1a1a1a;
                    font-size: 1.1em;
                    line-height: 1.5;
                    font-weight: 600;
                }
            </style>
            """, unsafe_allow_html=True)

            open_var_dict = st.session_state.data.get('open_var_options', {})

            st.markdown(f"""
            <div class="question-box">
                <div class="question-label">Primary Question</div>
                <div class="question-text">{open_var_dict.get(variable, "No question text")}</div>
            </div>
            """, unsafe_allow_html=True)

            # Common settings in a clean format
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    viz_type = st.radio(
                        "📊 Select visualization type",
                        ["Single Wordcloud", "Multi-group Comparison (2x2)",
                         "Side-by-side Comparison", "Synonym Groups"],
                        help="Choose how to display your wordclouds"
                    )
                with col2:
                    wordcloud_title = st.text_input("📝 Title (optional)", "")
                    subtitle = st.text_input("✏️ Subtitle (optional)", "")
                    source_text = st.text_input("📚 Source text (optional)", "")


            def add_download_buttons(fig_or_wc, word_freq, prefix, survey_id=""):
                """Helper function to add download buttons for PNG and CSV"""
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Image download button
                    buffer = BytesIO()
                    if isinstance(fig_or_wc, plt.Figure):
                        fig_or_wc.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    else:  # WordCloud object
                        fig_or_wc.to_image().save(buffer, format='PNG', optimize=True)
                    buffer.seek(0)

                    st.download_button(
                        label="💾 Download High-Quality Image",
                        data=buffer,
                        file_name=f"{prefix}_{survey_id}.png" if survey_id else f"{prefix}.png",
                        mime="image/png",
                        use_container_width=True
                    )

                    # CSV download button for word frequencies
                    if word_freq:
                        freq_df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
                        csv_data = freq_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Download Word Frequencies (CSV)",
                            data=csv_data,
                            file_name=f"{prefix}_freq{'_' + survey_id if survey_id else ''}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


            if viz_type == "Single Wordcloud":
                for survey_id, responses in responses_by_survey.items():
                    if responses:
                        st.markdown(f"### 📊 Analysis for {survey_id}")

                        # Debug information in a collapsible section
                        with st.expander("ℹ️ Response Statistics", expanded=False):
                            st.write(f"Total responses: {len(responses)}")
                            valid_responses = [r for r in responses if isinstance(r, str) and r.strip()]
                            st.write(f"Valid responses: {len(valid_responses)}")
                            st.write(f"Processed stopwords: {len(st.session_state.custom_stopwords)}")

                        try:
                            # Clean stopwords
                            clean_stopwords = {str(word).lower() for word in st.session_state.custom_stopwords
                                               if word is not None and not isinstance(word, float)}

                            # Generate wordclouds
                            static_wc, interactive_wc, word_freq = generate_wordcloud(
                                valid_responses,
                                clean_stopwords,
                                st.session_state.synonym_groups,
                                colormap=colormap,
                                highlight_words=highlight_words,
                                return_freq=True
                            )

                            if interactive_wc and static_wc:
                                # Create tabs for different views
                                view_tabs = st.tabs([
                                    "📸 Static View",
                                    "🔄 Interactive View",
                                    "📊 Frequency Analysis"
                                ])

                                # Static View Tab
                                with view_tabs[0]:
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.imshow(static_wc)
                                    ax.axis('off')
                                    if wordcloud_title:
                                        plt.title(f"{wordcloud_title} - {survey_id}", fontsize=16, pad=20)

                                    st.pyplot(fig)
                                    plt.close()

                                    add_download_buttons(static_wc, word_freq, "wordcloud", survey_id)

                                # Interactive View Tab
                                with view_tabs[1]:
                                    st.plotly_chart(interactive_wc, use_container_width=True)

                                    with st.expander("ℹ️ Interactive Features", expanded=False):
                                        st.markdown("""
                                        - 🖱️ Hover over words to see frequencies
                                        - 🔍 Zoom in/out using mouse wheel
                                        - 🖐️ Pan by clicking and dragging
                                        - 📱 Pinch to zoom on touch devices
                                        """)

                                # Frequency Analysis Tab
                                with view_tabs[2]:
                                    freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.dataframe(
                                            freq_df.style.background_gradient(
                                                subset=['Frequency'],
                                                cmap=colormap
                                            ),
                                            use_container_width=True
                                        )
                                    with col2:
                                        st.metric(
                                            "Most Frequent Word",
                                            freq_df.iloc[0]['Word'],
                                            f"Count: {freq_df.iloc[0]['Frequency']}"
                                        )

                        except Exception as e:
                            st.error(f"Error generating wordcloud: {str(e)}")
                            st.error(f"Detailed error: {traceback.format_exc()}")

            elif viz_type == "Multi-group Comparison (2x2)":
                if group_by:
                    # Group selection
                    all_groups = set()
                    for df in responses_dict.values():
                        if group_by in df.columns:
                            all_groups.update(df[group_by].dropna().unique())

                    selected_groups = st.multiselect(
                        "🎯 Select groups to compare (max 4)",
                        options=sorted(all_groups),
                        max_selections=4
                    )

                    if selected_groups:
                        # Process texts by group
                        texts_by_group = {}
                        for group in selected_groups:
                            group_responses = []
                            for df in responses_dict.values():
                                if group_by in df.columns and variable in df.columns:
                                    group_texts = df[df[group_by] == group][variable].dropna().tolist()
                                    group_responses.extend(group_texts)
                            texts_by_group[str(group)] = group_responses

                        try:
                            # Generate both versions and word frequencies
                            static_fig, interactive_fig, group_freqs = generate_multi_group_wordcloud(
                                texts_by_group,
                                stopwords=st.session_state.custom_stopwords,
                                synonyms=st.session_state.synonym_groups,
                                colormap=colormap,
                                highlight_words=highlight_words,
                                main_title=wordcloud_title,
                                subtitle=subtitle,
                                source_text=source_text,
                                return_freq=True
                            )

                            view_tabs = st.tabs([
                                "📸 Static View",
                                "🔄 Interactive View",
                                "📊 Group Statistics"
                            ])

                            with view_tabs[0]:
                                st.pyplot(static_fig)
                                plt.close()

                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    add_download_buttons(static_fig, None, "multigroup_comparison")

                                    # CSV downloads for each group
                                    for group, freq in group_freqs.items():
                                        freq_df = pd.DataFrame(freq, columns=['word', 'frequency'])
                                        csv_data = freq_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label=f"📊 Download {group} Frequencies (CSV)",
                                            data=csv_data,
                                            file_name=f"multigroup_{group}_freq.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )

                            with view_tabs[1]:
                                st.plotly_chart(interactive_fig, use_container_width=True)

                                with st.expander("ℹ️ Interactive Features", expanded=False):
                                    st.markdown("""
                                        - 🖱️ Hover over words to see frequencies
                                        - 🔍 Use buttons to zoom in/out
                                        - 🎯 Click words to highlight
                                        - 📱 Drag to pan view
                                        """)

                            with view_tabs[2]:
                                for group, texts in texts_by_group.items():
                                    with st.expander(f"📊 {group} Statistics", expanded=True):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Total Responses", len(texts))
                                        with col2:
                                            st.metric("Unique Words",
                                                      len(set(' '.join(texts).split())))

                                        if group in group_freqs:
                                            freq_df = pd.DataFrame(group_freqs[group], columns=['Word', 'Frequency'])
                                            st.dataframe(
                                                freq_df.style.background_gradient(
                                                    subset=['Frequency'],
                                                    cmap=colormap
                                                ),
                                                use_container_width=True
                                            )

                        except Exception as e:
                            st.error(f"Error generating multi-group wordcloud: {str(e)}")

                else:
                    st.warning("⚠️ Please select a grouping variable to use multi-group comparison")

            elif viz_type == "Side-by-side Comparison":
                if group_by:
                    all_groups = set()
                    for df in responses_dict.values():
                        if group_by in df.columns:
                            all_groups.update(df[group_by].dropna().unique())

                    invalid_categories = {'dk', 'na', 'n/a', 'none', 'dk/ref', 'ref', 'refused'}
                    valid_groups = {str(group).strip() for group in all_groups
                                    if str(group).strip().lower() not in invalid_categories}

                    col1, col2 = st.columns(2)
                    with col1:
                        categories_left = st.multiselect(
                            "Left side categories",
                            options=sorted(valid_groups),
                            key="left_categories"
                        )
                    with col2:
                        categories_right = st.multiselect(
                            "Right side categories",
                            options=sorted(valid_groups),
                            key="right_categories"
                        )

                    if categories_left and categories_right:
                        texts_by_group = {}
                        for group in set(categories_left + categories_right):
                            group_responses = []
                            for df in responses_dict.values():
                                if group_by in df.columns and variable in df.columns:
                                    group_texts = df[df[group_by] == group][variable].dropna().tolist()
                                    group_responses.extend(group_texts)
                            texts_by_group[str(group)] = group_responses

                        try:
                            static_fig, interactive_fig, side_freqs = generate_combined_comparison_wordcloud(
                                texts_by_group,
                                categories_left,
                                categories_right,
                                stopwords=st.session_state.custom_stopwords,
                                synonyms=st.session_state.synonym_groups,
                                colormap=colormap,
                                highlight_words=highlight_words,
                                return_freq=True
                            )

                            if static_fig and interactive_fig:
                                view_tabs = st.tabs([
                                    "📸 Static View",
                                    "🔄 Interactive View",
                                    "📊 Frequency Analysis"
                                ])

                                with view_tabs[0]:
                                    st.pyplot(static_fig, use_container_width=True)
                                    plt.close()

                                    add_download_buttons(static_fig, None, "sidebyside_comparison")

                                    # CSV downloads
                                    for side, freq in side_freqs.items():
                                        freq_df = pd.DataFrame(freq, columns=['word', 'frequency'])
                                        csv_data = freq_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label=f"📊 Download {side} Side Frequencies (CSV)",
                                            data=csv_data,
                                            file_name=f"sidebyside_{side}_freq.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )

                                with view_tabs[1]:
                                    st.plotly_chart(interactive_fig, use_container_width=True)

                                with view_tabs[2]:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.header("Left Side")
                                        left_freq_df = pd.DataFrame(side_freqs['left'], columns=['Word', 'Frequency'])
                                        st.dataframe(
                                            left_freq_df.style.background_gradient(
                                                subset=['Frequency'],
                                                cmap=colormap
                                            ),
                                            use_container_width=True
                                        )
                                    with col2:
                                        st.header("Right Side")
                                        right_freq_df = pd.DataFrame(side_freqs['right'], columns=['Word', 'Frequency'])
                                        st.dataframe(
                                            right_freq_df.style.background_gradient(
                                                subset=['Frequency'],
                                                cmap=colormap
                                            ),
                                            use_container_width=True
                                        )

                        except Exception as e:
                            st.error(f"Error generating comparison wordcloud: {str(e)}")

                    else:
                        st.info("👆 Please select categories for both sides")

                else:
                    st.warning("⚠️ Please select a grouping variable for comparison")

            else:  # Synonym Groups
                if st.session_state.synonym_groups:
                    st.subheader("🔤 Wordclouds by Synonym Groups")

                    col1, col2 = st.columns(2)
                    with col1:
                        selected_groups = st.multiselect(
                            "Select synonym groups to visualize",
                            options=sorted(st.session_state.synonym_groups.keys()),
                            default=list(st.session_state.synonym_groups.keys())[:4]
                        )

                    with col2:
                        separate_clouds = st.checkbox(
                            "Generate separate wordclouds",
                            value=True,
                            help="Create individual wordclouds for each synonym group"
                        )

                    if selected_groups:
                        selected_synonym_groups = {
                            group: st.session_state.synonym_groups[group]
                            for group in selected_groups
                        }

                        try:
                            if separate_clouds:
                                static_fig, interactive_fig, group_freqs = generate_synonym_group_wordclouds(
                                    responses,
                                    st.session_state.custom_stopwords,
                                    selected_synonym_groups,
                                    colormap=colormap,
                                    return_freq=True
                                )

                                view_tabs = st.tabs([
                                    "📸 Static View",
                                    "🔄 Interactive View",
                                    "📊 Synonym Analysis"
                                ])

                                with view_tabs[0]:
                                    st.pyplot(static_fig)
                                    plt.close()

                                    add_download_buttons(static_fig, None, "synonym_groups")

                                    # CSV downloads for each synonym group
                                    for group, freq in group_freqs.items():
                                        freq_df = pd.DataFrame(freq, columns=['word', 'frequency'])
                                        csv_data = freq_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label=f"📊 Download {group} Frequencies (CSV)",
                                            data=csv_data,
                                            file_name=f"synonym_{group}_freq.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )

                                with view_tabs[1]:
                                    st.plotly_chart(interactive_fig, use_container_width=True)

                                with view_tabs[2]:
                                    for group, synonyms in selected_synonym_groups.items():
                                        with st.expander(f"📊 {group} Statistics", expanded=True):
                                            st.write("Synonyms:", ", ".join(synonyms))
                                            count = sum(1 for text in responses
                                                        for syn in synonyms
                                                        if isinstance(text, str) and syn.lower() in text.lower())
                                            st.metric("Total Occurrences", count)

                                            if group in group_freqs:
                                                freq_df = pd.DataFrame(group_freqs[group],
                                                                       columns=['Word', 'Frequency'])
                                                st.dataframe(
                                                    freq_df.style.background_gradient(
                                                        subset=['Frequency'],
                                                        cmap=colormap
                                                    ),
                                                    use_container_width=True
                                                )

                            else:
                                # Combined wordcloud for all synonym groups
                                static_wc, interactive_wc, combined_freq = generate_wordcloud(
                                    responses,
                                    st.session_state.custom_stopwords,
                                    selected_synonym_groups,
                                    colormap=colormap,
                                    highlight_words=highlight_words,
                                    return_freq=True
                                )

                                view_tabs = st.tabs([
                                    "🔄 Interactive View",
                                    "📸 Static View",
                                    "📊 Frequency Analysis"
                                ])

                                with view_tabs[0]:
                                    st.plotly_chart(interactive_wc, use_container_width=True)

                                with view_tabs[1]:
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.imshow(static_wc)
                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close()

                                    add_download_buttons(static_wc, combined_freq, "combined_synonyms")

                                with view_tabs[2]:
                                    freq_df = pd.DataFrame(combined_freq, columns=['Word', 'Frequency'])
                                    st.dataframe(
                                        freq_df.style.background_gradient(
                                            subset=['Frequency'],
                                            cmap=colormap
                                        ),
                                        use_container_width=True
                                    )

                        except Exception as e:
                            st.error(f"Error generating synonym group wordclouds: {str(e)}")
                            st.error(f"Detailed error: {traceback.format_exc()}")
                else:
                    st.warning(
                        "⚠️ No synonym groups defined. Add synonym groups in the sidebar to use this visualization.")

        # Word Analysis Tab
        with tabs[2]:
            with st.expander("📊 About Word Analysis", expanded=False):
                st.markdown("""
                This analysis provides detailed insights into word usage patterns through:
                - **Frequency charts** showing most common terms
                - **Co-occurrence networks** revealing word relationships
                - **Statistical analysis** of word patterns

                **Available Visualizations:**
                1. 📈 Word frequency distribution
                2. 🕸️ Word co-occurrence network
                3. 📋 Detailed frequency tables
                """)

            # Word frequency and co-occurrence analysis by survey
            for survey_id, responses in responses_by_survey.items():
                st.write(f"### Analysis for {survey_id}")

                processed_texts = [
                    process_text(text, st.session_state.custom_stopwords, st.session_state.synonyms)
                    for text in responses
                    if text.strip()
                ]

                if processed_texts:
                    try:
                        # Frequency Analysis Section
                        st.subheader("📈 Word Frequencies")
                        vectorizer = CountVectorizer(
                            max_features=20,
                            stop_words=list(st.session_state.custom_stopwords)
                        )

                        X = vectorizer.fit_transform(processed_texts)
                        words = vectorizer.get_feature_names_out()
                        frequencies = X.sum(axis=0).A1

                        freq_df = pd.DataFrame({
                            'word': words,
                            'frequency': frequencies
                        }).sort_values('frequency', ascending=False)

                        fig = px.bar(
                            freq_df,
                            x='word',
                            y='frequency',
                            title='Word Frequency Distribution',
                            labels={'word': 'Word', 'frequency': 'Frequency'},
                            color='frequency',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_traces(texttemplate='%{y}', textposition='outside')
                        safe_plotly_chart(fig, st, "Unable to display word frequency chart")

                        # Network Analysis Section
                        st.markdown("---")
                        st.subheader("🕸️ Word Co-occurrence Network")

                        col1, col2 = st.columns(2)
                        with col1:
                            # Slider for Minimum Co-occurrence Threshold with unique key
                            min_edge_weight = st.slider(
                                "Minimum co-occurrence threshold",
                                min_value=1,
                                max_value=10,
                                value=2,
                                help="Minimum number of times words must appear together",
                                key=f"min_edge_weight_{survey_id}"
                            )

                            # Slider for Maximum Number of Words with unique key
                            max_words = st.slider(
                                "Maximum number of words",
                                min_value=10,
                                max_value=100,
                                value=30,
                                help="Maximum number of words to include in the network",
                                key=f"max_words_{survey_id}"
                            )

                        network_fig = create_word_cooccurrence_network(
                            processed_texts,
                            min_edge_weight=min_edge_weight,
                            max_words=max_words,
                            stopwords=st.session_state.custom_stopwords
                        )
                        safe_plotly_chart(network_fig, st, "Unable to display word co-occurrence network")

                        # Co-occurrence table
                        st.markdown("---")
                        st.subheader("📋 Top Co-occurring Word Pairs")
                        cooc_df = get_top_cooccurrences(
                            processed_texts,
                            n_words=max_words,
                            n_pairs=15,
                            stopwords=st.session_state.custom_stopwords
                        )
                        st.dataframe(
                            cooc_df,
                            use_container_width=True,
                            column_config={
                                "word1": "Word 1",
                                "word2": "Word 2",
                                "cooccurrences": st.column_config.NumberColumn(
                                    "Co-occurrences",
                                    help="Number of times these words appear together"
                                )
                            }
                        )

                    except Exception as e:
                        st.error(f"Error in analysis for {survey_id}: {e}")
                else:
                    st.warning(f"No valid texts found for analysis in {survey_id}")

        # Topic Discovery Tab
        with tabs[3]:
            with st.expander("🔍 About Topic Discovery", expanded=False):
                st.markdown("""
                Topic modeling uses advanced machine learning to uncover hidden themes in your text:

                **What it does:**
                - 🎯 Identifies main topics automatically
                - 🔄 Groups similar responses together
                - 📍 Maps relationships between topics
                - 📊 Quantifies topic importance

                **Key Features:**
                1. Interactive Topic Distribution
                2. Top Terms Analysis
                3. Sample Response Review
                4. Topic Size Visualization
                """)

            # Topic modeling by survey
            for survey_id, responses in responses_by_survey.items():
                st.write(f"### Topics for {survey_id}")

                processed_texts = [
                    process_text(text, st.session_state.custom_stopwords, st.session_state.synonyms)
                    for text in responses if isinstance(text, str) and text.strip()
                ]

                # Only proceed if we have enough valid responses
                if len(processed_texts) >= 20:  # Minimum threshold
                    try:
                        # Topic modeling settings adjusted for smaller datasets
                        col1, col2 = st.columns(2)
                        with col1:
                            num_topics = st.slider(
                                "Number of Topics",
                                min_value=2,
                                max_value=min(8, len(processed_texts) // 25),  # Conservative max topics
                                value=min(4, len(processed_texts) // 25),  # Conservative default
                                key=f"num_topics_{survey_id}"
                            )
                        with col2:
                            min_topic_size = st.slider(
                                "Minimum Topic Size",
                                min_value=2,
                                max_value=5,
                                value=2,
                                key=f"min_topic_size_{survey_id}"
                            )

                        # Create a simplified topic model configuration
                        @st.cache_resource
                        def get_sentence_transformer():
                            return SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model


                        # Get embeddings
                        with st.spinner("Computing text embeddings..."):
                            model = get_sentence_transformer()
                            embeddings = model.encode(processed_texts, show_progress_bar=False)

                        # Perform clustering
                        with st.spinner("Identifying topics..."):
                            kmeans = KMeans(
                                n_clusters=num_topics,
                                random_state=42
                            ).fit(embeddings)

                        # Organize texts by cluster
                        texts_by_cluster = defaultdict(list)
                        for text, label in zip(processed_texts, kmeans.labels_):
                            texts_by_cluster[label].append(text)

                        # Get top terms for each cluster using CountVectorizer
                        vectorizer = CountVectorizer(
                            max_features=10,
                            stop_words=list(st.session_state.custom_stopwords)
                        )

                        topic_info = []
                        topics = kmeans.labels_

                        for topic_id in range(num_topics):
                            cluster_texts = texts_by_cluster[topic_id]
                            if cluster_texts:
                                # Get top terms
                                try:
                                    X = vectorizer.fit_transform(cluster_texts)
                                    words = vectorizer.get_feature_names_out()
                                    freqs = X.sum(axis=0).A1
                                    top_words = [(word, freq) for word, freq in zip(words, freqs)]
                                    top_words.sort(key=lambda x: x[1], reverse=True)
                                except:
                                    top_words = [("", 0)]  # Fallback if vectorization fails

                                # Add topic info
                                topic_info.append({
                                    'Topic': topic_id,
                                    'Count': len(cluster_texts),
                                    'Name': f"Topic {topic_id}",
                                    'Top_Terms': top_words[:5]
                                })

                        topic_info = pd.DataFrame(topic_info)

                        # Create visualization tabs
                        topic_tabs = st.tabs(["Overview", "Details", "Samples"])

                        with topic_tabs[0]:
                            # Create treemap visualization
                            topic_viz_data = {
                                'Topic': [],
                                'Size': [],
                                'Terms': []
                            }

                            for _, row in topic_info.iterrows():
                                topic_viz_data['Topic'].append(f"Topic {row['Topic']}")
                                topic_viz_data['Size'].append(row['Count'])
                                topic_viz_data['Terms'].append(
                                    ", ".join([term for term, _ in row['Top_Terms'][:3]]))

                            fig = px.treemap(
                                topic_viz_data,
                                path=['Topic'],
                                values='Size',
                                hover_data=['Terms'],
                                title='Topic Distribution'
                            )
                            safe_plotly_chart(fig, st, "Unable to display topic distribution")

                            # Add topic size distribution
                            sizes_fig = px.bar(
                                topic_info,
                                x='Topic',
                                y='Count',
                                title='Topic Sizes',
                                labels={'Count': 'Number of Responses', 'Topic': 'Topic ID'}
                            )
                            safe_plotly_chart(sizes_fig, st, "Unable to display topic sizes")

                        with topic_tabs[1]:
                            # Display detailed topic information
                            for idx, row in topic_info.iterrows():
                                with st.expander(f"Topic {row['Topic']} ({row['Count']} responses)"):
                                    # Display top terms
                                    st.write("Top terms:")
                                    terms_df = pd.DataFrame(row['Top_Terms'], columns=['Term', 'Frequency'])

                                    # Create bar chart for term frequencies
                                    term_fig = px.bar(
                                        terms_df,
                                        x='Frequency',
                                        y='Term',
                                        orientation='h',
                                        title=f'Term Frequencies for Topic {row["Topic"]}'
                                    )
                                    safe_plotly_chart(term_fig, st, "Unable to display term frequencies")

                        with topic_tabs[2]:
                            # Display sample responses for each topic
                            selected_topic = st.selectbox(
                                "Select Topic to View Samples",
                                options=topic_info['Topic'].tolist(),
                                format_func=lambda x: f"Topic {x} ({len(texts_by_cluster[x])} responses)"
                            )

                            num_samples = st.slider(
                                "Number of samples to display",
                                min_value=1,
                                max_value=min(10, len(texts_by_cluster[selected_topic])),
                                value=3
                            )

                            st.write(f"### Sample Responses for Topic {selected_topic}")
                            sample_texts = texts_by_cluster[selected_topic][:num_samples]
                            for i, text in enumerate(sample_texts, 1):
                                with st.expander(f"Response {i}", expanded=True):
                                    st.write(text)

                    except Exception as e:
                        st.error(f"Error in topic modeling for {survey_id}: {str(e)}")
                        st.info("""
                            Troubleshooting tips:
                            - Try reducing the number of topics
                            - Make sure you have enough valid responses
                            - Check for very short or empty responses
                        """)
                else:
                    st.warning(
                        f"Not enough valid responses in {survey_id} for topic modeling (minimum 20 required)")

        # Sentiment Analysis Tab
        with tabs[4]:
            with st.expander("❤️ About Sentiment Analysis", expanded=False):
                st.markdown("""
                Analyze the emotional tone and attitude in responses:

                **Key Metrics:**
                - 😊 Positive sentiment score
                - 😐 Neutral sentiment detection
                - 😔 Negative sentiment identification
                - 📊 Overall sentiment distribution

                **Visualizations:**
                1. Sentiment Flow Analysis
                2. Comparative Radar Chart
                3. Sentiment Distribution Sunburst
                """)

            st.markdown("### Sentiment Analysis Results")

            if not group_by:
                st.warning("Please select a grouping variable to compare sentiments across groups")
            else:
                # Get texts by group
                texts_by_group = {}
                for df in responses_dict.values():
                    if group_by in df.columns and variable in df.columns:
                        groups = df[group_by].dropna().unique()
                        for group in groups:
                            group_texts = df[df[group_by] == group][variable].dropna().tolist()
                            if str(group) not in texts_by_group:
                                texts_by_group[str(group)] = []
                            texts_by_group[str(group)].extend(group_texts)

                # Analyze sentiment
                sentiment_stats = analyze_group_sentiment(texts_by_group)

                # Create tabs for different visualizations
                viz_tabs = st.tabs(["📊 Distribution", "📡 Radar", "🌟 Sunburst"])

                with viz_tabs[0]:
                    safe_plotly_chart(
                        create_sentiment_distribution(sentiment_stats),
                        st,
                        "Unable to display sentiment distribution"
                    )

                    st.markdown("""
                                **Understanding the Distribution:**
                                - Shows full range of sentiment scores
                                - Box shows 25th-75th percentile
                                - Line shows median
                                - Points show outliers
                                - Compare distributions across groups
                                """)

                # Add detailed statistics
                st.markdown("### 📊 Detailed Sentiment Statistics")

                # Create a DataFrame for the statistics
                stats_data = []
                for group, stats in sentiment_stats.items():
                    stats_data.append({
                        'Group': group,
                        'Total Responses': stats['total'],
                        'Positive %': f"{stats['pos_pct']:.1f}%",
                        'Neutral %': f"{stats['neu_pct']:.1f}%",
                        'Negative %': f"{stats['neg_pct']:.1f}%",
                        'Average Sentiment': f"{stats['avg_compound']:.3f}"
                    })

                stats_df = pd.DataFrame(stats_data)

                # Only apply styling if the dataframe has data and the required column exists
                if not stats_df.empty and 'Average Sentiment' in stats_df.columns:
                    styled_df = stats_df.style.background_gradient(
                        subset=['Average Sentiment'],
                        cmap='RdYlGn',
                        vmin=-1,
                        vmax=1
                    )
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    # Fallback to displaying the plain dataframe
                    st.dataframe(stats_df, use_container_width=True)

                with viz_tabs[1]:
                    safe_plotly_chart(
                        create_sentiment_radar(sentiment_stats),
                        st,
                        "Unable to display sentiment radar chart"
                    )

                    st.markdown("""
                                **Understanding the Radar Chart:**
                                - Each axis represents a sentiment metric
                                - Larger area = more positive overall
                                - Compare patterns between groups
                                - Hover for exact values
                                """)

                with viz_tabs[2]:
                    sunburst_fig = create_sentiment_sunburst(sentiment_stats)
                    safe_plotly_chart(sunburst_fig, st, "Unable to display sentiment sunburst chart")
                    if sunburst_fig is not None:
                        try:
                            st.plotly_chart(sunburst_fig, use_container_width=True)

                            st.markdown("""
                                **Understanding the Sunburst Chart:**
                                - Inner circle shows total responses per group
                                - Outer ring shows sentiment distribution
                                - 🟢 Green = Positive
                                - 🟡 Yellow = Neutral
                                - 🔴 Red = Negative
                                - Hover for detailed percentages
                                """)
                        except Exception as e:
                            st.warning("Unable to display sentiment visualization due to insufficient data")
                    else:
                        st.warning("Not enough data to generate sentiment visualization")

        # Theme Evolution Tab
        with tabs[5]:
            with st.expander("🌊 About Theme Evolution", expanded=False):
                st.markdown("""
                Track how themes and topics evolve across groups or time periods:

                **Features:**
                - 🔄 Theme flow visualization
                - 🌡️ Theme intensity tracking
                - 📈 Evolution patterns
                - 🔍 Detailed theme analysis

                **Available Views:**
                1. Sankey Flow Diagram
                2. Theme Intensity Heatmap
                3. Comparative Analysis
                """)

            if not group_by:
                st.warning("Please select a grouping variable to analyze theme evolution")
            else:
                # Get texts by group
                texts_by_group = {}
                for df in responses_dict.values():
                    if group_by in df.columns and variable in df.columns:
                        groups = df[group_by].dropna().unique()
                        for group in groups:
                            group_texts = df[df[group_by] == group][variable].dropna().tolist()
                            if str(group) not in texts_by_group:
                                texts_by_group[str(group)] = []
                            texts_by_group[str(group)].extend(group_texts)

                # Analysis settings
                col1, col2 = st.columns(2)
                with col1:
                    num_themes = st.slider(
                        "Number of themes to track",
                        min_value=3,
                        max_value=10,
                        value=5,
                        help="Maximum number of themes to track"
                    )
                with col2:
                    min_freq = st.slider(
                        "Minimum theme frequency",
                        min_value=2,
                        max_value=10,
                        value=3,
                        help="Minimum occurrences required for a theme"
                    )

                try:
                    # Calculate theme evolution
                    with st.spinner("Analyzing theme evolution..."):
                        evolution_data = calculate_theme_evolution(
                            texts_by_group,
                            num_themes=num_themes,
                            min_freq=min_freq
                        )

                    if evolution_data and evolution_data['themes']:
                        # Create visualizations in tabs
                        viz_tabs = st.tabs(["🔄 Flow Diagram", "🌡️ Heat Map"])

                        with viz_tabs[0]:
                            flow_fig = create_theme_flow_diagram(evolution_data)
                            if flow_fig:
                                st.plotly_chart(flow_fig, use_container_width=True)

                                st.markdown("""
                                            **Understanding the Flow Diagram:**
                                            - Columns represent groups/time periods
                                            - Flows show theme continuation
                                            - Width indicates theme strength
                                            - Colors distinguish themes
                                            - Hover for details
                                            """)
                            else:
                                st.warning("Not enough data to create flow diagram")

                        with viz_tabs[1]:
                            heat_fig = create_theme_heatmap(evolution_data)
                            if heat_fig:
                                st.plotly_chart(heat_fig, use_container_width=True)

                                st.markdown("""
                                            **Understanding the Heat Map:**
                                            - Rows show themes
                                            - Columns show groups
                                            - Color intensity = theme strength
                                            - Track theme prevalence
                                            - Compare across groups
                                            """)
                            else:
                                st.warning("Not enough data to create heatmap")

                        # Display theme details
                        st.markdown("### Theme Details")
                        theme_df = pd.DataFrame({
                            'Theme': evolution_data['themes'],
                            'Average Frequency': [np.mean(values) for values in
                                                  evolution_data['values']],
                            'Max Frequency': [np.max(values) for values in evolution_data['values']],
                            'Groups Present': [sum(1 for v in values if v > 0) for values in
                                               evolution_data['values']]
                        })
                        theme_df = theme_df.sort_values('Average Frequency', ascending=False)
                        st.dataframe(
                            theme_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning(
                            "No significant themes found across groups. Try adjusting the minimum frequency or number of themes.")
                except Exception as e:
                    st.error(f"Error analyzing themes: {str(e)}")
                    st.info("Try adjusting the analysis settings or check your data format.")

        # Sample Responses
        st.markdown("---")
        st.markdown("## Response Examples")

        # 1) Organize responses+metadata by group
        #    Instead of just storing text, store full info in a dict
        texts_by_group = {"All": []}

        # Go through each DataFrame, collecting rows that have the open-ended var
        for df in responses_dict.values():
            if variable in df.columns:
                # Build a list of dictionaries:
                # { "text": the open-ended response,
                #   "id": maybe df["id"],
                #   "jobtitle": maybe df["jobtitle"],
                #   "age": maybe df["age"],
                #   "province": maybe df["province"] or df["state"] if present, etc. }
                for idx, row in df.iterrows():
                    raw_text = row[variable]
                    if pd.notna(raw_text) and str(raw_text).strip() and str(raw_text).lower() not in {'nan', 'none',
                                                                                                      'n/a', 'na'}:
                        sample_dict = {
                            "text": str(raw_text).strip(),
                            "id": row["id"] if "id" in df.columns else None,
                            "jobtitle": row["jobtitle"] if "jobtitle" in df.columns else None,
                            "age": row["age"] if "age" in df.columns else None
                        }
                        # If you have "province" or "state" columns, try to store one of them
                        if "province" in df.columns and pd.notna(row["province"]):
                            sample_dict["province"] = row["province"]
                        elif "state" in df.columns and pd.notna(row["state"]):
                            sample_dict["province"] = row["state"]  # use "province" key for convenience
                        else:
                            sample_dict["province"] = None

                        # Put it in the "All" bucket
                        texts_by_group["All"].append(sample_dict)

        # Then handle grouping if requested
        if group_by:
            for df in responses_dict.values():
                if group_by in df.columns and variable in df.columns:
                    unique_groups = df[group_by].unique()
                    for group_val in unique_groups:
                        if pd.isna(group_val):
                            group_key = 'No Group'
                        else:
                            group_key = str(group_val)
                        # Filter matching rows
                        subset_df = df[df[group_by] == group_val].dropna(subset=[variable])
                        for idx, row in subset_df.iterrows():
                            raw_text = row[variable]
                            if pd.notna(raw_text) and str(raw_text).strip() and str(raw_text).lower() not in {'nan',
                                                                                                              'none',
                                                                                                              'n/a',
                                                                                                              'na'}:
                                sample_dict = {
                                    "text": str(raw_text).strip(),
                                    "id": row["id"] if "id" in df.columns else None,
                                    "jobtitle": row["jobtitle"] if "jobtitle" in df.columns else None,
                                    "age": row["age"] if "age" in df.columns else None
                                }
                                # Province or state
                                if "province" in df.columns and pd.notna(row["province"]):
                                    sample_dict["province"] = row["province"]
                                elif "state" in df.columns and pd.notna(row["state"]):
                                    sample_dict["province"] = row["state"]
                                else:
                                    sample_dict["province"] = None

                                # Add to group
                                if group_key not in texts_by_group:
                                    texts_by_group[group_key] = []
                                texts_by_group[group_key].append(sample_dict)

        # Remove duplicates while preserving order (based on "text")
        #   If you'd prefer a row-level dedup, you'd compare {text, id, etc.}
        for group_name, dict_list in texts_by_group.items():
            seen_texts = set()
            unique_list = []
            for d in dict_list:
                if d["text"] not in seen_texts:
                    seen_texts.add(d["text"])
                    unique_list.append(d)
            texts_by_group[group_name] = unique_list

        # If there's a search word, display matching responses
        if search_word:
            display_word_search_results(
                {
                    group: [d["text"] for d in dicts]
                    for group, dicts in texts_by_group.items()
                },
                search_word
            )
        else:
            # Display random samples if no search word
            if st.button("🔄 Generate New Random Samples"):
                st.session_state.sample_seed = int(time.time())

            st.markdown("### Assign to Groups Directly from Samples")

            group_tabs = st.tabs(list(texts_by_group.keys()))

            for tab, group_name in zip(group_tabs, texts_by_group.keys()):
                with tab:
                    dict_list = texts_by_group[group_name]
                    if dict_list:
                        # Filter valid (redundant check, but just in case)
                        valid_list = [d for d in dict_list if d["text"]]
                        if valid_list:
                            n = min(5, len(valid_list))  # default 5 samples
                            if st.session_state.get('sample_seed') is not None:
                                np.random.seed(st.session_state.sample_seed)
                            sample_dicts = np.random.choice(valid_list, size=n, replace=False)

                            st.write(f"Showing {n} of {len(valid_list)} responses for {group_name}")

                            for i, sample_obj in enumerate(sample_dicts, 1):
                                with st.expander(f"Response {i}", expanded=True):
                                    # Build italic line:
                                    # ID, Age, JobTitle, Province
                                    # e.g.: *ID: 1234 | Age: 29 | Job Title: Engineer | Province: Ontario*
                                    # Omit province if None
                                    id_str = f"ID: {sample_obj['id']}" if sample_obj['id'] else ""
                                    age_str = f"Age: {sample_obj['age']}" if sample_obj['age'] else ""
                                    job_str = f"Job Title: {sample_obj['jobtitle']}" if sample_obj['jobtitle'] else ""
                                    prov_str = f"Province/State: {sample_obj['province']}" if sample_obj[
                                        'province'] else ""

                                    # Combine only non-empty pieces
                                    meta_parts = [x for x in [id_str, age_str, job_str, prov_str] if x]
                                    if meta_parts:
                                        italic_line = " | ".join(meta_parts)
                                        st.markdown(f"*{italic_line}*")

                                    # Show the actual text
                                    st.write(sample_obj["text"])

                                    # "Assign to group" selectbox
                                    if 'open_coding_groups' not in st.session_state:
                                        st.session_state.open_coding_groups = []
                                    current_assigned = st.session_state.open_coding_assignments.get(sample_obj["text"],
                                                                                                    "Unassigned")

                                    assigned_group = st.selectbox(
                                        "Assign this response to a group:",
                                        options=["Unassigned"] + [g["name"] for g in
                                                                  st.session_state.open_coding_groups],
                                        index=(["Unassigned"] + [g["name"] for g in
                                                                 st.session_state.open_coding_groups]).index(
                                            current_assigned)
                                        if current_assigned in (["Unassigned"] + [g["name"] for g in
                                                                                  st.session_state.open_coding_groups])
                                        else 0,
                                        key=f"sample_assign_{group_name}_{i}"
                                    )

                                    st.session_state.open_coding_assignments[sample_obj["text"]] = assigned_group

                        else:
                            st.warning(f"No valid responses available for group: {group_name}")
                    else:
                        st.warning(f"No responses found for {group_name}")