import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import re
from io import BytesIO
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time
import re
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from itertools import combinations
import pandas as pd

# Set page config
st.set_page_config(
    page_title='Text Analysis Dashboard',
    page_icon='📊',
    layout="wide"
)

# Initialize session state
if 'custom_stopwords' not in st.session_state:
    try:
        # Attempt to load stopwords from CSV file
        custom_stopwords_df = pd.read_csv('custom_stopwords.csv')
        st.session_state.custom_stopwords = set(custom_stopwords_df['word'].tolist())
    except (FileNotFoundError, KeyError):
        # Fallback to default stopwords if file is not found or invalid
        st.session_state.custom_stopwords = set(list(STOPWORDS))

# Ensure preview stopwords are initialized as a copy of custom_stopwords
if 'preview_stopwords' not in st.session_state:
    st.session_state.preview_stopwords = st.session_state.custom_stopwords.copy()

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


@st.cache_data
def load_excel_file(file):
    """Load Excel file and return processed data structures"""
    try:
        excel_file = pd.ExcelFile(file)
        sheets = excel_file.sheet_names
        
        # Load question mapping
        if 'question_mapping' not in sheets:
            return None, None, None, None
            
        question_mapping = pd.read_excel(excel_file, 'question_mapping')
        if not all(col in question_mapping.columns for col in ['variable', 'question', 'surveyid']):
            return None, None, None, None

        # Load survey sheets
        responses_dict = {}
        available_open_vars = set()
        all_columns = set()
        
        for sheet in [s for s in sheets if s != 'question_mapping']:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            base_columns = {col.split('.')[0] for col in df.columns}
            all_columns.update(base_columns)
            sheet_open_vars = {col for col in base_columns if col.endswith('_open')}
            available_open_vars.update(sheet_open_vars)
            responses_dict[sheet] = df

        # Get grouping columns
        grouping_columns = sorted(col for col in all_columns 
                                if not col.endswith('_open') 
                                and not col.endswith('.1'))

        # Map open variables to questions
        open_var_options = {
            var: f"{var} - {question_mapping[question_mapping['variable'] == var].iloc[0]['question']}"
            if not question_mapping[question_mapping['variable'] == var].empty
            else var
            for var in sorted(available_open_vars)
        }

        return question_mapping, responses_dict, open_var_options, grouping_columns

    except Exception:
        return None, None, None, None

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


def display_standard_samples(texts_by_group, n_samples=5):
    """Display standard random samples for each group with improved error handling"""
    st.markdown("### Standard Sample Responses")
    
    # Debug print
    print(f"Number of groups: {len(texts_by_group)}")
    for group, texts in texts_by_group.items():
        print(f"Group {group}: {len(texts)} texts")
    
    if not texts_by_group:
        st.warning("No responses available to display.")
        return
        
    # Generate samples with error handling
    try:
        samples = get_standard_samples(texts_by_group, n_samples, st.session_state.get('sample_seed'))
        
        # Create tabs for each group
        if samples:
            group_tabs = st.tabs(list(samples.keys()))
            
            for tab, (group, group_samples) in zip(group_tabs, samples.items()):
                with tab:
                    if group_samples:
                        for i, sample in enumerate(group_samples, 1):
                            if isinstance(sample, str) and sample.strip():  # Validate sample
                                with st.expander(f"Response {i}", expanded=True):
                                    st.write(sample)
                            else:
                                st.warning(f"Invalid response found in group {group}")
                    else:
                        st.warning("No valid responses available for this group.")
        else:
            st.warning("No valid samples could be generated.")
    except Exception as e:
        st.error(f"Error displaying samples: {str(e)}")


# Core Synonym Management Functions
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

    # Save/Load functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("Save/Load Synonym Groups")

    if st.sidebar.button("Save Synonym Groups"):
        save_synonym_groups()

    synonym_file = st.sidebar.file_uploader("Load Synonym Groups CSV", type=['csv'])
    if synonym_file:
        load_synonym_groups(synonym_file)


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

# Add this function for synonym group visualization
def generate_synonym_group_wordclouds(texts, stopwords=None, synonym_groups=None, colormap='viridis'):
    """Generate separate wordclouds for each synonym group"""
    if not texts or not synonym_groups:
        return None

    # Create figure with subplots
    n_groups = len(synonym_groups)
    if n_groups == 0:
        return None

    rows = int(np.ceil(np.sqrt(n_groups)))
    cols = int(np.ceil(n_groups / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (group_name, synonyms) in enumerate(synonym_groups.items()):
        if idx >= len(axes):
            break

        # Process texts for this group
        processed_texts = []
        for text in texts:
            processed = process_text(text, stopwords, {group_name: synonyms})
            if processed:
                processed_texts.append(processed)

        if processed_texts:
            try:
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap=colormap,
                    stopwords=stopwords if stopwords else set(),
                    collocations=False,
                    min_word_length=2
                ).generate(' '.join(processed_texts))

                axes[idx].imshow(wc)
                axes[idx].axis('off')
                axes[idx].set_title(f"Group: {group_name}\nSynonyms: {', '.join(synonyms)}")
            except Exception:
                axes[idx].text(0.5, 0.5, 'Error generating wordcloud',
                               ha='center', va='center')
                axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, 'No valid text',
                           ha='center', va='center')
            axes[idx].axis('off')

    # Turn off any unused subplots
    for idx in range(len(synonym_groups), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig

# Modify the process_text
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

def load_stopwords_csv(file):
    """Load stopwords and synonyms from CSV file"""
    try:
        df = pd.read_csv(file)
        new_stopwords = set()
        new_synonyms = defaultdict(set)

        for _, row in df.iterrows():
            if pd.notna(row['grouping']) and row['grouping'].lower().strip() == 'stopword':
                word = str(row['word']).lower().strip()
                if word:
                    new_stopwords.add(word)
            elif pd.notna(row['grouping']) and row['grouping'].lower().strip() == 'synonym':
                if pd.notna(row['synonym_group']) and pd.notna(row['word']):
                    word = str(row['word']).lower().strip()
                    group = str(row['synonym_group']).lower().strip()
                    if word and group:
                        new_synonyms[group].add(word)

        return new_stopwords, new_synonyms
    except Exception as e:
        st.error(f"Error loading stopwords: {e}")
        return set(), defaultdict(set)


def get_text_columns(responses_df, question_mapping, survey_id):
    """Get all text-based columns that exist in the question mapping for the given survey"""
    # Get all variables for this survey from question mapping
    survey_vars = question_mapping[question_mapping['surveyid'] == survey_id]['variable'].tolist()

    # Filter for variables ending exactly with '*_open'
    open_vars = [var for var in survey_vars if str(var).endswith('*_open')]

    # Get base column names from responses
    response_cols = set()
    for col in responses_df.columns:
        base_col = col.split('.')[0]  # Remove .1 suffix if present
        response_cols.add(base_col)

    # Return only variables that exist in both mapping and responses
    valid_vars = [var for var in open_vars if var in response_cols]

    return sorted(valid_vars)


def process_text(text, stopwords=None, synonyms=None):
    """
    Clean and process text with improved stopword handling and synonym support
    """
    # Debug print to see what text is being received
    print(f"Input text: {text[:100]}...")  # First 100 chars

    if pd.isna(text) or not isinstance(text, (str, bytes)):
        print(f"Rejected - Invalid type or NaN: {type(text)}")
        return ""

    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Print the text after initial cleaning
    print(f"After initial cleaning: {text[:100]}...")
    
    # Check for invalid responses - print if found
    invalid_responses = {'dk', 'dk.', 'd/k', 'd.k.', 'dont know', "don't know", 
                        'na', 'n/a', 'n.a.', 'n/a.', 'not applicable',
                        'none', 'nil', 'no response', 'no answer', '.', '-', 'x'}
    
    if text in invalid_responses:
        print(f"Rejected - Invalid response: {text}")
        return ""

    # Remove HTML tags and clean text
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    # Print text after cleaning
    print(f"After cleaning: {text[:100]}...")

    # Split into words and apply stopwords/synonyms
    words = text.split()
    if stopwords:
        words = [word for word in words if word not in stopwords]
    
    if synonyms:
        words = [synonyms.get(word, word) for word in words]

    result = ' '.join(words)
    print(f"Final processed text: {result[:100]}...")
    
    return result

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


def generate_wordcloud(texts, stopwords=None, synonyms=None, colormap='viridis',
                       highlight_words=None):
    """Generate wordcloud with improved settings"""
    if not texts:
        return None

    # Get current stopwords from session state if none provided
    if stopwords is None:
        stopwords = st.session_state.get('preview_stopwords',
                                         st.session_state.get('custom_stopwords', set()))

    processed_texts = []
    for text in texts:
        processed = process_text(text, stopwords, synonyms)
        if processed:
            processed_texts.append(processed)

    if not processed_texts:
        return None

    text = ' '.join(processed_texts)

    def color_func(word, *args, **kwargs):
        if highlight_words and word.lower() in highlight_words:
            return "hsl(0, 100%, 50%)"
        return "hsl(0, 0%, 30%)"

    try:
        wc = WordCloud(
            width=2400,
            height=1200,
            background_color='white',
            colormap=colormap if not highlight_words else None,
            color_func=color_func if highlight_words else None,
            stopwords=stopwords,
            collocations=False,
            min_word_length=2,
            prefer_horizontal=0.7,
            scale=2
        ).generate(text)
        return wc
    except Exception:
        return None


def update_stopwords(new_stopwords, is_preview=False):
    """
    Update stopwords in session state and optionally save to CSV

    Parameters:
    - new_stopwords: set - New set of stopwords to use
    - is_preview: bool - Whether this is a preview update
    """
    if is_preview:
        st.session_state.preview_stopwords = new_stopwords
    else:
        # Update permanent stopwords
        st.session_state.custom_stopwords = new_stopwords
        # Save to CSV
        pd.DataFrame(list(new_stopwords), columns=['word']).to_csv(
            'custom_stopwords.csv',
            index=False
        )

def generate_multi_group_wordcloud(texts_by_group, stopwords=None, synonyms=None, colormap='viridis',
                                   highlight_words=None, main_title="", subtitle="", source_text=""):
    """
    Generate a 2x2 grid of wordclouds for different groups.

    Parameters:
    - texts_by_group: dict -> Dictionary with group names as keys and lists of texts as values
    - stopwords: set -> Set of stopwords to exclude
    - synonyms: dict -> Dictionary of synonym mappings
    - colormap: str -> Matplotlib colormap name
    - highlight_words: set -> Words to highlight in a different color
    - main_title: str -> Main title for the figure
    - subtitle: str -> Subtitle for the figure
    - source_text: str -> Source text to display at bottom
    """
    if len(texts_by_group) > 4:
        raise ValueError("Maximum 4 groups supported")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten()
    fig.patch.set_facecolor('white')

    # Title & Subtitle
    if main_title:
        fig.text(0.05, 0.96, main_title, fontsize=22, fontweight='bold', ha='left')
    if subtitle:
        fig.text(0.05, 0.94, subtitle, fontsize=18, ha='left', va='top', color='#404040')

    def color_func(word, *args, **kwargs):
        if highlight_words and word.lower() in highlight_words:
            return "hsl(0, 100%, 50%)"  # Red for highlighted words
        return None  # Use colormap for other words

    # Generate wordcloud for each group
    for i, (group_name, texts) in enumerate(texts_by_group.items()):
        ax = axes[i]

        if not texts:
            ax.axis("off")
            ax.set_title(f"{group_name}\n(No valid text)", fontsize=15, fontweight='bold')
            continue

        # Process texts
        processed_texts = []
        for text in texts:
            processed = process_text(text, stopwords, synonyms)
            if processed:
                processed_texts.append(processed)

        if not processed_texts:
            ax.axis("off")
            ax.set_title(f"{group_name}\n(No valid text)", fontsize=15, fontweight='bold')
            continue

        text = ' '.join(processed_texts)

        wc = WordCloud(
            width=1500,
            height=1000,
            background_color="white",
            collocations=False,
            stopwords=stopwords if stopwords else set(),
            colormap=colormap if not highlight_words else None,
            color_func=color_func if highlight_words else None,
            max_words=100,
            max_font_size=200,
            prefer_horizontal=1,
            scale=2
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"{group_name}\nTotal responses: {len(texts)}",
            fontsize=15,
            fontweight='bold'
        )

    # Turn off extra subplots
    for j in range(len(texts_by_group), 4):
        axes[j].axis("off")

    # Footnote
    if source_text:
        fig.text(
            0.05, 0.03,
            f"Source: {source_text}",
            fontsize=12,
            ha='left',
            va='bottom',
            color='#404040'
        )

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(
        top=0.88,
        bottom=0.10,
        left=0.07,
        right=0.95,
        wspace=0.1,
        hspace=0.1
    )

    return fig


def generate_comparison_wordcloud(texts1, texts2, stopwords=None, synonyms=None, colormap='viridis',
                                  highlight_words=None, main_title="", subtitle="", source_text="",
                                  label1="Group 1", label2="Group 2"):
    """
    Generate two wordclouds side by side for comparison.

    Parameters:
    - texts1, texts2: list -> Lists of texts for each group
    - stopwords: set -> Set of stopwords to exclude
    - synonyms: dict -> Dictionary of synonym mappings
    - colormap: str -> Matplotlib colormap name
    - highlight_words: set -> Words to highlight in a different color
    - main_title, subtitle, source_text: str -> Text for figure
    - label1, label2: str -> Labels for each group
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('white')

    # Title & Subtitle
    if main_title:
        fig.text(0.05, 0.96, main_title, fontsize=22, fontweight='bold', ha='left')
    if subtitle:
        fig.text(0.05, 0.94, subtitle, fontsize=18, ha='left', va='top', color='#404040')

    def color_func(word, *args, **kwargs):
        if highlight_words and word.lower() in highlight_words:
            return "hsl(0, 100%, 50%)"
        return None

    # Process and generate wordcloud for each group
    for ax, texts, label in [(ax1, texts1, label1), (ax2, texts2, label2)]:
        processed_texts = []
        for text in texts:
            processed = process_text(text, stopwords, synonyms)
            if processed:
                processed_texts.append(processed)

        if not processed_texts:
            ax.axis("off")
            ax.set_title(f"{label}\n(No valid text)", fontsize=15, fontweight='bold')
            continue

        text = ' '.join(processed_texts)

        wc = WordCloud(
            width=1500,
            height=1000,
            background_color="white",
            collocations=False,
            stopwords=stopwords if stopwords else set(),
            colormap=colormap if not highlight_words else None,
            color_func=color_func if highlight_words else None,
            max_words=150,
            max_font_size=200,
            prefer_horizontal=1,
            scale=2
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"{label}\nTotal responses: {len(texts)}",
            fontsize=15,
            fontweight='bold'
        )

    # Footnote
    if source_text:
        fig.text(
            0.05, 0.03,
            f"Source: {source_text}",
            fontsize=12,
            ha='left',
            va='bottom',
            color='#404040'
        )

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(
        top=0.88,
        bottom=0.10,
        left=0.07,
        right=0.95,
        wspace=0.1,
        hspace=0.1
    )

    return fig


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
            import logging
            logging.error(f"Error creating violin plot: {str(e)}")
            return None
            
    except Exception as e:
        import logging
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
                    import logging
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
        import logging
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return {}

def calculate_theme_evolution(texts_by_group, num_themes=5, min_freq=3):
    """Calculate the evolution of themes across groups."""
    from collections import Counter
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

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


def create_topic_distance_map(topic_model, processed_texts):
    """Create interactive topic distance visualization."""
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.manifold import TSNE

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
    import plotly.graph_objects as go

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
    import numpy as np

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
    import plotly.graph_objects as go
    import numpy as np

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

# Sidebar configuration
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

# Sidebar configuration
with st.sidebar:
    st.header("Analysis Settings")

    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

    if uploaded_file:
        question_mapping, responses_dict, open_var_options, grouping_columns = load_excel_file(uploaded_file)

        if question_mapping is not None and responses_dict is not None and open_var_options:
            # Add a separator for clarity
            st.markdown("---")
            st.subheader("📝 Variable Selection")

            # Select variable from all available *_open variables with questions
            variable = st.selectbox(
                "Select Open-ended Variable",
                options=list(open_var_options.keys()),
                format_func=lambda x: open_var_options[x],
                help="Variables ending with _open"
            )

            # Get base variable name (without _open)
            base_var = variable.replace('_open', '')

            # Get other open variables for comparison
            other_open_vars = [var for var in open_var_options.keys() if var != variable]

            # Group by options
            st.markdown("---")
            st.subheader("🔄 Grouping Options")
            group_by = st.selectbox(
                "Group responses by",
                options=['None'] + grouping_columns,
                help="Select any column to group by."
            )


            # Add a separator before the existing sidebar controls
            st.markdown("---")
            st.subheader("🎨 Wordcloud Settings")
            colormap = st.selectbox(
                "Color scheme",
                ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            )

            highlight_words_input = st.text_area(
                "Highlight Words (one per line)",
                help="Enter words to highlight in red"
            )
            highlight_words = set(word.strip().lower() for word in highlight_words_input.split('\n') if word.strip())

            # Word search functionality
            st.markdown("---")
            st.subheader("🔍 Search Responses")
            search_word = st.text_input(
                "Search for a specific word",
                help="Enter a word to find sample responses containing it"
            )

            # Add synonym management
            add_synonym_management_to_sidebar()

            with st.sidebar.expander("⚙️ Stopwords Management"):
                # Preview text area
                preview_text = st.text_area(
                    "Test stopwords (one per line)",
                    value="\n".join(sorted(st.session_state.preview_stopwords)),
                    key="preview_text",
                    height=100
                )

                # Update preview stopwords when preview button is clicked
                if st.button("Preview Changes"):
                    new_preview_stopwords = {
                        word.lower().strip()
                        for word in preview_text.split('\n')
                        if word.strip()
                    }
                    update_stopwords(new_preview_stopwords, is_preview=True)
                    st.success(f"Preview updated with {len(new_preview_stopwords)} words")

                st.markdown("---")

                # Permanent stopwords text area
                permanent_stopwords = st.text_area(
                    "Edit permanent stopwords",
                    value="\n".join(sorted(st.session_state.custom_stopwords)),
                    key="permanent_stopwords",
                    height=100
                )

                # Update permanent stopwords when update button is clicked
                if st.button("Update Dictionary"):
                    new_stopwords = {
                        word.lower().strip()
                        for word in permanent_stopwords.split('\n')
                        if word.strip()
                    }
                    update_stopwords(new_stopwords, is_preview=False)
                    st.success(f"Dictionary updated with {len(new_stopwords)} words")

if uploaded_file:
    if question_mapping is not None and responses_dict is not None and open_var_options and variable:
        # Show response counts by survey/group
        st.markdown("### Response Counts")
        responses_by_survey = get_responses_for_variable(responses_dict, variable, group_by)

        if not responses_by_survey:
            st.warning("No responses found for this variable in any survey")
            st.stop()

        for survey_id, responses in responses_by_survey.items():
            st.write(f"{survey_id}: {len(responses)} responses")

        # Analysis tabs
        tabs = st.tabs([
            "🎨 Word Cloud",
            "📊 Word Analysis",
            "🔍 Topic Discovery",
            "❤️ Sentiment Analysis",
            "🌊 Theme Evolution"
        ])

        # Word Cloud Tab
        with tabs[0]:
            with st.expander("🎨 About Word Cloud Visualization", expanded=False):
                st.markdown("""
                Word clouds provide an intuitive visual representation of your text data where:
                - **Larger words** appear more frequently in your responses
                - **Colors** help distinguish between different words
                - **Positioning** is optimized for visual appeal

                **Key Features:**
                - Automatically removes common stop words
                - Supports multiple visualization styles
                - Allows comparison between groups
                - Look up samples of responses with search responses tab on sidebar
                """)

            # Display full question for the selected variable
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
                    font-weight: 600;  /* Added this line to make text bold */
                }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="question-box">
                <div class="question-label">Primary Question</div>
                <div class="question-text">{open_var_options[variable]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Visualization type selector
            viz_type = st.radio(
                "Select visualization type",
                ["Single Wordcloud", "Multi-group Comparison (2x2)", "Side-by-side Comparison", "Synonym Groups"],
                help="Choose how to display your wordclouds"
            )

            # Common settings
            wordcloud_title = st.text_input("Wordcloud Title (optional)", "")
            wordcloud_subtitle = st.text_input("Subtitle (optional)", "")
            source_text = st.text_input("Source text (optional)", "")

            if viz_type == "Single Wordcloud":
                for survey_id, responses in responses_by_survey.items():
                    if responses:
                        st.subheader(f"Wordcloud for {survey_id}")

                        wc = generate_wordcloud(
                            responses,
                            st.session_state.custom_stopwords,
                            st.session_state.synonym_groups,
                            colormap=colormap,
                            highlight_words=highlight_words
                        )

                        if wc:
                            fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
                            ax.imshow(wc)
                            ax.axis('off')

                            if wordcloud_title:
                                plt.title(f"{wordcloud_title} - {survey_id}", fontsize=16, pad=20)

                            st.pyplot(fig)
                            plt.close()

                            # Save functionality
                            buffer = BytesIO()
                            wc.to_image().save(buffer, format="PNG")
                            buffer.seek(0)

                            st.download_button(
                                label="Download Wordcloud",
                                data=buffer,
                                file_name=f"{variable}_wordcloud_{survey_id}.png",
                                mime="image/png",
                                key=f"download_button_{survey_id}"
                            )
            elif viz_type == "Synonym Groups":
                if st.session_state.synonym_groups:
                    st.subheader("Wordclouds by Synonym Groups")

                    # Options for visualization
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

                        if separate_clouds:
                            # Generate separate wordclouds for each group
                            fig = generate_synonym_group_wordclouds(
                                responses,
                                st.session_state.custom_stopwords,
                                selected_synonym_groups,
                                colormap=colormap
                            )

                            if fig:
                                st.pyplot(fig)

                                # Save functionality
                                buffer = BytesIO()
                                fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                                buffer.seek(0)

                                st.download_button(
                                    label="Download Synonym Group Wordclouds",
                                    data=buffer,
                                    file_name=f"{variable}_synonym_wordclouds.png",
                                    mime="image/png"
                                )
                        else:
                            # Generate single wordcloud with all synonym replacements
                            wc = generate_wordcloud(
                                responses,
                                st.session_state.custom_stopwords,
                                selected_synonym_groups,
                                colormap=colormap,
                                highlight_words=highlight_words
                            )

                            if wc:
                                fig, ax = plt.subplots(figsize=(24, 12))
                                ax.imshow(wc)
                                ax.axis('off')
                                st.pyplot(fig)

                                # Save functionality
                                buffer = BytesIO()
                                wc.to_image().save(buffer, format="PNG")
                                buffer.seek(0)

                                st.download_button(
                                    label="Download Combined Wordcloud",
                                    data=buffer,
                                    file_name=f"{variable}_combined_wordcloud.png",
                                    mime="image/png"
                                )
                else:
                        st.warning(
                            "No synonym groups defined. Add synonym groups in the sidebar to use this visualization.")
            elif viz_type == "Multi-group Comparison (2x2)":
                if group_by:
                    all_groups = set()
                    for df in responses_dict.values():
                        if group_by in df.columns:
                            all_groups.update(df[group_by].dropna().unique())

                    selected_groups = st.multiselect(
                        "Select groups to compare (max 4)",
                        options=sorted(all_groups),
                        max_selections=4
                    )

                    if selected_groups:
                        texts_by_group = {}
                        for group in selected_groups:
                            group_responses = []
                            for df in responses_dict.values():
                                if group_by in df.columns and variable in df.columns:
                                    group_texts = df[df[group_by] == group][variable].dropna().tolist()
                                    group_responses.extend(group_texts)
                            texts_by_group[str(group)] = group_responses

                        fig = generate_multi_group_wordcloud(
                            texts_by_group,
                            stopwords=st.session_state.custom_stopwords,
                            synonyms=st.session_state.synonyms,
                            colormap=colormap,
                            highlight_words=highlight_words,
                            main_title=wordcloud_title,
                            subtitle=wordcloud_subtitle,
                            source_text=source_text
                        )

                        st.pyplot(fig)
                        plt.close()

                        buffer = BytesIO()
                        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                        buffer.seek(0)

                        st.download_button(
                            label="Download Multi-group Wordcloud",
                            data=buffer,
                            file_name=f"{variable}_multigroup_wordcloud.png",
                            mime="image/png",
                            key="download_multigroup"
                        )
                else:
                    st.warning("Please select a grouping variable to use multi-group comparison")

            else:  # Side-by-side Comparison
                if group_by:
                    all_groups = set()
                    for df in responses_dict.values():
                        if group_by in df.columns:
                            all_groups.update(df[group_by].dropna().unique())

                    col1, col2 = st.columns(2)
                    with col1:
                        group1 = st.selectbox("Select first group", options=sorted(all_groups), key="group1")
                    with col2:
                        group2 = st.selectbox("Select second group", options=sorted(all_groups), key="group2")

                    if group1 and group2:
                        texts1, texts2 = [], []
                        for df in responses_dict.values():
                            if group_by in df.columns and variable in df.columns:
                                texts1.extend(df[df[group_by] == group1][variable].dropna().tolist())
                                texts2.extend(df[df[group_by] == group2][variable].dropna().tolist())

                        fig = generate_comparison_wordcloud(
                            texts1, texts2,
                            stopwords=st.session_state.custom_stopwords,
                            synonyms=st.session_state.synonyms,
                            colormap=colormap,
                            highlight_words=highlight_words,
                            main_title=wordcloud_title,
                            subtitle=wordcloud_subtitle,
                            source_text=source_text,
                            label1=str(group1),
                            label2=str(group2)
                        )

                        st.pyplot(fig)
                        plt.close()

                        buffer = BytesIO()
                        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                        buffer.seek(0)

                        st.download_button(
                            label="Download Comparison Wordcloud",
                            data=buffer,
                            file_name=f"{variable}_comparison_wordcloud.png",
                            mime="image/png",
                            key="download_comparison"
                        )
                else:
                    st.warning("Please select a grouping variable to use side-by-side comparison")

        # Word Analysis Tab
        with tabs[1]:
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
                            min_edge_weight = st.slider(
                                "Minimum co-occurrence threshold",
                                min_value=1,
                                max_value=10,
                                value=2,
                                help="Minimum number of times words must appear together"
                            )

                        with col2:
                            max_words = st.slider(
                                "Maximum number of words",
                                min_value=10,
                                max_value=100,
                                value=30,
                                help="Maximum number of words to include in the network"
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
        with tabs[2]:
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
                        from sentence_transformers import SentenceTransformer
                        from sklearn.cluster import KMeans
                        from collections import defaultdict
                        import numpy as np


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
        with tabs[3]:
            with st.expander("❤️ About Sentiment Analysis", expanded=False):
                st.markdown("""
                Analyze the emotional tone and attitude in responses:

                **Key Metrics:**
                - 😊 Positive sentiment score
                - 😐 Neutral sentiment detection
                - 😔 Negative sentiment identification
                - 📊 Overall sentiment distribution

                **Visualizations:**
                1. Sentiment Distribution Sunburst
                2. Comparative Radar Chart
                3. Sentiment Flow Analysis
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
                viz_tabs = st.tabs(["🌟 Sunburst", "📡 Radar", "📊 Distribution"])

                with viz_tabs[0]:
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

        # Theme Evolution Tab
        with tabs[4]:
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

# Sample Responses Section
        st.markdown("---")
        st.markdown("## Response Examples")

        # Organize responses using the already loaded responses_by_survey
        if group_by:
            # With grouping
            texts_by_group = {}
            for survey_responses in responses_by_survey.values():
                # Since the responses are already filtered and processed in responses_by_survey,
                # we just need to split them by group
                group = survey_responses[0].split('_')[1] if '_' in survey_responses[0] else 'All'
                if group not in texts_by_group:
                    texts_by_group[group] = []
                texts_by_group[group].extend(survey_responses)
        else:
            # Without grouping, combine all responses under a single key
            texts_by_group = {'All Responses': []}
            for responses in responses_by_survey.values():
                texts_by_group['All Responses'].extend(responses)

        # If there's a search word, display matching responses
        if search_word:
            st.subheader(f"Responses containing '{search_word}'")
            matching_responses = find_word_in_responses(texts_by_group, search_word)

            if matching_responses:
                total_matches = sum(len(responses) for responses in matching_responses.values())
                st.metric("Total Matching Responses", total_matches)

                for group, responses in matching_responses.items():
                    if responses:
                        # Display up to 5 sample responses for each group
                        st.markdown(f"#### {group} ({len(responses)} matches)")
                        samples = responses[:5]
                        for i, response in enumerate(samples, 1):
                            with st.expander(f"Response {i}", expanded=True):
                                # Highlight the search word
                                pattern = re.compile(f"({re.escape(search_word)})", re.IGNORECASE)
                                highlighted_text = pattern.sub(r"**:red[\1]**", response)
                                st.markdown(highlighted_text)
            else:
                st.warning(f"No responses found containing '{search_word}'.")
        else:
            # Display random samples when no search word is entered
            if st.button("🔄 Generate New Random Samples"):
                st.session_state.sample_seed = int(time.time())
            
            display_standard_samples(texts_by_group, n_samples=5)


        # If there's a search word, display matching responses
        if search_word:
            st.subheader(f"Responses containing '{search_word}'")
            matching_responses = find_word_in_responses(texts_by_group, search_word)

            if matching_responses:
                total_matches = sum(len(responses) for responses in matching_responses.values())
                st.metric("Total Matching Responses", total_matches)

                for group, responses in matching_responses.items():
                    if responses:
                        # Display up to 5 sample responses for each group
                        st.markdown(f"#### {group} ({len(responses)} matches)")
                        samples = responses[:5]
                        for i, response in enumerate(samples, 1):
                            with st.expander(f"Response {i}", expanded=True):
                                # Highlight the search word
                                pattern = re.compile(f"({re.escape(search_word)})", re.IGNORECASE)
                                highlighted_text = pattern.sub(r"**:red[\1]**", response)
                                st.markdown(highlighted_text)
            else:
                st.warning(f"No responses found containing '{search_word}'.")
        else:
            # Display random samples when no search word is entered
            if st.button("🔄 Generate New Random Samples"):
                st.session_state.sample_seed = int(time.time())
        
            display_standard_samples(texts_by_group, n_samples=5)
    else:
        st.error("No open-ended variables found in the file")
