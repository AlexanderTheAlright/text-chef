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

# Set page config
st.set_page_config(
    page_title='Text Analysis Dashboard',
    page_icon='📊',
    layout="wide"
)

# Initialize session state
if 'custom_stopwords' not in st.session_state:
    # Try to load existing stopwords from CSV
    try:
        custom_stopwords_df = pd.read_csv('custom_stopwords.csv')
        st.session_state.custom_stopwords = set(custom_stopwords_df['word'].tolist())
    except FileNotFoundError:
        st.session_state.custom_stopwords = set(list(STOPWORDS))
if 'synonyms' not in st.session_state:
    st.session_state.synonyms = defaultdict(set)


@st.cache_data
def load_excel_file(file):
    """Load Excel file and return question mapping and all survey responses"""
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file)
        all_sheets = excel_file.sheet_names

        # First sheet is always question_mapping
        question_mapping = pd.read_excel(file, sheet_name='question_mapping')

        # All other sheets are surveys
        survey_sheets = [sheet for sheet in all_sheets if sheet != 'question_mapping']

        # Load all survey sheets
        responses_dict = {}
        for sheet in survey_sheets:
            responses_dict[sheet] = pd.read_excel(
                file,
                sheet_name=sheet,
                na_values=['', '#N/A', 'N/A', 'n/a', '<NA>', 'NULL', 'null', 'None', 'none'],
                keep_default_na=True
            )

        # Find all *_open variables in question mapping
        open_vars = question_mapping[
            question_mapping['variable'].str.endswith('_open', na=False)
        ]

        # Create variable display names with questions (showing full question directly)
        open_var_options = {
            row['variable']: f"{row['variable']} - {row['question']}"
            for _, row in open_vars.iterrows()
        }

        # Debug info
        with st.expander("Debug Information", expanded=False):
            st.write(f"Loaded {len(survey_sheets)} survey sheets: {survey_sheets}")
            st.write(f"Found {len(open_var_options)} *_open variables")
            for sheet, df in responses_dict.items():
                st.write(f"Sheet {sheet}: {len(df)} rows")

        return question_mapping, responses_dict, open_var_options
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, {}



def get_responses_for_variable(dfs_dict, var, group_by=None):
    """Get responses for a variable across all surveys, optionally grouped by another variable"""
    responses_by_survey = {}

    for survey_id, df in dfs_dict.items():
        # Find matching columns for this variable
        var_pattern = f"^{re.escape(var)}(?:\.1)?$"
        matching_cols = [col for col in df.columns if re.match(var_pattern, col, re.IGNORECASE)]

        if group_by and group_by in df.columns:
            # Group responses by the specified variable
            grouped_responses = defaultdict(list)

            for col in matching_cols:
                temp_df = df[[col, group_by]].copy()
                temp_df[col] = temp_df[col].fillna('').astype(str)
                temp_df[col] = temp_df[col].apply(lambda x: re.sub(r'<[^>]+>', '', x.strip()))

                for group_val, group_df in temp_df.groupby(group_by):
                    responses = [resp for resp in group_df[col].tolist() if resp.strip()]
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
                series = df[col].fillna('').astype(str)
                series = series.apply(lambda x: re.sub(r'<[^>]+>', '', x.strip()))

                na_values = {'', 'nan', 'n/a', 'na', '<na>', 'none', 'null', '#n/a', '0'}
                valid = series[~series.str.lower().isin(na_values)]
                responses.extend([resp for resp in valid.tolist() if resp.strip()])

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
    """Clean and process text"""
    if pd.isna(text) or not isinstance(text, (str, bytes)):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))

    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into words and filter out empty strings
    words = [word.strip() for word in text.split() if word.strip()]

    # Replace synonyms
    if synonyms:
        for main_word, syn_set in synonyms.items():
            words = [main_word if word in syn_set else word for word in words]

    # Remove stopwords (if provided)
    if stopwords:
        words = [word for word in words if word not in stopwords]

    return ' '.join(words)


def generate_wordcloud(texts, stopwords=None, synonyms=None, colormap='viridis', highlight_words=None, title=None):
    """Generate wordcloud with improved settings"""
    if not texts:
        return None

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
            stopwords=stopwords if stopwords else set(),
            collocations=False,
            min_word_length=2,
            prefer_horizontal=0.7,
            scale=2
        ).generate(text)
        return wc
    except Exception:
        return None


# Sidebar configuration
with st.sidebar:
    st.header("Text Processing Settings")

    # Manual stopwords management
    st.subheader("Stopwords Management")

    # Display current stopwords
    current_stopwords = "\n".join(sorted(st.session_state.custom_stopwords))
    new_stopwords = st.text_area(
        "Edit Stopwords (one per line)",
        value=current_stopwords,
        height=200,
        help="Edit stopwords directly. Each word should be on a new line."
    )

    if st.button("Update Stopwords"):
        new_stopwords_set = {word.strip().lower() for word in new_stopwords.split('\n') if word.strip()}
        st.session_state.custom_stopwords = new_stopwords_set
        # Save stopwords to CSV
        pd.DataFrame(list(new_stopwords_set), columns=['word']).to_csv('custom_stopwords.csv', index=False)
        st.success(f"Updated stopwords list ({len(new_stopwords_set)} words) and saved to file")

    # Existing stopwords CSV upload
    st.subheader("Import Additional Stopwords")
    stopwords_file = st.file_uploader("Upload Stopwords CSV", type=['csv'])
    if stopwords_file:
        new_stopwords, new_synonyms = load_stopwords_csv(stopwords_file)
        if st.button("Apply Stopwords"):
            st.session_state.custom_stopwords.update(new_stopwords)
            st.session_state.synonyms.update(new_synonyms)
            st.success(
                f"Added {len(new_stopwords)} stopwords and {sum(len(s) for s in new_synonyms.values())} synonyms")

    # Wordcloud settings
    st.subheader("Wordcloud Settings")
    colormap = st.selectbox(
        "Color scheme",
        ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    )

    highlight_words_input = st.text_area(
        "Highlight Words (one per line)",
        help="Enter words to highlight in red"
    )
    highlight_words = set(word.strip().lower() for word in highlight_words_input.split('\n') if word.strip())

# Main app
st.title('📊 Text Analysis Dashboard')
st.write('Analyze open-text responses across all surveys')

uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

if uploaded_file:
    question_mapping, responses_dict, open_var_options = load_excel_file(uploaded_file)

    if question_mapping is not None and responses_dict is not None and open_var_options:
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

        # Group by options are either the base variable or other open variables
        group_options = ['None', base_var] + other_open_vars

        # Group by option
        group_by = st.selectbox(
            "Group responses or compare with another variable",
            options=group_options,
            help="Select base variable to group by, or another open variable to compare"
        )

        group_by = None if group_by == 'None' else group_by

        if variable:
            # Get responses across all surveys
            responses_by_survey = get_responses_for_variable(responses_dict, variable, group_by)

            if not responses_by_survey:
                st.warning("No responses found for this variable in any survey")
                st.stop()

            # Show response counts by survey/group
            st.write("### Response Counts")
            for survey_id, responses in responses_by_survey.items():
                st.write(f"{survey_id}: {len(responses)} responses")

            # Combine all responses for analysis
            all_responses = []
            for responses in responses_by_survey.values():
                all_responses.extend(responses)

            # Analysis tabs
            tab1, tab2, tab3 = st.tabs(["Word Cloud", "Topic Model", "Word Frequency"])

            with tab1:
                # Display full question for the selected variable
                st.markdown(f"### Primary Question:\n{open_var_options[variable]}")

                # If comparing with another open variable, display its question too
                if group_by and group_by.endswith('_open'):
                    st.markdown(f"### Comparison Question:\n{open_var_options[group_by]}")

                wordcloud_title = st.text_input("Wordcloud Title (optional)", "")

                # Create a wordcloud for each group
                for survey_id, responses in responses_by_survey.items():
                    if responses:
                        st.subheader(f"Wordcloud for {survey_id}")

                        wc = generate_wordcloud(
                            responses,
                            st.session_state.custom_stopwords,
                            st.session_state.synonyms,
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

                            # Save the WordCloud image to a BytesIO buffer
                            buffer = BytesIO()
                            wc_image = wc.to_image()  # Convert WordCloud to a PIL Image
                            wc_image.save(buffer, format="PNG")  # Save the image to the buffer as PNG
                            buffer.seek(0)  # Move the pointer to the beginning of the buffer

                            # Provide the download option through the browser
                            st.download_button(
                                label="Download Wordcloud",
                                data=buffer,
                                file_name=f"{variable}_wordcloud_{survey_id}.png",
                                mime="image/png",
                                key=f"download_button_{survey_id}"  # Ensure unique key
                            )

            with tab2:
                # Topic modeling by survey
                for survey_id, responses in responses_by_survey.items():
                    st.write(f"### Topics for {survey_id}")

                    processed_texts = [
                        process_text(text, st.session_state.custom_stopwords, st.session_state.synonyms)
                        for text in responses
                    ]

                    if len(processed_texts) >= 5:  # Only do topic modeling if we have enough responses
                        try:
                            num_topics = min(10, len(processed_texts) // 5)  # Adjust topics based on data size
                            topic_model = BERTopic(n_gram_range=(1, 2), nr_topics=num_topics)
                            topics, _ = topic_model.fit_transform(processed_texts)
                            topic_info = topic_model.get_topic_info()

                            col1, col2 = st.columns([2, 1])

                            with col1:
                                fig = px.bar(
                                    topic_info,
                                    x='Topic',
                                    y='Count',
                                    title=f'Topic Distribution - {survey_id}'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.write("Top words per topic:")
                                for idx, row in topic_info.iterrows():
                                    if row['Topic'] != -1:
                                        st.write(f"Topic {row['Topic']}: {row['Name']}")
                        except Exception as e:
                            st.error(f"Error in topic modeling for {survey_id}: {e}")
                    else:
                        st.warning(f"Not enough responses in {survey_id} for topic modeling")

            with tab3:
                # Word frequency by survey
                for survey_id, responses in responses_by_survey.items():
                    st.write(f"### Word Frequencies for {survey_id}")

                    processed_texts = [
                        process_text(text, st.session_state.custom_stopwords, st.session_state.synonyms)
                        for text in responses
                        if text.strip()
                    ]

                    if processed_texts:
                        try:
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

                            st.plotly_chart(
                                px.bar(
                                    freq_df,
                                    x='word',
                                    y='frequency',
                                    title=f'Word Frequency Distribution - {survey_id}'
                                ),
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error in frequency analysis for {survey_id}: {e}")
                    else:
                        st.warning(f"No valid texts found for frequency analysis in {survey_id}")

            # Sample responses at the bottom
            with st.expander("Sample Responses by Survey", expanded=False):
                for survey_id, responses in responses_by_survey.items():
                    st.write(f"### {survey_id}")
                    for i, resp in enumerate(responses[:5]):
                        st.write(f"{i + 1}. {resp}")
    else:
        st.error("No open-ended variables found in the file")