################################################################################
# STREAMLIT TEXT ANALYSIS APP
################################################################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import time
import re
import random
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from io import BytesIO
from datetime import datetime
import logging
import matplotlib
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from PIL import Image


matplotlib.use('Agg')  # For headless environments
px.defaults.color_continuous_scale = px.colors.sequential.Viridis # For default viridis
px.defaults.color_discrete_sequence = px.colors.sequential.Viridis

st.set_page_config(
    page_title='Text Analysis Dashboard',
    page_icon='📊',
    layout="wide",
    initial_sidebar_state="expanded"
)

###############################################################################
# 1) INITIALIZATION AND CACHING
###############################################################################

@st.cache_resource
def load_nltk_resources():
    """Ensure NLTK stopwords are present."""
    for lang in ['english', 'french']:
        try:
            nltk.data.find(f'corpora/stopwords/{lang}')
        except LookupError:
            nltk.download('stopwords')

load_nltk_resources()

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.file_processed = False
    st.session_state.selected_analysis = "Open Coding"  # default
    st.session_state.synonym_groups = defaultdict(set)

# Make sure synonyms in
if 'synonyms' not in st.session_state:
    st.session_state.synonyms = {}

# Make sure custom_stopwords is in session_state
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = set()

# We'll keep open coding groups/assignments in session_state too
if 'open_coding_groups' not in st.session_state:
    st.session_state.open_coding_groups = []
if 'open_coding_assignments' not in st.session_state:
    st.session_state.open_coding_assignments = {}
if 'last_save_time' not in st.session_state:
    st.session_state.last_save_time = time.time()

###############################################################################
# 2) STOPWORDS MANAGEMENT
###############################################################################

def load_custom_stopwords_file():
    """Load custom stopwords from CSV if exists."""
    try:
        if os.path.exists('custom_stopwords.csv'):
            df = pd.read_csv('custom_stopwords.csv')
            return set(str(x).lower().strip() for x in df['word'] if isinstance(x, str))
    except Exception as e:
        st.error(f"Error loading custom stopwords: {e}")
    return set()

def normalize_stopword_set(words_set):
    """Convert stopwords to lowercase, remove punctuation/spaces, etc."""
    normalized = set()
    for w in words_set:
        w = str(w).lower().strip()
        w = re.sub(r'<[^>]+>', '', w)
        w = re.sub(r'[^\w\s]', ' ', w)
        w = re.sub(r'\s+', ' ', w).strip()
        if w:
            for tok in w.split():
                normalized.add(tok)
    return normalized

def initialize_stopwords():
    """Load NLTK + custom CSV, unify them, store in session_state."""
    if not st.session_state.get('custom_stopwords'):
        eng = set(stopwords.words('english'))
        fr = set(stopwords.words('french'))
        custom = load_custom_stopwords_file()
        merged = eng.union(fr).union(custom)
        st.session_state.custom_stopwords = normalize_stopword_set(merged)

initialize_stopwords()

def update_stopwords_batch(new_words):
    """Add multiple new stopwords + save to CSV."""
    try:
        valid = {str(w).lower().strip() for w in new_words if w and isinstance(w, str)}
        if not valid:
            return False, "No valid stopwords provided."

        original_len = len(st.session_state.custom_stopwords)
        st.session_state.custom_stopwords.update(valid)
        st.session_state.custom_stopwords = normalize_stopword_set(st.session_state.custom_stopwords)

        # Merge with existing CSV
        existing = set()
        if os.path.exists('custom_stopwords.csv'):
            try:
                df = pd.read_csv('custom_stopwords.csv')
                for x in df['word']:
                    if isinstance(x, str):
                        existing.add(x.lower().strip())
            except:
                pass
        all_words = existing.union(st.session_state.custom_stopwords)
        newdf = pd.DataFrame(sorted(all_words), columns=['word'])
        newdf.to_csv('custom_stopwords.csv', index=False)

        added_count = len(st.session_state.custom_stopwords) - original_len
        return True, f"Added {added_count} new stopwords."
    except Exception as e:
        return False, f"Error: {e}"

def remove_stopword(word):
    """Remove a single stopword + update CSV."""
    try:
        if word in st.session_state.custom_stopwords:
            st.session_state.custom_stopwords.remove(word)
            # Save back
            df = pd.DataFrame(sorted(st.session_state.custom_stopwords), columns=['word'])
            df.to_csv('custom_stopwords.csv', index=False)
            return True, f"Removed {word}."
        else:
            return False, f"{word} not in list."
    except Exception as e:
        return False, f"Error removing stopword: {e}"

def reset_stopwords_to_nltk():
    """Reset to NLTK defaults + custom CSV merges."""
    try:
        eng = set(stopwords.words('english'))
        fr = set(stopwords.words('french'))
        custom = load_custom_stopwords_file()
        merged = eng.union(fr).union(custom)
        st.session_state.custom_stopwords = normalize_stopword_set(merged)
        # Save
        df = pd.DataFrame(sorted(st.session_state.custom_stopwords), columns=['word'])
        df.to_csv('custom_stopwords.csv', index=False)
        return True, "Reset stopwords to default + custom."
    except Exception as e:
        return False, f"Error: {e}"

def render_stopwords_management():
    """Render the stopwords manager in sidebar, including a new 'Preview' section."""
    # Keep a session-level set for preview stopwords (not saved to CSV)
    if 'temp_preview_stopwords' not in st.session_state:
        st.session_state.temp_preview_stopwords = set()

    # -- NEW UI for preview stopwords --
    st.markdown("#### Preview Stopwords")
    preview_stops = st.text_area("Add Preview Stopwords (one per line)", key="preview_stopwords_input")
    if st.button("Add Preview Stopwords"):
        if preview_stops.strip():
            words = [w.strip() for w in preview_stops.split('\n') if w.strip()]
            words = normalize_stopword_set(words)  # normalize them just like permanent stops
            st.session_state.temp_preview_stopwords = st.session_state.temp_preview_stopwords.union(words)
            st.success(
                f"Preview stopwords added (not saved permanently). "
                f"Current preview set size: {len(st.session_state.temp_preview_stopwords)}"
            )
        else:
            st.warning("No preview stopwords given.")

    if st.button("Clear Preview Stopwords"):
        st.session_state.temp_preview_stopwords.clear()
        st.success("Preview stopwords cleared.")

    st.markdown("#### Permanent Stopwords")

    # -- Existing UI for permanently adding stopwords --
    new_stops = st.text_area("Add Stopwords (one per line)", key="new_stopwords_input")
    if st.button("Add Stopwords"):
        if new_stops.strip():
            words = [w.strip() for w in new_stops.split('\n') if w.strip()]
            ok, msg = update_stopwords_batch(words)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.warning("No input given.")

    show_current = st.checkbox("View/Edit Current Stopwords", False)
    if show_current:
        stoplist = sorted(st.session_state.custom_stopwords)
        st.write(f"Total: {len(stoplist)} stopwords.")
        filter_st = st.text_input("Filter list", "")
        if filter_st.strip():
            stoplist = [s for s in stoplist if filter_st.lower() in s.lower()]

        cols = st.columns(3)
        for i, w in enumerate(stoplist):
            if cols[i % 3].button(f"❌ {w}", key=f"remove_stop_{w}"):
                ok, msg = remove_stopword(w)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    if st.button("Reset to defaults"):
        ok, msg = reset_stopwords_to_nltk()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")

def get_cmap_fixed(name):
    """
    A helper function that uses the new Matplotlib colormaps API
    if available, otherwise falls back to the older approach.
    """
    import matplotlib as mpl
    if hasattr(mpl, "colormaps"):  # Matplotlib 3.7+
        return mpl.colormaps.get_cmap(name)
    else:
        # For older Matplotlib versions
        return mpl.cm.get_cmap(name)

###############################################################################
# 3) SYNONYM GROUP MANAGEMENT
###############################################################################

def load_synonym_groups_from_csv(file='synonym_groups.csv'):
    """Load synonyms from CSV -> session_state."""
    from collections import defaultdict
    try:
        if os.path.exists(file):
            df = pd.read_csv(file)
            syns = defaultdict(set)
            for _, row in df.iterrows():
                gname = str(row['group_name']).strip().lower()
                syn = str(row['synonym']).strip().lower()
                if gname and syn:
                    syns[gname].add(syn)
            st.session_state.synonym_groups = syns
    except Exception as e:
        st.error(f"Error loading synonyms: {e}")

def save_synonym_groups_to_csv(file='synonym_groups.csv'):
    """Save synonyms from session_state -> CSV."""
    rows = []
    for group_name, synonyms in st.session_state.synonym_groups.items():
        for syn in synonyms:
            rows.append({'group_name': group_name, 'synonym': syn})
    df = pd.DataFrame(rows)
    df.to_csv(file, index=False)

if 'synonym_groups_loaded' not in st.session_state:
    load_synonym_groups_from_csv()
    st.session_state.synonym_groups_loaded = True

def render_synonym_groups_management():
    """Render synonyms manager in sidebar."""
    st.markdown("#### Synonym Groups")
    # SINGLE EXPANDER
    with st.expander("Manage Synonym Groups", expanded=False):
        new_grp = st.text_input("Group Name")
        new_syns = st.text_area("Synonyms (one per line)")
        if st.button("Add/Update Group"):
            if new_grp and new_syns:
                gname = new_grp.lower().strip()
                synonyms = {s.lower().strip() for s in new_syns.split('\n') if s.strip()}
                if gname in st.session_state.synonym_groups:
                    st.session_state.synonym_groups[gname].update(synonyms)
                    st.success(f"Updated group '{gname}' with {len(synonyms)} synonyms.")
                else:
                    st.session_state.synonym_groups[gname] = synonyms
                    st.success(f"Created group '{gname}' with {len(synonyms)} synonyms.")
                save_synonym_groups_to_csv()
            else:
                st.warning("Please enter a group name + synonyms.")

        st.markdown("---")
        st.write("#### Existing Groups")
        # NO SUB-EXPANDERS NOW
        for g, syns in dict(st.session_state.synonym_groups).items():
            st.write(f"**Group:** {g}")
            st.write(", ".join(sorted(list(syns))))
            if st.button(f"Delete group '{g}'", key=f"del_{g}"):
                del st.session_state.synonym_groups[g]
                save_synonym_groups_to_csv()
                st.warning(f"Deleted group {g}.")



###############################################################################
# 4) DATA LOADING
###############################################################################

@st.cache_data
def load_excel_file(excel, chosen_survey="All"):
    """
    Load the question_mapping sheet + either chosen_survey or all sheets,
    then add 'primary_code'/'secondary_code' as valid grouping columns.
    (No other functionality changed.)
    """
    import pandas as pd

    xls = pd.ExcelFile(excel)
    sheets = xls.sheet_names
    if 'question_mapping' not in sheets:
        return None, None, None, None

    qmap = pd.read_excel(xls, 'question_mapping')
    if not all(c in qmap.columns for c in ['variable', 'question', 'surveyid']):
        return None, None, None, None

    responses_dict = {}
    all_cols = set()
    open_vars_set = set()

    valid_sheets = [s for s in sheets if s != 'question_mapping']
    if chosen_survey == "All":
        sheets_to_load = valid_sheets
    else:
        sheets_to_load = [s for s in valid_sheets if s == chosen_survey]

    invalid_resps = {
        'dk', 'dk.', 'd/k', 'd.k.', 'dont know', "don't know", "na", "n/a", "n.a.", "n/a.",
        'not applicable', 'none', 'nil', 'no response', 'no answer', '.', '-', 'x', 'refused', 'ref',
        'dk/ref', 'nan', 'NaN', 'NAN', '_dk_', '_na_', '___dk___', '___na___', '__dk__', '__na__',
        '_____dk_____', '_____na_____',''
    }

    for sheet in sheets_to_load:
        df = pd.read_excel(xls, sheet_name=sheet, na_values=['', 'NA', 'nan', 'NaN', 'null', 'none', '#N/A', 'N/A'])
        base_cols = {c.split('.')[0] for c in df.columns}
        all_cols.update(base_cols)

        open_candidates = {c for c in base_cols if c.endswith('_open')}
        open_vars_set.update(open_candidates)

        # Clean any _open columns
        for col in df.columns:
            if col.split('.')[0] in open_candidates:
                def clean_val(x):
                    if pd.isna(x):
                        return pd.NA
                    x = str(x).lower().strip()
                    return pd.NA if x in invalid_resps else x

                df[col] = df[col].apply(clean_val)

        responses_dict[sheet] = df

    # Instead of excluding _open columns from the grouping options,
    # we keep them as well. Remove only the .1 duplicates if you want.
    grouping_cols = sorted(c for c in all_cols if not c.endswith('.1'))

    # Build "open_var_opts" from question_mapping
    open_var_opts = {}
    for v in sorted(open_vars_set):
        row_for_v = qmap[qmap['variable'] == v]
        if not row_for_v.empty:
            open_var_opts[v] = f"{v} - {row_for_v.iloc[0]['question']}"
        else:
            open_var_opts[v] = v

    # -----------------------------------------------------------------
    # NEW: Also allow 'primary_code' and 'secondary_code' as group-by
    # -----------------------------------------------------------------
    # So the user can group by the open-coding assignments from assignments.csv
    if "primary_code" not in grouping_cols:
        grouping_cols.append("primary_code")
    if "secondary_code" not in grouping_cols:
        grouping_cols.append("secondary_code")

    return qmap, responses_dict, open_var_opts, grouping_cols

# -----------------------------------------------------------
# In your sidebar or wherever you choose "Group By":
# Now use `st.multiselect` (instead of `st.selectbox`)
# so the user can pick multiple columns.
# -----------------------------------------------------------

# Example sidebar usage:
def sidebar_controls(open_var_list, group_col_list):
    """
    :param open_var_list: A list of all available _open variables (e.g. ['job_open','feedback_open',...])
    :param group_col_list: A list of all potential grouping columns (including _open if desired)

    Returns:
      chosen_opens -> list of open vars user selected
      chosen_groups -> list of columns to group by
    """
    st.markdown("### Select Open Variables")
    chosen_opens = st.multiselect(
        "Pick one or more _open variables to analyze:",
        options=open_var_list,
        default=[]
    )

    st.markdown("### Group By Columns")
    chosen_groups = st.multiselect(
        "Pick one or more columns to group by:",
        options=group_col_list,
        default=[]
    )

    return chosen_opens, chosen_groups


def get_responses_for_variable(dfs_dict, var, group_by=None):
    """
    Retrieve text responses for a single open variable, optionally grouped.
    If 'primary_code' or 'secondary_code' is chosen as a grouping column,
    we'll look up the assignment in st.session_state.open_coding_assignments.

    Returns a dict {group_key -> list_of_texts}.
    """
    import re
    from collections import defaultdict

    out = defaultdict(list)
    pattern = f"^{re.escape(var)}(?:\\.1)?$"

    # Normalize group_by to a list if it's a single string
    if group_by and isinstance(group_by, str):
        group_by = [group_by]

    # Check if we might need to look up codes
    needs_primary = False
    needs_secondary = False
    if group_by:
        needs_primary = ('primary_code' in group_by)
        needs_secondary = ('secondary_code' in group_by)

    # Ensure open_coding_assignments is loaded
    # The code that sets up st.session_state.open_coding_assignments
    # typically runs in open coding or initialization. Make sure it's loaded:
    if 'open_coding_assignments' not in st.session_state:
        st.session_state.open_coding_assignments = {}

    for sid, df in dfs_dict.items():
        # Find columns that match var or var.1
        matching_cols = [c for c in df.columns if re.match(pattern, c, re.IGNORECASE)]
        if not matching_cols:
            continue

        # We must ensure each row has an 'id' if we plan to do lookups
        # If 'id' doesn't exist, we can't do assignment lookups
        has_id_col = ('id' in df.columns)

        if group_by and all(
                (gb in df.columns) or (gb in ['primary_code', 'secondary_code'])
                for gb in group_by
        ):
            # Group by multiple columns (including possible 'primary_code','secondary_code')
            for col in matching_cols:
                sub_df = df[[col] + (group_by if not has_id_col else group_by + ['id'])].copy()
                sub_df.dropna(subset=[col], inplace=True)

                for _, row in sub_df.iterrows():
                    text_val = str(row[col]).strip()
                    if text_val.lower() == 'nan' or text_val == '':
                        continue

                    row_id = str(row['id']) if (has_id_col and pd.notna(row['id'])) else ""

                    # Retrieve assignment codes if needed
                    p_code = "Unassigned"
                    s_code = "Unassigned"
                    if (needs_primary or needs_secondary) and row_id:
                        assigned = st.session_state.open_coding_assignments.get((row_id, var), None)
                        if assigned:
                            p_code = assigned.get("primary_code", "Unassigned")
                            s_code = assigned.get("secondary_code", "Unassigned")

                    # Build a composite key from all group_by columns
                    group_values = []
                    for gb in group_by:
                        if gb == 'primary_code':
                            group_values.append(p_code)
                        elif gb == 'secondary_code':
                            group_values.append(s_code)
                        else:
                            val_ = row[gb]
                            group_values.append(str(val_) if pd.notna(val_) else "Missing")

                    group_key = "_".join(group_values)
                    out[group_key].append(text_val)

        else:
            # If no valid group_by or invalid group_by, unify everything into a single 'All' group
            all_texts = []
            for col in matching_cols:
                colvals = [
                    str(r).strip() for r in df[col].dropna()
                    if str(r).strip().lower() != 'nan'
                ]
                all_texts.extend(colvals)
            if all_texts:
                out["All"].extend(all_texts)

    # Sort by descending number of responses
    out = dict(sorted(out.items(), key=lambda x: len(x[1]), reverse=True))
    return out

def build_var_resps_for_multiselect(
        dfs_dict,
        open_vars,
        group_by_cols=None
):
    """
    :param dfs_dict: dict of DataFrame by sheet (e.g., from load_excel_file)
    :param open_vars: list of open variables the user wants to analyze
    :param group_by_cols: list of columns to group by (can be 0, 1, or many)

    Returns a dict { "label_for_graph": [list of text], ... }
    where label_for_graph is e.g. "job_open|Female|Ontario" if multiple group-bys.
    """
    if group_by_cols is None:
        group_by_cols = []

    out = defaultdict(list)

    # Build a regex pattern for each open var to match var or var.1 columns
    for open_var in open_vars:
        pattern = f"^{re.escape(open_var)}(?:\\.1)?$"

        # Loop each sheet
        for sid, df in dfs_dict.items():
            # Find columns that match the open var
            matching_cols = [c for c in df.columns if re.match(pattern, c, re.IGNORECASE)]
            if not matching_cols:
                continue

            # If no group_by_cols, everything lumps into a single label
            if not group_by_cols:
                combined_label = open_var  # or f"{open_var} (All)"
                texts = []
                for col in matching_cols:
                    colvals = df[col].dropna().astype(str).str.strip()
                    colvals = colvals[colvals.str.lower() != 'nan']
                    texts.extend(colvals.tolist())
                if texts:
                    out[combined_label].extend(texts)

            else:
                # We do grouping across all chosen columns
                used_cols = matching_cols + group_by_cols
                sub_df = df[used_cols].dropna(subset=matching_cols)

                # For each row, build a label from group_by_cols
                for _, row in sub_df.iterrows():
                    # pick the open column that is not empty
                    # (in practice, there might be multiple matching_cols, e.g. var and var.1)
                    found_text = None
                    for mc in matching_cols:
                        val_ = str(row[mc]).strip()
                        if val_.lower() != 'nan' and val_ != '':
                            found_text = val_
                            break
                    if not found_text:
                        continue

                    # Build the group part
                    group_parts = []
                    for gcol in group_by_cols:
                        gval = str(row[gcol]).strip() if pd.notna(row[gcol]) else "Missing"
                        group_parts.append(gval)

                    # Combine open var name + group values => final label
                    # E.g. "job_open|Female|Ontario"
                    combined_label = "|".join([open_var] + group_parts)
                    out[combined_label].append(found_text)

###############################################################################
# 5) TEXT PROCESSING
###############################################################################

def process_text(text, stopwords=None, synonym_groups=None):
    """Clean text, remove stopwords, and apply synonyms.
       If any word in a synonym group is in stopwords, remove that entire group."""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Basic cleaning
    txt = str(text).lower().strip()
    txt = re.sub(r'<[^>]+>', '', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()

    words = txt.split()

    # 1) Combine permanent + preview stopwords
    effective_stopwords = set(stopwords) if stopwords else set()
    if 'temp_preview_stopwords' in st.session_state and st.session_state.temp_preview_stopwords:
        effective_stopwords |= st.session_state.temp_preview_stopwords

    # 2) Identify entire synonym groups to remove if any synonym is in stopwords
    synonyms_to_remove = set()
    if synonym_groups:
        for gname, synset in synonym_groups.items():
            # If this group's synonyms intersect with stopwords => remove them all
            if synset.intersection(effective_stopwords):
                synonyms_to_remove |= synset  # add all synonyms in that group

    # 3) Filter out words that are in synonyms_to_remove OR in effective_stopwords
    new_words = []
    for w in words:
        # If w is in the "to_remove" set or in the final stopwords, skip it
        if w in synonyms_to_remove or w in effective_stopwords:
            continue

        # Otherwise, see if it belongs to a synonym group
        replaced = False
        if synonym_groups:
            for gname, synset in synonym_groups.items():
                if w in synset:
                    # Replace w with group name
                    new_words.append(gname)
                    replaced = True
                    break

        # If no synonym found, keep original word
        if not replaced:
            new_words.append(w)

    return ' '.join(new_words)


###############################################################################
# OPEN CODING
###############################################################################

def auto_save_check():
    """
    Automatically save coding state every 5 minutes if changes have been made
    and not saved since then.
    """
    if "last_save_time" not in st.session_state:
        st.session_state.last_save_time = time.time()
        return

    # If 5 minutes (300 seconds) have passed since last_save_time, do an auto-save
    if (time.time() - st.session_state.last_save_time) > 300:
        if save_coding_state():
            st.sidebar.info("Auto-saved coding changes.")
        else:
            st.sidebar.warning("Auto-save attempted but failed.")

def save_coding_state():
    """
    Saves the group definitions and the coded assignments to disk,
    inside a 'cache/' folder (without creating extra backup files).
    """
    try:
        os.makedirs('cache', exist_ok=True)
        # 1) Save group definitions
        if st.session_state.open_coding_groups:
            save_open_coding_groups('cache/groups.csv')

        # 2) Save open-coding assignments
        if st.session_state.open_coding_assignments:
            save_open_coding_assignments('cache/assignments.csv')

        st.session_state.last_save_time = time.time()
        return True

    except Exception as e:
        st.error(f"Error saving coding state: {e}")
        return False

def initialize_coding_state():
    """
    Ensure open coding data is loaded exactly once
    and we have placeholders in st.session_state for data.
    """
    if 'coding_initialized' not in st.session_state:
        st.session_state.open_coding_groups = load_open_coding_groups('cache/groups.csv')
        st.session_state.open_coding_assignments = load_open_coding_assignments('cache/assignments.csv')
        st.session_state.coding_initialized = True

    if 'open_coding_table_data' not in st.session_state:
        st.session_state.open_coding_table_data = {}  # dict {var -> final_df}

def load_open_coding_groups(file='cache/groups.csv'):
    """
    Load group definitions from CSV -> st.session_state.open_coding_groups.
    Each row expected: [name, desc].
    """
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            return df.to_dict('records')
        except Exception as e:
            print(f"Could not load groups file: {e}")
            return []
    return []

def save_open_coding_groups(file='cache/groups.csv'):
    """
    Save the current open_coding_groups to CSV.
    Each row: name, desc
    """
    df_g = pd.DataFrame(st.session_state.open_coding_groups)
    if not df_g.empty:
        df_g.to_csv(file, index=False)


def load_open_coding_assignments(file='cache/assignments.csv'):
    assignment_dict = {}
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)

            # Expect these columns, add them if they're missing
            expected_cols = ["id", "surveyid", "variable", "primary_code", "secondary_code"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None

            # Fill missing codes with 'Unassigned'
            df["primary_code"] = df["primary_code"].fillna("Unassigned").astype(str)
            df["secondary_code"] = df["secondary_code"].fillna("Unassigned").astype(str)

            for _, row in df.iterrows():
                key = (str(row["id"]), str(row["surveyid"]), str(row["variable"]))
                assignment_dict[key] = {
                    "primary_code": row["primary_code"],
                    "secondary_code": row["secondary_code"]
                }
        except Exception as e:
            print(f"Could not load assignments file: {e}")
    return assignment_dict

def save_open_coding_assignments(file='cache/assignments.csv'):
    rows = []
    for (row_id, row_surveyid, row_var), codes in st.session_state.open_coding_assignments.items():
        rows.append({
            "id": row_id,
            "surveyid": row_surveyid,         # <--- newly added
            "variable": row_var,
            "primary_code": codes.get("primary_code", "Unassigned"),
            "secondary_code": codes.get("secondary_code", "Unassigned"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(file, index=False)

def update_coded_assignments(variable, final_df, df_updated=None):
    """
    1) Merge user edits from df_updated into st.session_state.open_coding_assignments.
    2) Keep final_df in the same row order/sorting that df_updated shows.
    3) Re-populate 'primary_desc' and 'secondary_desc' from the dictionary.
    4) Save everything to disk.
    """
    if df_updated is not None and not df_updated.empty:
        # STEP 1: Store user changes in assignment dictionary
        for i, row in df_updated.iterrows():
            row_id = str(row.get("id", ""))
            survid = str(row.get("surveyid", ""))
            dict_key = (row_id, survid, variable)

            p_val = row.get("primary_code", "Unassigned") or "Unassigned"
            s_val = row.get("secondary_code", "Unassigned") or "Unassigned"

            # Remove if both codes = "Unassigned"
            if p_val == "Unassigned" and s_val == "Unassigned":
                if dict_key in st.session_state.open_coding_assignments:
                    del st.session_state.open_coding_assignments[dict_key]
            else:
                st.session_state.open_coding_assignments[dict_key] = {
                    "primary_code": p_val,
                    "secondary_code": s_val
                }

        # STEP 2: Overwrite final_df's code columns with df_updated's codes (preserving the user order)
        #         This ensures final_df stays in the same row order as df_updated
        if "primary_code" in df_updated.columns and "primary_code" in final_df.columns:
            final_df["primary_code"] = df_updated["primary_code"]
        if "secondary_code" in df_updated.columns and "secondary_code" in final_df.columns:
            final_df["secondary_code"] = df_updated["secondary_code"]

        # STEP 3: Re-generate primary_desc, secondary_desc using the assignment dictionary
        for i, row in final_df.iterrows():
            row_id_ = str(row.get("id", ""))
            survid_ = str(row.get("surveyid", ""))
            dict_key = (row_id_, survid_, variable)

            assigned_obj = st.session_state.open_coding_assignments.get(
                dict_key, {"primary_code": "Unassigned", "secondary_code": "Unassigned"}
            )

            # If you have these columns, fill them in from your group definitions
            if "primary_desc" in final_df.columns:
                if assigned_obj["primary_code"] != "Unassigned":
                    grp = next(
                        (g for g in st.session_state.open_coding_groups
                         if g["name"] == assigned_obj["primary_code"]),
                        None
                    )
                    final_df.at[i, "primary_desc"] = grp["desc"] if grp else ""
                else:
                    final_df.at[i, "primary_desc"] = ""

            if "secondary_desc" in final_df.columns:
                if assigned_obj["secondary_code"] != "Unassigned":
                    grp2 = next(
                        (g for g in st.session_state.open_coding_groups
                         if g["name"] == assigned_obj["secondary_code"]),
                        None
                    )
                    final_df.at[i, "secondary_desc"] = grp2["desc"] if grp2 else ""
                else:
                    final_df.at[i, "secondary_desc"] = ""

    # STEP 4: Save the updated assignments (and any other needed files)
    try:
        return save_coding_state()
    except Exception as e:
        st.error(f"Error saving coding: {e}")
        return False

def render_open_coding_interface(variable, responses_dict, open_var_options, grouping_columns):
    """
    Main function to display and manage the open-coding interface.
      * A single 'cache/groups.csv' for code groups (name/desc)
      * A single 'cache/assignments.csv' for coded assignments
         (id, variable, primary_code, secondary_code)
      * Everything is easily pushed to Git by any user (no backups needed).
    """

    # 0) Initialization & autosave
    initialize_coding_state()
    auto_save_check()

    # 1) Build or retrieve the main DataFrame for 'variable'
    if variable not in st.session_state.open_coding_table_data:
        all_dfs = []
        for sid, df in responses_dict.items():
            if variable in df.columns:
                cpy = df.copy()
                cpy["surveyid"] = sid
                all_dfs.append(cpy)

        if not all_dfs:
            st.warning(f"No data found for variable '{variable}'.")
            return

        cdf = pd.concat(all_dfs, ignore_index=True)
        cdf = cdf.dropna(subset=[variable])
        cdf = cdf[cdf[variable].astype(str).str.strip() != ""]

        for needed in ["primary_code", "primary_desc", "secondary_code", "secondary_desc"]:
            if needed not in cdf.columns:
                if "desc" in needed:
                    cdf[needed] = ""
                else:
                    cdf[needed] = "Unassigned"

        # Merge existing assignments from session_state
        for idx, row in cdf.iterrows():
            row_id = str(row.get("id", "")) or ""
            survid_ = str(row.get("surveyid", "")) or ""
            key_ = (row_id, survid_, variable)

            assigned_obj = st.session_state.open_coding_assignments.get(
                key_, {"primary_code": "Unassigned", "secondary_code": "Unassigned"}
            )
            cdf.at[idx, "primary_code"] = assigned_obj.get("primary_code", "Unassigned")
            cdf.at[idx, "secondary_code"] = assigned_obj.get("secondary_code", "Unassigned")

            if cdf.at[idx, "primary_code"] != "Unassigned":
                gobj = next(
                    (g for g in st.session_state.open_coding_groups
                     if g["name"] == cdf.at[idx, "primary_code"]),
                    None
                )
                cdf.at[idx, "primary_desc"] = gobj["desc"] if gobj else ""

            if cdf.at[idx, "secondary_code"] != "Unassigned":
                gobj2 = next(
                    (g for g in st.session_state.open_coding_groups
                     if g["name"] == cdf.at[idx, "secondary_code"]),
                    None
                )
                cdf.at[idx, "secondary_desc"] = gobj2["desc"] if gobj2 else ""

        st.session_state.open_coding_table_data[variable] = cdf

    final_df = st.session_state.open_coding_table_data[variable]
    if final_df.empty:
        st.warning("No valid rows for this variable.")
        return

    # 2) Pie Charts of Primary & Secondary Codes
    c1, c2 = st.columns(2)
    group_data = pd.DataFrame(st.session_state.open_coding_groups)
    if group_data.empty:
        st.info("No coding groups defined yet, so pie charts will be minimal.")
    else:
        with c1:
            if "primary_code" in final_df.columns:
                pc_counts = final_df["primary_code"].value_counts(dropna=False).reset_index()
                pc_counts.columns = ["Group", "Count"]
                merged_pc = pd.merge(
                    pc_counts, group_data, left_on="Group", right_on="name", how="left"
                ).rename(columns={"desc": "Description"})
                merged_pc["Description"] = merged_pc["Description"].fillna("No Description")

                if merged_pc["Count"].sum() > 0:
                    fig_p = px.pie(
                        merged_pc,
                        values="Count",
                        names="Group",
                        hover_data=["Description"],
                        title="Primary Code"
                    )
                    fig_p.update_layout(width=400, height=400, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_p, use_container_width=False)
                else:
                    st.info("No primary codes assigned yet.")

        with c2:
            if "secondary_code" in final_df.columns:
                sc_counts = final_df["secondary_code"].value_counts(dropna=False).reset_index()
                sc_counts.columns = ["Group", "Count"]
                merged_sc = pd.merge(
                    sc_counts, group_data, left_on="Group", right_on="name", how="left"
                ).rename(columns={"desc": "Description"})
                merged_sc["Description"] = merged_sc["Description"].fillna("No Description")

                if merged_sc["Count"].sum() > 0:
                    fig_s = px.pie(
                        merged_sc,
                        values="Count",
                        names="Group",
                        hover_data=["Description"],
                        title="Secondary Code"
                    )
                    fig_s.update_layout(width=400, height=400, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_s, use_container_width=False)
                else:
                    st.info("No secondary codes assigned yet.")

    # 3) Table Editor with optional filtering
    st.markdown("---")
    st.subheader("Edit Coding Assignments")

    base_var = variable.replace("_open", "")
    desired_cols = [
        "id", "surveyid", "age", "gender", "region", "jobtitle",
        base_var,
        variable,
        "primary_code", "primary_desc",
        "secondary_code", "secondary_desc"
    ]
    existing_cols = [c for c in desired_cols if c in final_df.columns]
    df_for_editor = final_df[existing_cols].copy()

    # --- Global search & column filters ---
    global_search = st.text_input("Global Search / Filter:", "", key="oc_global_search")
    show_col_filters = st.checkbox("Show column-based filters", value=False, key="oc_show_col_filters")
    col_filters = {}

    if show_col_filters:
        st.caption("Case-insensitive substring match per column.")
        ncols_per_line = 3
        filter_cols = st.columns(ncols_per_line)
        for i, col_name in enumerate(existing_cols):
            with filter_cols[i % ncols_per_line]:
                col_filters[col_name] = st.text_input(f"Filter: {col_name}", "")

    if global_search.strip():
        mask_global = df_for_editor.apply(
            lambda row: row.astype(str).str.contains(global_search, case=False, na=False).any(),
            axis=1
        )
        df_for_editor = df_for_editor[mask_global]

    if show_col_filters:
        for col_name, val in col_filters.items():
            if val.strip():
                df_for_editor = df_for_editor[
                    df_for_editor[col_name].astype(str).str.contains(val, case=False, na=False)
                ]

    # --- Configure columns in the data_editor ---
    group_names = ["Unassigned"] + [g["name"] for g in st.session_state.open_coding_groups]
    col_config = {}
    for col_ in df_for_editor.columns:
        if col_ == "primary_code":
            col_config[col_] = st.column_config.SelectboxColumn(
                label="primary_code",
                options=group_names
            )
        elif col_ == "secondary_code":
            col_config[col_] = st.column_config.SelectboxColumn(
                label="secondary_code",
                options=group_names
            )
        elif col_ in ["primary_desc", "secondary_desc"]:
            col_config[col_] = st.column_config.TextColumn(
                label=col_,
                disabled=True
            )
        else:
            col_config[col_] = st.column_config.TextColumn(
                label=col_,
                disabled=True
            )

    df_updated = st.data_editor(
        df_for_editor,
        column_config=col_config,
        hide_index=True,
        use_container_width=True
    )

    # --- Button to save coding changes ---
    if st.button("Save Coding Changes"):
        ok = update_coded_assignments(variable, final_df, df_updated=df_updated)
        if ok:
            st.success("Coding saved successfully.")
        else:
            st.error("Error saving coding.")

    # Random Samples for convenience
    st.markdown("---")
    st.markdown("### 🎲 Random Samples for Review")

    def build_samples_dict():
        out = {"All": []}
        for sid, df in responses_dict.items():
            if variable in df.columns:
                for _, row in df.iterrows():
                    val = row[variable]
                    if pd.notna(val) and str(val).strip():
                        out["All"].append({
                            "id": str(row.get("id", "")) or "",
                            "surveyid": sid,
                            "text": str(val).strip(),
                            "age": row.get("age"),
                            "gender": row.get("gender"),
                            "jobtitle": row.get("jobtitle"),
                            "region": row.get("region")
                        })
        return out

    samples_dict = build_samples_dict()
    # Deduplicate by text
    for cat_k, items in samples_dict.items():
        seen_txt = set()
        newlist = []
        for obj in items:
            if obj["text"] not in seen_txt:
                seen_txt.add(obj["text"])
                newlist.append(obj)
        samples_dict[cat_k] = newlist

    num_samp = st.slider("Number of samples per group", 1, 20, 5)
    if "sample_seed" not in st.session_state:
        st.session_state["sample_seed"] = 1234
    random.seed(st.session_state["sample_seed"])

    for cat, arr in samples_dict.items():
        st.markdown(f"#### {cat} (Total: {len(arr)})")
        if not arr:
            st.write("🚫 No data in this category.")
            continue

        sub_samp = random.sample(arr, min(num_samp, len(arr)))
        for i, obj in enumerate(sub_samp, 1):
            with st.expander(f"[{cat}] Sample #{i}", expanded=True):
                meta_parts = []
                if obj["id"]:
                    meta_parts.append(f"ID: {obj['id']}")
                if obj["age"]:
                    meta_parts.append(f"Age: {obj['age']}")
                if obj["gender"]:
                    meta_parts.append(f"Gender: {obj['gender']}")
                if obj["jobtitle"]:
                    meta_parts.append(f"Job: {obj['jobtitle']}")
                if obj["region"]:
                    meta_parts.append(f"Region: {obj['region']}")
                if meta_parts:
                    st.markdown("*" + " | ".join(meta_parts) + "*")

                st.write(obj["text"])

                # FIX HERE: 3-tuple key instead of 2-tuple
                dict_key = (obj["id"], obj["surveyid"], variable)

                if dict_key not in st.session_state.open_coding_assignments:
                    st.session_state.open_coding_assignments[dict_key] = {
                        "primary_code": "Unassigned",
                        "secondary_code": "Unassigned"
                    }

                assigned_obj_ = st.session_state.open_coding_assignments[dict_key]
                grp_list = ["Unassigned"] + [g["name"] for g in st.session_state.open_coding_groups]

                # Primary
                st.write("**Primary Code**:")
                p_current = assigned_obj_.get("primary_code", "Unassigned")
                if p_current not in grp_list:
                    p_current = "Unassigned"
                new_p = st.selectbox(
                    f"Assign primary code (Sample {i})",
                    options=grp_list,
                    index=grp_list.index(p_current) if p_current in grp_list else 0,
                    key=f"sample_{cat}_{i}_primary"
                )
                assigned_obj_["primary_code"] = new_p

                # Secondary
                st.write("**Secondary Code**:")
                s_current = assigned_obj_.get("secondary_code", "Unassigned")
                if s_current not in grp_list:
                    s_current = "Unassigned"
                new_s = st.selectbox(
                    f"Assign secondary code (Sample {i})",
                    options=grp_list,
                    index=grp_list.index(s_current) if s_current in grp_list else 0,
                    key=f"sample_{cat}_{i}_secondary"
                )
                assigned_obj_["secondary_code"] = new_s

    if st.button("🎲 Save and Shuffle"):
        ok = update_coded_assignments(variable, final_df, df_updated=None)
        if ok:
            st.session_state["sample_seed"] = int(time.time())
            st.success("All coding saved. Shuffled a new set of samples.")
            st.rerun()
        else:
            st.error("Error saving coding.")

def render_group_manager():
    # 1) Always load the CSV to show the real on-disk group info
    df_groups = pd.DataFrame(load_open_coding_groups('cache/groups.csv'))
    if df_groups.empty:
        df_groups = pd.DataFrame(columns=["name", "desc"])  # Add more cols if needed

    st.subheader("Manage Coding Groups")

    # 2) Add or update a group
    new_group_name = st.text_input("Group Name (new or existing)")
    new_group_desc = st.text_input("Group Description")

    if st.button("➕ Add / Update Group"):
        gname = new_group_name.strip()
        if gname:
            # Check if that name already exists
            existing_idx = df_groups.index[df_groups["name"] == gname].tolist()
            if existing_idx:
                # Update the existing group's description
                for idx in existing_idx:
                    df_groups.at[idx, "desc"] = new_group_desc.strip()
                st.success(f"Updated group '{gname}'.")
            else:
                # Append a new row
                new_row = pd.DataFrame({"name": [gname], "desc": [new_group_desc.strip()]})
                df_groups = pd.concat([df_groups, new_row], ignore_index=True)
                st.success(f"Created new group '{gname}'.")

            # Save to disk
            df_groups.to_csv('cache/groups.csv', index=False)
            # Also update session state so the rest of your code sees changes
            st.session_state.open_coding_groups = df_groups.to_dict('records')
            st.rerun()
        else:
            st.warning("Please enter a valid group name.")

    # 3) Delete a group
    with st.expander("Delete Groups"):
        all_group_names = df_groups["name"].tolist()
        if all_group_names:
            delete_choice = st.selectbox("❌ Select a group to remove", ["(None)"] + all_group_names)
            if delete_choice != "(None)":
                if st.button(f"🗑️ Remove Group '{delete_choice}'"):
                    # Remove from DataFrame
                    df_groups = df_groups[df_groups["name"] != delete_choice]

                    # Also unassign references in your coded assignments
                    for (k_id, k_surveyid, k_var), valdict in list(st.session_state.open_coding_assignments.items()):
                        if valdict.get("primary_code") == delete_choice:
                            valdict["primary_code"] = "Unassigned"
                        if valdict.get("secondary_code") == delete_choice:
                            valdict["secondary_code"] = "Unassigned"

                    df_groups.to_csv('cache/groups.csv', index=False)
                    st.session_state.open_coding_groups = df_groups.to_dict('records')
                    st.success(f"Removed group '{delete_choice}'.")
                    st.rerun()

    # 4) Optional: Merge or combine groups
    with st.expander("Combine Groups"):
        groups_to_merge = st.multiselect(
            "Select two or more groups to combine",
            all_group_names
        )
        merged_group_name = st.text_input("New Merged Group Name")
        merged_group_desc = st.text_input("Merged Group Description")

        if st.button("Combine & Save"):
            if len(groups_to_merge) < 2:
                st.warning("Please select at least two groups to combine.")
            elif not merged_group_name.strip():
                st.warning("Please provide a valid name for the new group.")
            else:
                # Update any assignments in session_state
                for (k_id, k_surveyid, k_var), valdict in list(st.session_state.open_coding_assignments.items()):
                    if valdict["primary_code"] in groups_to_merge:
                        valdict["primary_code"] = merged_group_name.strip()
                    if valdict["secondary_code"] in groups_to_merge:
                        valdict["secondary_code"] = merged_group_name.strip()

                # Remove old groups from df_groups
                df_groups = df_groups[~df_groups["name"].isin(groups_to_merge)]

                # Add the merged group
                df_groups = pd.concat([
                    df_groups,
                    pd.DataFrame({
                        "name": [merged_group_name.strip()],
                        "desc": [merged_group_desc.strip()]
                    })
                ], ignore_index=True)

                # Save back to CSV
                df_groups.to_csv('cache/groups.csv', index=False)
                st.session_state.open_coding_groups = df_groups.to_dict('records')
                st.success(f"Merged {groups_to_merge} into '{merged_group_name.strip()}' successfully.")
                st.rerun()

    with st.expander("Existing Groups"):
        if df_groups.empty:
            st.info("No coding groups found in the CSV.")
        else:
            st.dataframe(df_groups, use_container_width=True)


################################################################################
# WORDCLOUD FUNCTIONS
################################################################################

def generate_word_freq(texts, exact_words=200):
    processed_texts = []
    for txt in texts:
        cleaned_txt = process_text(
            txt,
            stopwords=st.session_state.custom_stopwords,  # Pass permanent stops...
            synonym_groups=st.session_state.synonym_groups
        )
        if cleaned_txt:
            processed_texts.append(cleaned_txt)

    # Build frequency data from processed_texts
    merged = " ".join(processed_texts)
    freq_counter = Counter(merged.split())
    return freq_counter.most_common(exact_words)



def generate_interactive_wordcloud(freq_data, highlight_words=None, title='', exact_words=200, colormap='viridis'):
    """
    Returns a Plotly figure (scatter-based wordcloud).
    - highlight_words (set): highlight in a viridis-like color (#443983), else grey
    - exact_words: top words to consider
    - colormap is only used if highlight_words is empty; otherwise highlight logic is used.
    """
    if not freq_data:
        return None

    freq_data = freq_data[:exact_words]
    words = [w for w, _ in freq_data]
    freqs = [f for _, f in freq_data]
    max_f = max(freqs) if freqs else 1
    sizes = [20 + (f / max_f) * 60 for f in freqs]

    import numpy as np
    golden_ratio = (1 + 5 ** 0.5) / 2
    positions = []
    for i, f_val in enumerate(freqs):
        r = (1 - f_val / max_f) * 200
        theta = i * 2 * np.pi * golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append((x, y))

    def nearest_terms(idx, k=3):
        xi, yi = positions[idx]
        dists = []
        for j, (xx, yy) in enumerate(positions):
            if j != idx:
                d_sq = (xi - xx) ** 2 + (yi - yy) ** 2
                dists.append((d_sq, j))
        dists.sort(key=lambda x: x[0])
        neigh = [words[d[1]] for d in dists[:k]]
        return ", ".join(neigh)

    custom_data = []
    color_list = []

    if highlight_words:
        # highlight in #443983, else gray
        for i, w in enumerate(words):
            highlight = (w.lower() in highlight_words)
            color_list.append("#443983" if highlight else "gray")
            custom_data.append((freqs[i], w, nearest_terms(i, 3)))
    else:
        # Use the new colormaps approach
        import numpy as np
        import random
        random_state = np.random.RandomState(42)
        selected_cmap = get_cmap_fixed(colormap)

        for i, w in enumerate(words):
            color_idx = random_state.randint(0, 256)
            r, g, b, _ = selected_cmap(color_idx / 255.0)
            color_list.append(f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})")
            custom_data.append((freqs[i], w, nearest_terms(i, 3)))

    x_vals, y_vals = zip(*positions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        text=words,
        mode='text+markers',
        textposition='middle center',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        textfont=dict(size=sizes, color=color_list),
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Frequency: %{customdata[0]}<br>"
            "Closest Terms: %{customdata[2]}<extra></extra>"
        )
    ))
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        width=900, height=700,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   scaleanchor='x', scaleratio=1),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig


def render_wordclouds(var_resps, var_name="open_var"):
    """
    Renders multiple tabs of wordcloud outputs (Static, Interactive, etc.)
    and includes the variable name in downloaded filenames.
    """

    layout_choice = st.selectbox("Number of quadrants", [1, 2, 3, 4], index=0)
    groups_available = sorted(list(var_resps.keys()))

    quadrant_selections = {}
    for q_idx in range(layout_choice):
        quadrant_label = f"Quadrant {chr(65 + q_idx)}"
        quadrant_selections[quadrant_label] = st.multiselect(
            f"Select categories for {quadrant_label}",
            groups_available,
            []
        )

    exact_words = st.slider(
        "Exact number of words per quadrant",
        min_value=10,
        max_value=300,
        value=50,
        step=10
    )

    # For overall coloring if no highlight scheme is used
    color_schemes = [
        "viridis", "plasma", "inferno", "magma", "cividis", "winter",
        "coolwarm", "bone", "terrain", "twilight"
    ]
    chosen_cmap = st.selectbox("Color Scheme (if no highlights)", color_schemes, index=0)

    highlight_input = st.text_area("Highlight words (one per line)", "")
    highlight_set = {h.strip().lower() for h in highlight_input.split('\n') if h.strip()}

    # Combine text for each quadrant
    quadrant_texts = {}
    for quad_label, cat_list in quadrant_selections.items():
        combined_texts = []
        for cat in cat_list:
            combined_texts.extend(var_resps[cat])
        quadrant_texts[quad_label] = combined_texts

    # Check if there's data
    nonempty_quadrants = any(len(txts) > 0 for txts in quadrant_texts.values())
    if not nonempty_quadrants:
        st.warning("No groups selected or no data to show.")
        return

    # Generate frequencies for each quadrant
    quadrant_freqs = {}
    for quad_label, txt_list in quadrant_texts.items():
        freq_ = generate_word_freq(txt_list, exact_words=exact_words)
        quadrant_freqs[quad_label] = freq_

    # Prepare CSV download
    all_quadrants_freq = []
    for quad_label, freq_dat in quadrant_freqs.items():
        for w, f_ in freq_dat:
            all_quadrants_freq.append([quad_label, w, f_])
    df_all_quadrants_freq = pd.DataFrame(all_quadrants_freq, columns=["Quadrant", "Word", "Frequency"])
    csv_all_freq = df_all_quadrants_freq.to_csv(index=False).encode('utf-8')

    # Let user decide if they want per-quadrant formatting or single shared settings
    use_unified_formatting = st.checkbox("Use the same 'Advanced' formatting for all quadrants?", value=True)

    # A helper to collect advanced settings
    def collect_advanced_settings(quad_label=None):
        """
        Return a dictionary of advanced settings for one quadrant (or shared).
        Add unique keys so Streamlit won't raise 'StreamlitDuplicateElementId'.
        """
        label_txt = f"Quadrant {quad_label} Formatting" if quad_label else "Advanced Formatting (shared)"
        with st.expander(label_txt):
            spiral_type = st.selectbox(
                f"Shape Type - {quad_label}",
                ["archimedean", "rectangular", "default"],
                index=0,
                key=f"spiral_type_{quad_label}"
            )

            use_gradient = st.checkbox(
                "Apply gradient color by frequency?",
                value=False,
                key=f"use_gradient_{quad_label}"
            )

            use_custom_freq_colors = st.checkbox(
                "Use custom ascending color for frequencies?",
                value=False,
                key=f"use_custom_freq_colors_{quad_label}"
            )

            low_freq_color = "#D3D3D3"
            high_freq_color = "#000000"
            if use_custom_freq_colors:
                low_freq_color = st.color_picker(
                    "Low frequency color",
                    value="#D3D3D3",
                    key=f"low_freq_color_{quad_label}"
                )
                high_freq_color = st.color_picker(
                    "High frequency color",
                    value="#000000",
                    key=f"high_freq_color_{quad_label}"
                )

            # Example mask options
            shape_options = ["None (no mask)", "Transparent Background", "some_mask.png"]
            shape_option = st.selectbox(
                f"Wordcloud Shape (mask) - {quad_label}",
                shape_options,
                index=0,
                key=f"shape_option_{quad_label}"
            )
            region_option = None
            show_icon_behind = False
            icon_size = 500
            if shape_option not in ("None (no mask)", "Transparent Background"):
                region_option = st.radio(
                    f"Fill words inside or outside the icon - {quad_label}",
                    ["Inside the black icon", "Outside (in the white region)"],
                    index=0,
                    key=f"region_option_{quad_label}"
                )
                icon_size = st.slider(
                    f"Mask Pixelation/Size - {quad_label}",
                    min_value=200,
                    max_value=2000,
                    value=500,
                    step=100,
                    key=f"icon_size_{quad_label}"
                )
                show_icon_behind = st.checkbox(
                    f"Show icon behind words - {quad_label}",
                    value=False,
                    key=f"show_icon_behind_{quad_label}"
                )

            contour_checkbox = st.checkbox(
                f"Add a contour around words? - {quad_label}",
                value=False,
                key=f"contour_checkbox_{quad_label}"
            )
            contour_color = "#000000"
            if contour_checkbox:
                contour_color = st.color_picker(
                    f"Contour color - {quad_label}",
                    value="#000000",
                    key=f"contour_color_{quad_label}"
                )

        return {
            "spiral_type": spiral_type,
            "use_gradient": use_gradient,
            "use_custom_freq_colors": use_custom_freq_colors,
            "low_freq_color": low_freq_color,
            "high_freq_color": high_freq_color,
            "shape_option": shape_option,
            "region_option": region_option,
            "show_icon_behind": show_icon_behind,
            "icon_size": icon_size,
            "contour_checkbox": contour_checkbox,
            "contour_color": contour_color,
        }

    if use_unified_formatting:
        # Collect once and reuse
        shared_format = collect_advanced_settings()
        quad_formatting = {q_label: shared_format for q_label in quadrant_freqs.keys()}
    else:
        # Collect for each quadrant individually
        quad_formatting = {}
        for q_label in quadrant_freqs.keys():
            quad_formatting[q_label] = collect_advanced_settings(quad_label=q_label)

    tabs = st.tabs(["📸 Static", "🔄 Interactive"])

    # ----------------------------------------------------------------------------------------
    # TAB 0: STATIC WORDCLOUD
    # ----------------------------------------------------------------------------------------
    with tabs[0]:

        # Because each quadrant can differ, we'll build subplots accordingly
        fig_cols = 2 if layout_choice > 1 else 1
        fig_rows = (layout_choice + 1) // 2 if layout_choice > 2 else (1 if layout_choice < 3 else 2)

        # Increase figure size slightly but rely on large WordCloud resolution to keep clarity
        fig = plt.figure(figsize=(8 * fig_cols, 6 * fig_rows))
        gs = fig.add_gridspec(fig_rows, fig_cols)

        # We'll define a color blender to handle custom freq colors
        def blend_hex_colors(hex1, hex2, t):
            def hex_to_rgb(hx):
                hx = hx.lstrip('#')
                return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))
            r1, g1, b1 = hex_to_rgb(hex1)
            r2, g2, b2 = hex_to_rgb(hex2)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            return f"rgb({r},{g},{b})"

        idx = 0
        for quad_label, freq_dat in quadrant_freqs.items():
            row_ = idx // fig_cols
            col_ = idx % fig_cols
            ax_ = fig.add_subplot(gs[row_, col_])

            # Grab the advanced settings for this quadrant
            fmt = quad_formatting[quad_label]
            shape_option = fmt["shape_option"]
            region_option = fmt["region_option"]
            show_icon_behind = fmt["show_icon_behind"]
            icon_size = fmt["icon_size"]
            contour_checkbox = fmt["contour_checkbox"]
            contour_color = fmt["contour_color"]
            use_gradient = fmt["use_gradient"]
            use_custom_freq_colors = fmt["use_custom_freq_colors"]
            low_freq_color = fmt["low_freq_color"]
            high_freq_color = fmt["high_freq_color"]
            spiral_type = fmt["spiral_type"]

            if freq_dat:
                freq_dict = dict(freq_dat)
                max_count = max(freq_dict.values()) if freq_dict else 1
                min_count = min(freq_dict.values()) if freq_dict else 0

                # Decide how the colors are assigned
                if use_custom_freq_colors:
                    def custom_asc_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                        f_ = freq_dict.get(word, 0)
                        scale = (f_ - min_count) / (max_count - min_count) if max_count > min_count else 0.0
                        return blend_hex_colors(low_freq_color, high_freq_color, scale)
                    wc_color_func = custom_asc_color_func

                elif highlight_set and not use_gradient:
                    # Non-gradient approach, highlight words in highlight_set
                    def color_func(word, *args, **kwargs):
                        return "rgb(68, 57, 131)" if word.lower() in highlight_set else "gray"
                    wc_color_func = color_func

                elif use_gradient:
                    selected_cmap = get_cmap_fixed(chosen_cmap)
                    def gradient_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                        f_ = freq_dict.get(word, 0)
                        scale = (f_ - min_count) / (max_count - min_count) if max_count > min_count else 0
                        r, g, b, _ = selected_cmap(scale)
                        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
                    wc_color_func = gradient_color_func

                else:
                    random_state = np.random.RandomState(42)
                    selected_cmap = get_cmap_fixed(chosen_cmap)
                    def random_cmap_color_func(word, font_size, position, orientation, random_state=random_state, **kwargs):
                        color_idx = random_state.randint(0, 256)
                        r, g, b, _ = selected_cmap(color_idx / 255.0)
                        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
                    wc_color_func = random_cmap_color_func

                # Build the mask (if any)
                mask = None
                transparent_bg = False
                icon_img = None

                if shape_option == "Transparent Background (No Bounding Box)":
                    transparent_bg = True
                elif shape_option not in ("None (no mask)", "Transparent Background (No Bounding Box)"):
                    try:
                        mask_path = os.path.join("masks", shape_option)
                        loaded_icon = Image.open(mask_path).convert("RGBA")
                        # Resize icon
                        loaded_icon = loaded_icon.resize((icon_size, icon_size), Image.LANCZOS)
                        icon_img = loaded_icon.copy()

                        # Merge alpha with white background so we can threshold
                        white_bg = Image.new("RGBA", loaded_icon.size, "WHITE")
                        merged_img = Image.alpha_composite(white_bg, loaded_icon)

                        mask_array = np.array(merged_img)
                        # Threshold for "black"
                        rgb_sum = mask_array[..., :3].sum(axis=-1)
                        threshold = 40
                        if region_option == "Outside (in the white region)":
                            mask = np.where(rgb_sum < threshold, 255, 0).astype(np.uint8)
                        else:
                            mask = np.where(rgb_sum < threshold, 0, 255).astype(np.uint8)

                        ratio_unmasked = (mask > 0).sum() / float(mask.size)
                        if ratio_unmasked < 0.05:
                            st.warning(
                                f"Quadrant '{quad_label}' mask is very restrictive: "
                                "More than 95% of the area is masked. "
                                "You might get drawing errors."
                            )
                    except Exception as e:
                        st.warning(f"Could not load or process {shape_option} from 'masks': {e}")
                        mask = None
                        icon_img = None

                # Build the wordcloud with large resolution to avoid pixelation
                wc_params = {
                    "width": 1600,        # increased for higher default pixel resolution
                    "height": 1200,       # increased for higher default pixel resolution
                    "collocations": False,
                    "max_words": exact_words,
                    "color_func": wc_color_func,
                    "mask": mask,
                    "contour_width": 5 if contour_checkbox else 0,
                    "contour_color": contour_color if contour_checkbox else None,
                    "prefer_horizontal": 1.0,  # slightly preference for horizontal words
                }

                if transparent_bg:
                    wc_params["mode"] = "RGBA"
                    wc_params["background_color"] = None
                    wc_params["margin"] = 1
                else:
                    wc_params["background_color"] = "white"

                # Optionally set the spiral type
                if spiral_type in ["archimedean", "rectangular"]:
                    wc_params["prefer_horizontal"] = 0.9 if spiral_type == "archimedean" else 0.8
                    wc_params["collocation_threshold"] = 2

                wc = WordCloud(**wc_params)

                # Build text from frequencies
                word_list = []
                for w, f_ in freq_dat:
                    word_list.extend([w] * f_)
                joined_text = ' '.join(word_list)

                try:
                    wc.generate(joined_text)

                    # Convert the wordcloud to an RGBA image
                    wc_img = wc.to_image()

                    # If user wants to show the icon behind words, alpha-composite them
                    if show_icon_behind and icon_img is not None:
                        # We'll ensure the background icon matches the WC size
                        icon_img = icon_img.resize((wc_img.width, wc_img.height), Image.LANCZOS)
                        # Alpha-composite: icon behind, then WC on top
                        combined_img = Image.alpha_composite(icon_img, wc_img)
                        ax_.imshow(combined_img, interpolation='bilinear')
                    else:
                        # Just show the wordcloud
                        ax_.imshow(wc_img, interpolation='bilinear')

                    ax_.axis('off')

                except ValueError as ve:
                    ax_.text(
                        0.5, 0.5,
                        f"Mask too restrictive:\n{ve}",
                        ha='center',
                        va='center',
                        fontsize=12,
                        color='red'
                    )
            else:
                ax_.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)

            idx += 1

        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="💾 Download Static Wordcloud PNG",
            data=buf.getvalue(),
            file_name=f"{var_name}_wordcloud_quadrants_static.png",
            mime="image/png",
            use_container_width=True
        )
        plt.close(fig)

        st.download_button(
            label="📥 Download Frequencies CSV",
            data=csv_all_freq,
            file_name=f"{var_name}_wordcloud_quadrants_frequencies.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ----------------------------------------------------------------------------------------
    # TAB 1: INTERACTIVE WORDCLOUD
    # ----------------------------------------------------------------------------------------
    with tabs[1]:
        rows_ = (layout_choice + 1) // 2 if layout_choice > 2 else (1 if layout_choice < 3 else 2)
        cols_ = 2 if layout_choice > 1 else 1
        fig_int = make_subplots(rows=rows_, cols=cols_, subplot_titles=None)

        idx = 1
        for quad_label, freq_dat in quadrant_freqs.items():
            row_ = (idx - 1) // cols_ + 1
            col_ = (idx - 1) % cols_ + 1

            iwc = generate_interactive_wordcloud(
                freq_dat,
                highlight_words=highlight_set,
                title='',
                exact_words=exact_words,
                colormap=chosen_cmap
            )
            if iwc:
                for trace in iwc.data:
                    fig_int.add_trace(trace, row=row_, col=col_)
                fig_int.update_xaxes(visible=False, showgrid=False, zeroline=False, row=row_, col=col_)
                fig_int.update_yaxes(visible=False, showgrid=False, zeroline=False, row=row_, col=col_)

            idx += 1

        fig_int.update_layout(
            width=1000 + 400 * (cols_ - 1),
            height=600 * rows_,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            title="",
        )
        st.plotly_chart(fig_int, use_container_width=True)

        st.download_button(
            label="Download Frequencies CSV",
            data=csv_all_freq,
            file_name=f"{var_name}_wordcloud_quadrants_frequencies.csv",
            mime="text/csv",
            use_container_width=True
        )


###############################################################################
# 8) WORD DIVE
###############################################################################
def flesch_kincaid_grade_level(text):
    """
    Rough, self-contained Flesch-Kincaid grade level calculation.
    """
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if s.strip()]
    num_sents = len(sents) if sents else 1

    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return 0.0

    def syllable_count(w):
        return len(re.findall(r'[aeiou]+', w.lower()))

    total_syllables = sum(syllable_count(w) for w in words)
    # Flesch-Kincaid formula: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
    return 0.39 * (num_words / num_sents) + 11.8 * (total_syllables / num_words) - 15.59

def render_word_dive(chosen_var, rdict, open_var_options, grouping_columns):
    """
    Extended "Word Dive" section analyzing usage of a specific word across:
      - Multiple grouping variables
      - Flesch-Kincaid complexity
      - Common bigrams/trigrams including the focus term
      - Example responses near the analyses with interactive sampling

    rdict: dictionary of dataframes keyed by "survey id" or similar
    chosen_var: name of the open-ended variable to analyze
    open_var_options: dictionary mapping variable names to display text
    grouping_columns: potential grouping columns the user can pick from
    """

    st.markdown("## 🏊 Word Dive")

    # User picks the focus term
    focus_term = st.text_input("Enter the term to examine", "work")

    # Optional grouping column
    st.markdown("### Choose Grouping Column")
    grouping_choice = st.selectbox("Group by column", [None, "None"] + grouping_columns, index=0)
    if grouping_choice in [None, "None"]:
        grouping_choice = None

    # Gather texts by group
    group_texts = defaultdict(list)
    # We might also keep track of total responses by group (for proportion)
    group_counts = defaultdict(int)

    for sid, df in rdict.items():
        if chosen_var not in df.columns:
            continue
        sub_df = df.dropna(subset=[chosen_var])
        if grouping_choice and (grouping_choice in sub_df.columns):
            for grp_val, gdf in sub_df.groupby(grouping_choice):
                valid_texts = gdf[chosen_var].dropna().astype(str).tolist()
                label_name = f"{sid}_{grp_val}"
                group_texts[label_name].extend(valid_texts)
                group_counts[label_name] += len(valid_texts)
        else:
            # no group chosen, treat the entire survey as a single "group"
            group_texts[sid].extend(sub_df[chosen_var].dropna().astype(str).tolist())
            group_counts[sid] += len(sub_df[chosen_var].dropna())

    # Prepare data frames for analysis
    results = []
    distribution_data = []  # For box/violin plots of F-K

    for grp, txts in group_texts.items():
        # How many contain the focus term?
        total_in_group = len(txts)
        if total_in_group == 0:
            # no responses
            results.append({
                "Group": grp,
                "Total_Responses": 0,
                "Num_Responses_Containing_Term": 0,
                "Proportion_Containing_Term": 0.0,
                "Avg_FK_Grade": None
            })
            continue

        # Filter only those responses containing the focus term
        focus_responses = [t for t in txts if focus_term.lower() in t.lower()]
        num_contains = len(focus_responses)
        proportion_contains = num_contains / total_in_group

        if focus_responses:
            fk_scores = []
            for r_ in focus_responses:
                fk_val = flesch_kincaid_grade_level(r_)
                fk_scores.append(fk_val)
                # For distribution chart
                distribution_data.append({
                    "Group": grp,
                    "FK_Grade": fk_val
                })
            avg_fk = sum(fk_scores) / len(fk_scores)
        else:
            avg_fk = None

        results.append({
            "Group": grp,
            "Total_Responses": total_in_group,
            "Num_Responses_Containing_Term": num_contains,
            "Proportion_Containing_Term": round(proportion_contains, 3),
            "Avg_FK_Grade": round(avg_fk, 2) if avg_fk is not None else None
        })

    df_summary = pd.DataFrame(results)

    # Build the layout with tabs for different analyses:
    dive_tabs = st.tabs(["Focus Term Prevalence",
                         "F-K Complexity & Distribution",
                         "Bigrams/Trigrams"])

    # =============== TAB 1: Focus Term Prevalence ====================
    with dive_tabs[0]:
        st.markdown("### Focus Term Prevalence by Group")
        st.dataframe(df_summary, use_container_width=True)

        # Plotly bar chart: Number of responses containing the term (or Proportion)
        st.markdown("#### Bar Chart: Proportion of Responses Containing the Term")
        if not df_summary.empty:
            fig_prop = px.bar(
                df_summary,
                x="Group",
                y="Proportion_Containing_Term",
                hover_data=["Num_Responses_Containing_Term", "Total_Responses"],
                title=f"Proportion of Responses Containing '{focus_term}' by Group",
                color="Proportion_Containing_Term",
                color_continuous_scale="Blues",
            )
            fig_prop.update_layout(
                xaxis_title="Group",
                yaxis_title="Proportion Containing the Term",
                coloraxis_showscale=False,
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_prop, use_container_width=True)

            # Another bar chart: raw counts
            st.markdown("#### Bar Chart: Number of Responses Containing the Term")
            fig_count = px.bar(
                df_summary,
                x="Group",
                y="Num_Responses_Containing_Term",
                hover_data=["Proportion_Containing_Term", "Total_Responses"],
                title=f"Count of Responses Containing '{focus_term}' by Group",
                color="Num_Responses_Containing_Term",
                color_continuous_scale="Teal",
            )
            fig_count.update_layout(
                xaxis_title="Group",
                yaxis_title="Count Containing the Term",
                coloraxis_showscale=False,
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_count, use_container_width=True)
        else:
            st.write("No data found for this variable or no responses in rdict.")

    # =============== TAB 2: F-K Complexity & Distribution ============
    with dive_tabs[1]:
        st.markdown("### Flesch-Kincaid Complexity by Group")

        # Show the summary
        st.dataframe(df_summary[["Group", "Avg_FK_Grade",
                                 "Num_Responses_Containing_Term"]].sort_values(
            by="Avg_FK_Grade", ascending=False
        ), use_container_width=True)

        # Build distribution chart (box plot or violin) if we have data
        dist_df = pd.DataFrame(distribution_data)
        if not dist_df.empty:
            # Box plot
            st.markdown("#### Box Plot of F-K Grade Levels (Responses Containing the Term)")
            fig_box = px.box(
                dist_df,
                x="Group",
                y="FK_Grade",
                color="Group",
                title="F-K Grade Distribution by Group (Focus Term Responses)",
            )
            fig_box.update_layout(
                showlegend=False,
                xaxis_title="Group",
                yaxis_title="F-K Grade Level",
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Violin plot
            st.markdown("#### Violin Plot of F-K Grade Levels")
            fig_violin = px.violin(
                dist_df,
                x="Group",
                y="FK_Grade",
                color="Group",
                box=True,
                points="all",
                title="F-K Grade Distribution by Group (Focus Term Responses) - Violin Plot",
            )
            fig_violin.update_layout(
                showlegend=False,
                xaxis_title="Group",
                yaxis_title="F-K Grade Level",
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.write("No responses contained the focus term, so no F-K distributions to show.")

    # =============== TAB 3: Bigrams/Trigrams =========================
    with dive_tabs[2]:
        st.markdown("### Most Common Bigrams/Trigrams Containing the Focus Term")
        top_n = st.slider("How many top bigrams/trigrams to show?", 5, 30, 10)

        for grp_key, txts in group_texts.items():
            st.subheader(f"Group: {grp_key}")
            relevant = [t for t in txts if focus_term.lower() in t.lower()]
            if not relevant:
                st.write("No responses contain the focus term.")
                continue

            # If you stored custom stopwords in SessionState:
            custom_stops = None
            if "custom_stopwords" in st.session_state:
                custom_stops = list(st.session_state.custom_stopwords)

            vec = CountVectorizer(ngram_range=(2, 3), stop_words=custom_stops)

            # --- FIX: Catch empty vocabulary to avoid ValueError ---
            try:
                X = vec.fit_transform(relevant)
            except ValueError as e:
                # Often "empty vocabulary; perhaps the documents only contain stop words"
                st.warning(
                    f"Skipping {grp_key} because we got an empty vocabulary. "
                    "Check if all text is extremely short or all stop words."
                )
                continue

            freqs = X.sum(axis=0).A1
            vocab = vec.get_feature_names_out()

            focus_ngram_counts = []
            for i, v in enumerate(vocab):
                if focus_term.lower() in v:
                    focus_ngram_counts.append((v, freqs[i]))

            if not focus_ngram_counts:
                st.write("No bigrams/trigrams containing the focus term.")
                continue

            focus_ngram_counts.sort(key=lambda x: x[1], reverse=True)
            top_ngrams = focus_ngram_counts[:top_n]
            df_ngrams = pd.DataFrame(top_ngrams, columns=["Ngram", "Frequency"])
            st.dataframe(df_ngrams, use_container_width=True)

            fig_ngrams = px.bar(
                df_ngrams,
                x="Ngram",
                y="Frequency",
                title=f"Top Bigrams/Trigrams (containing '{focus_term}') in {grp_key}",
                color="Frequency",
                color_continuous_scale="Reds",
            )
            fig_ngrams.update_layout(
                xaxis_title="N-gram",
                yaxis_title="Frequency",
                coloraxis_showscale=False,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_ngrams, use_container_width=True)

###############################################################################
# 9) TOPIC DISCOVERY
###############################################################################
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

def create_3d_topic_visualization(df,
                                  text_col="open_response",
                                  topic_labels=None,
                                  jobtitle_col="jobtitle",
                                  age_col="age",
                                  gender_col="gender",
                                  region_col="region",
                                  model_name='all-MiniLM-L6-v2',
                                  n_components=3,
                                  n_clusters=5,
                                  random_state=42):
    """
    Create a 3D interactive scatter plot showing topics for each response.

    Parameters:
    -----------
    df              : pd.DataFrame
                     Must contain the open_response column plus columns for jobtitle, age, etc.
    text_col        : str, name of the column with text
    topic_labels    : List or dict that maps each row's topic index to a label or cluster name.
                      If None, we'll just use KMeans with n_clusters.
    jobtitle_col    : str, name of the job title column
    age_col         : str, name of the age column
    gender_col      : str, name of the gender column
    region_col      : str, name of the region/state column
    model_name      : str, the SentenceTransformer model to use
    n_components    : int, dimension for TSNE (3 recommended here)
    n_clusters      : int, how many clusters if we do a quick KMeans for topic labeling
    random_state    : int, random seed

    Returns:
    --------
    fig : Plotly Figure (3D scatter)
    """

    # Filter valid texts
    df_valid = df.dropna(subset=[text_col]).copy()
    df_valid = df_valid[df_valid[text_col].astype(str).str.strip() != ""]
    if df_valid.empty:
        st.warning("No valid text responses for 3D topic visualization.")
        return None

    # Embed
    embedder = SentenceTransformer(model_name)
    text_list = df_valid[text_col].astype(str).tolist()
    embeddings = embedder.encode(text_list, show_progress_bar=False)

    # If topic_labels isn't provided, we do a quick KMeans
    if topic_labels is None:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(embeddings)
        df_valid["topic_label"] = kmeans.labels_
        # We'll map cluster IDs to strings like "Topic 0", "Topic 1", ...
        df_valid["topic_label"] = df_valid["topic_label"].apply(lambda x: f"Topic {x}")
    else:
        # We assume df already has a topic_label column matching each row
        # Or topic_labels is a dictionary keyed by text or index
        # e.g. df_valid["topic_label"] = [topic_labels.get(i, "Unknown") for i in df_valid.index]
        if "topic_label" not in df_valid.columns:
            # fallback: try to apply the dictionary
            df_valid["topic_label"] = df_valid.index.map(lambda i: topic_labels.get(i, "Unknown"))

    # TSNE 3D
    tsne = TSNE(n_components=n_components, random_state=random_state)
    reduced_3d = tsne.fit_transform(embeddings)

    df_valid["x"] = reduced_3d[:, 0]
    df_valid["y"] = reduced_3d[:, 1]
    if n_components == 3:
        df_valid["z"] = reduced_3d[:, 2]
    else:
        # fallback for 2D if n_components = 2
        df_valid["z"] = 0.0

    # Build hover text
    # e.g. "Job: ...<br>Age: ...<br>Gender: ...<br>Region: ...<br>Response: ..."
    hover_info = []
    for idx, row in df_valid.iterrows():
        job_ = str(row.get(jobtitle_col, ""))
        age_ = str(row.get(age_col, ""))
        gen_ = str(row.get(gender_col, ""))
        reg_ = str(row.get(region_col, ""))
        txt_ = str(row.get(text_col, ""))
        htxt = f"Job: {job_}<br>Age: {age_}<br>Gender: {gen_}<br>Region: {reg_}<br>{txt_}"
        hover_info.append(htxt)

    df_valid["hover_text"] = hover_info

    # Plot
    fig = px.scatter_3d(
        df_valid,
        x="x", y="y", z="z",
        color="topic_label",
        hover_data={"hover_text": True, "x": False, "y": False, "z": False},
        symbol=None
    )
    fig.update_traces(hovertemplate="%{customdata[0]}")
    fig.update_layout(
        title="3D Topic Visualization",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        legend_title_text="Topic"
    )

    return fig

def integrate_sentiment_into_df(df, text_col="open_response"):
    """
    Example helper that adds sentiment info into a DataFrame using your existing
    sentiment analysis function `analyze_sentiment`.
    Modify if your function name/logic is different.

    Returns:
    --------
    df_new : pd.DataFrame with new columns "sentiment_category" and "sentiment_compound"
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    df = df.copy()
    cat_list = []
    comp_list = []

    for txt in df[text_col]:
        if not isinstance(txt, str) or not txt.strip():
            cat_list.append("Neutral")
            comp_list.append(0.0)
            continue
        scores = analyzer.polarity_scores(txt)
        comp = scores["compound"]
        if comp >= 0.05:
            cat = "Positive"
        elif comp <= -0.05:
            cat = "Negative"
        else:
            cat = "Neutral"
        cat_list.append(cat)
        comp_list.append(comp)

    df["sentiment_category"] = cat_list
    df["sentiment_compound"] = comp_list

    return df

def show_example_responses(df,
                           text_col="open_response",
                           group_col="group_var",
                           topic_col="topic_label",
                           sentiment_col="sentiment_category",
                           theme_col="theme_label",
                           max_examples=5):
    """
    Displays example responses in multiple tabs:
      - By the grouping variable (group_col)
      - By topic (topic_col)
      - By sentiment (sentiment_col)
      - By theme (theme_col)

    Each tab will show sub-tabs for each category, and up to 'max_examples' random examples.
    If you only want a subset, pass None for whichever grouping you don't need.

    NOTE: Make sure your df has columns for each of these if you want them displayed.
    """

    import random

    # We'll define a helper to build per-category random examples
    def render_category_examples(subdf, category_value, text_column=text_col, max_n=5):
        subdf = subdf.dropna(subset=[text_column])
        subdf = subdf[subdf[text_column].astype(str).str.strip() != ""]
        if subdf.empty:
            st.write("No examples for this category.")
            return
        if len(subdf) <= max_n:
            examples = subdf[text_column].tolist()
        else:
            examples = subdf.sample(n=max_n, random_state=42)[text_column].tolist()

        for i, ex in enumerate(examples, 1):
            st.markdown(f"**Example {i}:** {ex}")

    # Build tabs
    tabs_to_show = []
    if group_col is not None and group_col in df.columns:
        tabs_to_show.append("Grouped By Variable")
    if topic_col is not None and topic_col in df.columns:
        tabs_to_show.append("By Topic")
    if sentiment_col is not None and sentiment_col in df.columns:
        tabs_to_show.append("By Sentiment")
    if theme_col is not None and theme_col in df.columns:
        tabs_to_show.append("By Theme")

    if not tabs_to_show:
        st.warning("No valid grouping columns found (group_col, topic_col, sentiment_col, theme_col).")
        return

    main_tabs = st.tabs(tabs_to_show)

    tab_index = 0
    if "Grouped By Variable" in tabs_to_show:
        with main_tabs[tab_index]:
            st.subheader(f"Example Responses By '{group_col}'")
            unique_vals = sorted([x for x in df[group_col].dropna().unique()])
            sub_tabs = st.tabs([str(x) for x in unique_vals])
            for i, val in enumerate(unique_vals):
                with sub_tabs[i]:
                    sub_ = df[df[group_col] == val]
                    st.write(f"**Category**: {val}, total: {len(sub_)}")
                    render_category_examples(sub_, val)
        tab_index += 1

    if "By Topic" in tabs_to_show:
        with main_tabs[tab_index]:
            st.subheader(f"Example Responses By '{topic_col}'")
            unique_vals = sorted([x for x in df[topic_col].dropna().unique()])
            sub_tabs = st.tabs([str(x) for x in unique_vals])
            for i, val in enumerate(unique_vals):
                with sub_tabs[i]:
                    sub_ = df[df[topic_col] == val]
                    st.write(f"**Topic**: {val}, total: {len(sub_)}")
                    render_category_examples(sub_, val)
        tab_index += 1

    if "By Sentiment" in tabs_to_show:
        with main_tabs[tab_index]:
            st.subheader(f"Example Responses By '{sentiment_col}'")
            unique_vals = sorted([x for x in df[sentiment_col].dropna().unique()])
            sub_tabs = st.tabs([str(x) for x in unique_vals])
            for i, val in enumerate(unique_vals):
                with sub_tabs[i]:
                    sub_ = df[df[sentiment_col] == val]
                    st.write(f"**Sentiment**: {val}, total: {len(sub_)}")
                    render_category_examples(sub_, val)
        tab_index += 1

    if "By Theme" in tabs_to_show:
        with main_tabs[tab_index]:
            st.subheader(f"Example Responses By '{theme_col}'")
            unique_vals = sorted([x for x in df[theme_col].dropna().unique()])
            sub_tabs = st.tabs([str(x) for x in unique_vals])
            for i, val in enumerate(unique_vals):
                with sub_tabs[i]:
                    sub_ = df[df[theme_col] == val]
                    st.write(f"**Theme**: {val}, total: {len(sub_)}")
                    render_category_examples(sub_, val)
        tab_index += 1

###############################################################################
# 10) SENTIMENT ANALYSES
###############################################################################
def analyze_group_sentiment(texts_by_group):
    """
    Analyzes sentiment for each group's texts using VADER
    and returns a dict like:
    {
      group_name: {
         'total': int,
         'positive': int,
         'negative': int,
         'neutral': int,
         'scores': [list_of_compound_scores],
         'avg_compound': float,
         'pos_pct': float,
         'neg_pct': float,
         'neu_pct': float
      },
      ...
    }
    """
    analyzer = SentimentIntensityAnalyzer()
    results = defaultdict(lambda: {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'scores': [],
        'avg_compound': 0.0,
        'pos_pct': 0.0,
        'neg_pct': 0.0,
        'neu_pct': 0.0
    })

    for group_name, texts in texts_by_group.items():
        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
        for txt in valid_texts:
            vs = analyzer.polarity_scores(txt)
            results[group_name]['total'] += 1
            results[group_name]['scores'].append(vs['compound'])

            if vs['compound'] >= 0.05:
                results[group_name]['positive'] += 1
            elif vs['compound'] <= -0.05:
                results[group_name]['negative'] += 1
            else:
                results[group_name]['neutral'] += 1

        # post-calc
        total_n = results[group_name]['total']
        if total_n > 0:
            comp_list = results[group_name]['scores']
            results[group_name]['avg_compound'] = sum(comp_list) / total_n
            results[group_name]['pos_pct'] = 100.0 * results[group_name]['positive'] / total_n
            results[group_name]['neg_pct'] = 100.0 * results[group_name]['negative'] / total_n
            results[group_name]['neu_pct'] = 100.0 * results[group_name]['neutral'] / total_n

    return dict(results)

def analyze_group_sentiment(texts_by_group):
    import logging
    from collections import defaultdict
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import pandas as pd

    # If the dict is empty (meaning no group_by selected, or no data),
    # fall back to a single group "All" with no texts. (At least it won't crash.)
    if not texts_by_group:
        texts_by_group = {"All": []}

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
        valid_texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
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
                logging.error(f"Error analyzing text: {str(e)}")

        # Calculate percentages/averages
        c = sentiment_stats[group]['total']
        if c > 0:
            sentiment_stats[group]['avg_compound'] = sum(sentiment_stats[group]['scores']) / c
            sentiment_stats[group]['pos_pct'] = (sentiment_stats[group]['positive'] / c) * 100
            sentiment_stats[group]['neg_pct'] = (sentiment_stats[group]['negative'] / c) * 100
            sentiment_stats[group]['neu_pct'] = (sentiment_stats[group]['neutral'] / c) * 100

    return dict(sentiment_stats)

def sentiment_radar_chart(sentiment_summary):
    """
    Build a simple Plotly radar chart from a list of dictionaries like:
    [
      {'Group': 'All', 'Positive%': 45, 'Negative%': 30, 'Neutral%': 25, 'AvgCompound': 0.15},
      ...
    ]
    """
    categories = ["Positive%", "Neutral%", "Negative%", "CompoundScaled"]

    fig = go.Figure()
    for row in sentiment_summary:
        grp = row["Group"]
        # Convert to float
        pos_val = float(row["Positive%"])
        neu_val = float(row["Neutral%"])
        neg_val = float(row["Negative%"])
        # Scale compound from -1..1 => 0..100
        comp_val = float(row["AvgCompound"])
        comp_scaled = (comp_val + 1.0) * 50.0  # e.g. -1 => 0, 1 => 100

        fig.add_trace(go.Scatterpolar(
            r=[pos_val, neu_val, neg_val, comp_scaled],
            theta=categories,
            fill='toself',
            name=str(grp)
        ))

    fig.update_layout(
        title="Sentiment Radar Chart",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        width=700, height=600
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

###############################################################################
# 11) THEMATIC EVOLUTION
###############################################################################
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
    """
    Create a heatmap showing theme intensity across groups with 'Viridis' colorscale.
    """
    groups = evolution_data['groups']
    themes = evolution_data['themes']
    values = evolution_data['values']

    if not groups or not themes:
        return None

    import numpy as np
    values_array = np.array(values)  # shape: (len(themes), len(groups))
    if values_array.size == 0 or values_array.max() == 0:
        return None

    # We'll normalize for nice color range
    values_normalized = values_array / values_array.max()

    fig = go.Figure(data=go.Heatmap(
        z=values_normalized,
        x=groups,
        y=themes,
        colorscale='Viridis',  # <--- Make sure it's Viridis
        hoverongaps=False
    ))

    fig.update_layout(
        title='Theme Intensity Across Groups',
        xaxis_title='Groups',
        yaxis_title='Themes',
        height=600
    )
    return fig

def create_theme_waterfall(evolution_data):
    """
    Create a single waterfall chart showing usage of the top theme across groups.
    Picks the theme with the highest total usage (sum across all groups).
    """
    groups = evolution_data['groups']  # e.g. ['G1','G2', ...]
    themes = evolution_data['themes']  # e.g. ['themeA','themeB',...]
    values = evolution_data['values']  # list of lists, each row i -> theme i's usage per group

    if not groups or not themes or not values:
        return None

    # Sum usage for each theme
    theme_sums = []
    for i, t in enumerate(themes):
        total_usage = sum(values[i])
        theme_sums.append((t, total_usage))

    # Pick top theme
    theme_sums.sort(key=lambda x: x[1], reverse=True)
    if not theme_sums or theme_sums[0][1] == 0:
        return None

    top_theme, _ = theme_sums[0]
    top_index = themes.index(top_theme)
    usage_arr = values[top_index]  # usage across each group

    measure_list = []
    x_list = []
    y_list = []
    for grp_idx, grp_name in enumerate(groups):
        measure_list.append('relative')
        x_list.append(grp_name)
        y_list.append(usage_arr[grp_idx])

    fig = go.Figure(go.Waterfall(
        name=top_theme,
        orientation="v",
        measure=measure_list,
        x=x_list,
        y=y_list,
        textposition="outside"
    ))
    fig.update_layout(
        title=f"Waterfall: usage changes for top theme '{top_theme}'",
        height=400
    )
    return fig

###############################################################################
# 12) APP LAYOUT
###############################################################################

st.markdown("""
<style>
body {
    background-color: #FBFBFB;
    color: #333;
}
.sidebar .sidebar-content {
    background-color: #FFFFFF;
}
.block-container {
    padding: 1.5rem;
}
h1, h2, h3, h4 {
    color: #2C3E50;
    font-family: "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}
.dataframe tbody tr:nth-child(even) {
    background-color: #F7F9FB;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Text Analysis Dashboard")

# ----------------------------------------------------------------
# SIDEBAR: FILE LOADER & ANALYSIS CHOICES
# ----------------------------------------------------------------
with st.sidebar:
    st.title("📊 Text Analysis Dashboard")
    st.header("File / Data Selection")

    file_up = st.file_uploader("Upload Excel File", type=['xlsx'])
    if file_up and not st.session_state.file_processed:
        with st.spinner("Loading data..."):
            qm, rd, ov, gc = load_excel_file(file_up, chosen_survey="All")
            if qm is not None and rd is not None:
                st.session_state.data = {
                    'qmap': qm,
                    'responses_dict': rd,
                    'open_var_options': ov,
                    'grouping_columns': gc
                }
                st.session_state.file_processed = True
                st.rerun()
            else:
                st.error("Invalid file or missing 'question_mapping' sheet. Please verify.")

    if st.session_state.file_processed:
        var_opts = st.session_state.data['open_var_options']
        grp_cols = st.session_state.data['grouping_columns']

        st.markdown("---")
        st.markdown("**Choose Analysis**")
        analyses = [
            "Open Coding",
            "Word Cloud",
            "Word Analysis",
            "Word Dive",
            "Topic Discovery",
            "Sentiment Analysis",
            "Theme Evolution"
        ]
        selected_index = 0
        if st.session_state.selected_analysis in analyses:
            selected_index = analyses.index(st.session_state.selected_analysis)

        st.session_state.selected_analysis = st.selectbox(
            "Analysis Section",
            options=analyses,
            index=selected_index
        )

        st.markdown("---")
        # Choose variable
        if var_opts:
            st.subheader("Variable to analyze")
            chosen_var = st.selectbox(
                "Select an open-ended variable",
                list(var_opts.keys()),
                format_func=lambda x: var_opts[x]
            )
        else:
            chosen_var = None

        st.markdown("---")
        # Group-by (optional)
        if grp_cols:
            st.subheader("Group By")
            chosen_grp = st.selectbox("Group responses by", [None, "None"] + grp_cols, index=1)
            if chosen_grp in [None, "None"]:
                chosen_grp = None
        else:
            chosen_grp = None

        st.markdown("---")
        # Stopwords manager
        render_stopwords_management()

        # Synonym manager
        render_synonym_groups_management()


        # Coding group manager
        st.markdown("---")
        render_group_manager()
        st.markdown("---")

        # Refresh button
        if st.button("🔄 Refresh All"):
            st.session_state.file_processed = False
            st.rerun()

    else:
        chosen_var = None
        chosen_grp = None


# ----------------------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------------------
if st.session_state.file_processed and chosen_var:
    qmap = st.session_state.data['qmap']
    rdict = st.session_state.data['responses_dict']
    open_var_options = st.session_state.data['open_var_options']
    grouping_columns = st.session_state.data['grouping_columns']

    # Gather responses for chosen variable
    var_resps = get_responses_for_variable(rdict, chosen_var, chosen_grp)
    if not var_resps:
        st.warning("No responses found for this variable. Please verify your selection.")
        st.stop()

    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; 
                background-color: #f8f9fa; margin: 10px 0;">
        <div style="color: #2E7D32; font-size: 1.2em; margin-bottom: 10px; font-weight: bold;">
            <strong>Primary Question</strong>
        </div>
        <div style="color: #1a1a1a; font-size: 1.1em; line-height: 1.5; font-weight: bold;">
            <strong>{open_var_options.get(chosen_var, "No question text")}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Determine which section of the dashboard to show
    if st.session_state.selected_analysis == "Open Coding":
        st.markdown("## Open Coding")
        render_open_coding_interface(chosen_var, rdict, open_var_options, grouping_columns)

    elif st.session_state.selected_analysis == "Word Cloud":
        st.markdown("## 🎨 Word Cloud")
        render_wordclouds(var_resps)

    elif st.session_state.selected_analysis == "Word Analysis":
        st.markdown("## 📊 Word Analysis")

        # Each group
        from sklearn.feature_extraction.text import CountVectorizer

        for key_, arr_ in var_resps.items():
            st.markdown(f"### {key_} ({len(arr_)} responses)")
            if not arr_:
                st.warning("No valid text found.")
                continue

            # Clean
            cleaned = [
                process_text(tx, st.session_state.custom_stopwords, st.session_state.synonym_groups)
                for tx in arr_ if isinstance(tx, str) and tx.strip()
            ]
            if not cleaned:
                st.warning("No valid text after cleaning.")
                continue

            # Word Frequencies
            st.markdown("#### Word Frequencies")
            freq_vec = CountVectorizer(max_features=20, stop_words=list(st.session_state.custom_stopwords))
            X_ = freq_vec.fit_transform(cleaned)
            w_ = freq_vec.get_feature_names_out()
            fr_ = X_.sum(axis=0).A1
            df_freq = pd.DataFrame({'Word': w_, 'Frequency': fr_}).sort_values("Frequency", ascending=False)
            fig_f = px.bar(df_freq, x='Word', y='Frequency', title=f"Top Words - {key_}")
            st.plotly_chart(fig_f, use_container_width=True)

            # Co-occurrence
            st.markdown("#### Co-occurrence Network")
            min_edge_ = st.slider("Minimum edge weight", 1, 10, 2, key=f"{key_}_minedge")
            maxw__ = st.slider("Max # of words", 10, 100, 30, key=f"{key_}_maxwords")

            net_vec = CountVectorizer(max_features=maxw__, stop_words=list(st.session_state.custom_stopwords))
            XX_ = net_vec.fit_transform(cleaned)
            wnames_ = net_vec.get_feature_names_out()
            mat_ = (XX_.T @ XX_).toarray()
            np.fill_diagonal(mat_, 0)

            G_ = nx.Graph()
            freq_ar = XX_.sum(axis=0).A1
            for i, wnm in enumerate(wnames_):
                G_.add_node(wnm, frequency=int(freq_ar[i]))

            for i, j in combinations(range(len(wnames_)), 2):
                w_ = mat_[i, j]
                if w_ >= min_edge_:
                    G_.add_edge(wnames_[i], wnames_[j], weight=w_)

            pos_ = nx.spring_layout(G_, k=1 / np.sqrt(len(G_.nodes)), iterations=50)
            edge_x, edge_y = [], []
            for (na, nb, dic_) in G_.edges(data=True):
                x0, y0 = pos_[na]
                x1, y1 = pos_[nb]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines', opacity=0.5
            )

            node_x, node_y, node_text, node_size = [], [], [], []
            for node__ in G_.nodes():
                xx, yy = pos_[node__]
                node_x.append(xx)
                node_y.append(yy)
                fr_val = G_.nodes[node__]['frequency']
                node_text.append(f"{node__} (freq={fr_val})")
                node_size.append(np.sqrt(fr_val) * 10 + 5)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=list(G_.nodes()),
                textposition='top center',
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    colorscale='YlOrRd',
                    color=[G_.nodes()[n]['frequency'] for n in G_.nodes()],
                    size=node_size,
                    line_width=2
                )
            )
            net_fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title='Word Co-occurrence Network',
                                    showlegend=False,
                                    hovermode='closest',
                                    width=800, height=800,
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                ))
            st.plotly_chart(net_fig, use_container_width=True)

            st.markdown("#### Top Co-occurring Pairs")
            pairs_ = []
            for i in range(len(wnames_)):
                for j in range(i + 1, len(wnames_)):
                    co_val = mat_[i, j]
                    if co_val > 0:
                        pairs_.append({'Word1': wnames_[i], 'Word2': wnames_[j], 'Co-occurrences': int(co_val)})
            if pairs_:
                pair_df = pd.DataFrame(pairs_).sort_values("Co-occurrences", ascending=False).head(15)
                st.dataframe(pair_df, use_container_width=True)

    elif st.session_state.selected_analysis == "Word Dive":
        render_word_dive(chosen_var, rdict, open_var_options, grouping_columns)

    elif st.session_state.selected_analysis == "Topic Discovery":

        st.markdown("## 🔍 Topic Discovery")

        # ------------------------------------------------------------
        # FIRST: 3D Topic Visualization
        # ------------------------------------------------------------

        st.subheader("3D Topic Visualization")

        # Combine all responses for the chosen variable into a single DataFrame
        combined_rows = []
        for sid, dfx in rdict.items():
            if chosen_var in dfx.columns:
                sub_df = dfx[[chosen_var]].copy()
                sub_df["surveyid"] = sid

                # If user has columns like id, age, etc., include them if they exist:
                for extra_col in ["id", "age", "gender", "region", "jobtitle"]:
                    if extra_col in dfx.columns:
                        sub_df[extra_col] = dfx[extra_col]

                # Filter out NaNs
                sub_df.dropna(subset=[chosen_var], inplace=True)
                sub_df = sub_df[sub_df[chosen_var].astype(str).str.strip() != ""]

                combined_rows.append(sub_df)

        if combined_rows:
            big_df_3d = pd.concat(combined_rows, ignore_index=True)
            big_df_3d.rename(columns={chosen_var: "open_response"}, inplace=True)

            fig_3d = create_3d_topic_visualization(
                df=big_df_3d,
                text_col="open_response",
                jobtitle_col="jobtitle",
                age_col="age",
                gender_col="gender",
                region_col="region",
                model_name='all-MiniLM-L6-v2',
                n_components=3,
                n_clusters=5,
                random_state=42
            )

            if fig_3d is not None:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Not enough valid text data for 3D topic visualization.")
        else:
            st.warning("No valid responses to display in 3D topic visualization.")

        # ------------------------------------------------------------
        # SECOND: Simple K-Means (or similar) Clustering Overview + Representative Docs
        # ------------------------------------------------------------

        st.subheader("Topic Clusters Overview")
        num_topics = st.slider("Number of Topics (K)", 2, 10, 4)
        min_topic_size = st.slider("Min Topic Size (unused for basic KMeans)", 2, 5, 2)

        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        for group_name, arr_ in var_resps.items():
            st.markdown(f"### {group_name} ({len(arr_)} responses)")
            cleaned_txts = [tx for tx in arr_ if isinstance(tx, str) and tx.strip()]
            if len(cleaned_txts) < 20:
                st.warning("Not enough data (<20) to do topic modeling.")
                continue

            # 1) Embed all texts
            embeddings_ = emb_model.encode(cleaned_txts, show_progress_bar=False)

            # 2) K-Means
            kmeans_ = KMeans(n_clusters=num_topics, random_state=42).fit(embeddings_)

            from collections import defaultdict
            cluster_map = defaultdict(list)

            # 3) Collect cluster -> list of doc indices (and doc texts)
            for idx, lbl_ in enumerate(kmeans_.labels_):
                cluster_map[lbl_].append(idx)

            # 4) Build table data
            stats_list = []
            cvec = CountVectorizer(max_features=10, stop_words=list(st.session_state.custom_stopwords))

            for c_ in sorted(cluster_map.keys()):
                indices_in_cluster = cluster_map[c_]
                if not indices_in_cluster:
                    continue

                # Gather the documents in this cluster.
                cluster_docs = [cleaned_txts[i] for i in indices_in_cluster if isinstance(cleaned_txts[i], str)]
                cluster_docs = [doc.strip() for doc in cluster_docs if doc.strip()]

                if not cluster_docs:
                    st.warning(f"No valid documents in cluster {c_}. Skipping.")
                    continue

                # Safely call cvec.fit_transform to avoid "empty vocabulary" errors.
                try:
                    X_ = cvec.fit_transform(cluster_docs)
                except ValueError as ve:
                    st.warning(f"Skipping cluster {c_} due to error: {ve}")
                    continue

                # A) Word frequencies for these docs
                cluster_docs = [cleaned_txts[i] for i in indices_in_cluster]
                X_ = cvec.fit_transform(cluster_docs)
                w_ = cvec.get_feature_names_out()
                fr_ = X_.sum(axis=0).A1
                top_ = sorted(zip(w_, fr_), key=lambda x: x[1], reverse=True)[:5]
                top_terms = ", ".join([t[0] for t in top_])

                # B) Identify top 3 representative docs (closest to centroid)
                center = kmeans_.cluster_centers_[c_]
                distances = []
                for doc_idx in indices_in_cluster:
                    dist = np.linalg.norm(embeddings_[doc_idx] - center)
                    distances.append((dist, doc_idx))
                distances.sort(key=lambda x: x[0])
                top_3_docs = [cleaned_txts[d[1]] for d in distances[:3]]
                rep_docs_str = "\n\n".join(top_3_docs)

                stats_list.append({
                    'Topic': c_,
                    'Count': len(cluster_docs),
                    'Top Terms': top_terms,
                    'Representative Docs': rep_docs_str
                })

            # 5) Show final DataFrame with top terms + representative docs
            if stats_list:
                df_stats = pd.DataFrame(stats_list)
                st.dataframe(df_stats, use_container_width=True)
            else:
                st.info(f"No clusters found for {group_name} or data was insufficient.")


    elif st.session_state.selected_analysis == "Sentiment Analysis":

        st.markdown("## ❤️ Sentiment Analysis")

        # If no group is chosen, var_resps will have a single key "All" with all texts.
        # If a group is chosen, var_resps will have separate keys for each group.

        from collections import defaultdict
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        # Initialize analyzer
        analyzer = SentimentIntensityAnalyzer()

        # We store sentiment counts & scores per group
        results_dict = defaultdict(lambda: {
            'count': 0,
            'pos': 0,
            'neg': 0,
            'neu': 0,
            'scores': []
        })

        # Go through each group key in var_resps (could be just 'All' if no group is selected)
        for group_key, texts_list in var_resps.items():
            for txt in texts_list:
                if not isinstance(txt, str) or not txt.strip():
                    continue
                sc = analyzer.polarity_scores(txt)
                results_dict[group_key]['count'] += 1
                results_dict[group_key]['scores'].append(sc['compound'])

                if sc['compound'] >= 0.05:
                    results_dict[group_key]['pos'] += 1
                elif sc['compound'] <= -0.05:
                    results_dict[group_key]['neg'] += 1
                else:
                    results_dict[group_key]['neu'] += 1

        # Build summary rows for table and chart
        sum_rows = []
        for grp_key, stats_ in results_dict.items():
            c_ = stats_['count']
            if c_ > 0:
                avg_comp = sum(stats_['scores']) / c_
                pos_pct = (stats_['pos'] / c_) * 100
                neg_pct = (stats_['neg'] / c_) * 100
                neu_pct = (stats_['neu'] / c_) * 100

                sum_rows.append({
                    'Group': grp_key,
                    'Total': c_,
                    'Positive%': f"{pos_pct:.1f}",
                    'Neutral%': f"{neu_pct:.1f}",
                    'Negative%': f"{neg_pct:.1f}",
                    'AvgCompound': f"{avg_comp:.3f}"
                })

        if not sum_rows:
            st.warning("No sentiment data found for this variable.")
        else:
            # Display summary table
            df_summary = pd.DataFrame(sum_rows)
            st.dataframe(df_summary, use_container_width=True)

            # ===== 1) Build a Radar Chart (Positive%, Neutral%, Negative%, CompoundScaled) =====
            # We'll scale compound from -1..1 => 0..100 so it fits on the same radar as the percentages
            categories = ["Positive%", "Neutral%", "Negative%", "CompoundScaled"]

            radar_fig = go.Figure()
            for row in sum_rows:
                grp = row["Group"]
                pos_val = float(row["Positive%"])
                neu_val = float(row["Neutral%"])
                neg_val = float(row["Negative%"])
                comp_val = float(row["AvgCompound"])
                comp_scaled = (comp_val + 1.0) * 50.0  # e.g., -1 => 0, 1 => 100

                radar_fig.add_trace(go.Scatterpolar(
                    r=[pos_val, neu_val, neg_val, comp_scaled],
                    theta=categories,
                    fill='toself',
                    name=str(grp)
                ))

            radar_fig.update_layout(
                title="Sentiment Radar Chart",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                width=700,
                height=600
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            # ===== 2) Violin Plot of Compound Distributions =====
            # Flatten out each group's compound scores
            all_points = []
            for grp_key, stats_ in results_dict.items():
                for cval in stats_['scores']:
                    all_points.append({"Group": grp_key, "Compound": cval})

            if all_points:
                df_vio = pd.DataFrame(all_points)
                fig_violin = px.violin(
                    df_vio,
                    x='Group',
                    y='Compound',
                    box=True,
                    points='all',
                    title='Sentiment Compound Score Distribution'
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                st.write("No data to display in violin plot.")

    elif st.session_state.selected_analysis == "Theme Evolution":

        st.markdown("## 🌊 Theme Evolution")
        st.info("Explore how themes evolve across different groups/time. Adjust parameters below.")

        # Build texts_by_group
        group_texts = defaultdict(list)
        for key_, arr_ in var_resps.items():

            label_ = key_

            if chosen_grp:

                parts = key_.split('_', 1)

                if len(parts) == 2:
                    label_ = parts[1]

            group_texts[label_].extend(arr_)
        if group_texts:

            n_th = st.slider("Number of Themes (bigrams/trigrams)", 3, 10, 5)

            min_fr = st.slider("Minimum frequency for each bigram/trigram", 2, 10, 3)

            evo_data = calculate_theme_evolution(dict(group_texts), num_themes=n_th, min_freq=min_fr)

            colA, colB = st.columns(2)

            with colA:

                sankey_fig = create_theme_flow_diagram(evo_data)

                if sankey_fig:

                    st.plotly_chart(sankey_fig, use_container_width=True)

                else:

                    st.warning("Not enough data for the Theme Flow Diagram.")

            with colB:

                heatmap_fig = create_theme_heatmap(evo_data)

                if heatmap_fig:

                    st.plotly_chart(heatmap_fig, use_container_width=True)

                else:

                    st.warning("Not enough data for the Theme Heatmap.")

            # NOW add the waterfall below (or in a new column if you prefer):

            waterfall_fig = create_theme_waterfall(evo_data)

            if waterfall_fig:

                st.plotly_chart(waterfall_fig, use_container_width=True)

            else:

                st.info("No suitable data for a waterfall chart.")
        else:
            st.warning("No data found for Theme Evolution.")

