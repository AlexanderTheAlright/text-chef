################################################################################
# STREAMLIT TEXT ANALYSIS APP
################################################################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import networkx as nx
import os
import time
import re
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
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

matplotlib.use('Agg')  # For headless environments

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

def auto_save_check():
    """Auto-save every 5 min."""
    if (time.time() - st.session_state.last_save_time) > 300:
        save_coding_state()

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
    """Render the stopwords manager in sidebar."""
    st.markdown("#### Stopwords Management")

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
    import csv
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
    """Load the question_mapping sheet + either chosen_survey or all."""
    try:
        xls = pd.ExcelFile(excel)
        sheets = xls.sheet_names
        if 'question_mapping' not in sheets:
            return None, None, None, None

        qmap = pd.read_excel(xls, 'question_mapping')
        if not all(c in qmap.columns for c in ['variable','question','surveyid']):
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
            'dk','dk.','d/k','d.k.','dont know',"don't know","na","n/a","n.a.","n/a.",
            'not applicable','none','nil','no response','no answer','.','-','x','refused','ref',
            'dk/ref','nan','NaN','NAN','_dk_','_na_','___dk___','___na___','__dk__','__na__',
            '_____dk_____','_____na_____',''
        }

        for sheet in sheets_to_load:
            df = pd.read_excel(xls, sheet_name=sheet,
                               na_values=['','NA','nan','NaN','null','none','#N/A','N/A'])
            base_cols = {c.split('.')[0] for c in df.columns}
            all_cols.update(base_cols)

            open_candidates = {c for c in base_cols if c.endswith('_open')}
            open_vars_set.update(open_candidates)

            for col in df.columns:
                if col.split('.')[0] in open_candidates:
                    def clean_val(x):
                        if pd.isna(x):
                            return pd.NA
                        x = str(x).lower().strip()
                        if x in invalid_resps:
                            return pd.NA
                        return x
                    df[col] = df[col].apply(clean_val)
            responses_dict[sheet] = df

        # grouping columns
        grouping_cols = sorted(c for c in all_cols if not c.endswith('_open') and not c.endswith('.1'))

        # open var options
        open_var_opts = {}
        for v in sorted(open_vars_set):
            row_for_v = qmap[qmap['variable'] == v]
            if not row_for_v.empty:
                open_var_opts[v] = f"{v} - {row_for_v.iloc[0]['question']}"
            else:
                open_var_opts[v] = v

        return qmap, responses_dict, open_var_opts, grouping_cols

    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None


def get_responses_for_variable(dfs_dict, var, group_by=None):
    """Fetch responses for a chosen var across all sheets, optionally grouped."""
    import re
    from collections import defaultdict

    out = {}
    pattern = f"^{re.escape(var)}(?:\\.1)?$"

    for sid, df in dfs_dict.items():
        # columns that match var or var.1
        matching_cols = [c for c in df.columns if re.match(pattern, c, re.IGNORECASE)]
        if not matching_cols:
            continue

        if group_by and group_by in df.columns:
            grouped_responses = defaultdict(list)
            for col in matching_cols:
                sub_df = df[[col, group_by]].dropna(subset=[col])
                for gval, cdf in sub_df.groupby(group_by):
                    # filter empties
                    texts = [r for r in cdf[col].astype(str).tolist() if r.strip().lower() != 'nan']
                    if texts:
                        grouped_responses[str(gval)].extend(texts)
            for gval, arr in grouped_responses.items():
                if arr:
                    out[f"{sid}_{gval}"] = arr
        else:
            all_texts = []
            for col in matching_cols:
                colvals = [r for r in df[col].dropna().astype(str).tolist() if r.strip().lower() != 'nan']
                all_texts.extend(colvals)
            if all_texts:
                # remove duplicates
                seen = set()
                uniq = []
                for t in all_texts:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                out[sid] = uniq

    # Sort by descending # of responses
    out = dict(sorted(out.items(), key=lambda x: len(x[1]), reverse=True))
    return out


###############################################################################
# 5) TEXT PROCESSING
###############################################################################

def process_text(text, stopwords=None, synonym_groups=None):
    """Basic cleaning + optional synonyms. Already used in wordcloud functions."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    txt = str(text).lower().strip()

    txt = re.sub(r'<[^>]+>', '', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()

    words = txt.split()

    # stopwords
    if stopwords:
        words = [w for w in words if w not in stopwords]

    # synonyms
    if synonym_groups:
        replaced_words = []
        for w in words:
            replaced = False
            for gname, synset in synonym_groups.items():
                if w in synset:
                    replaced_words.append(gname)
                    replaced = True
                    break
            if not replaced:
                replaced_words.append(w)
        words = replaced_words

    return ' '.join(words)


###############################################################################
# 6) OPEN CODING (Groups & Assignments) + SAMPLES
###############################################################################

def load_open_coding_groups(file='cached_groups.csv'):
    """Load group definitions from CSV -> session_state."""
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            return df.to_dict('records')
        except:
            pass
    return []

def load_open_coding_assignments(file='cached_assignments.csv'):
    """Load text->group assignment from CSV -> session_state."""
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            return dict(zip(df.text, df.group))
        except:
            pass
    return {}

def save_coding_state():
    """Save open_coding_groups & open_coding_assignments to disk + backups."""
    try:
        if st.session_state.open_coding_groups:
            df = pd.DataFrame(st.session_state.open_coding_groups)
            df.to_csv('cached_groups.csv', index=False)

        if st.session_state.open_coding_assignments:
            df2 = pd.DataFrame(
                [{'text': k, 'group': v} for k, v in st.session_state.open_coding_assignments.items()]
            )
            df2.to_csv('cached_assignments.csv', index=False)

        # backups
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('coding_backups', exist_ok=True)
        if st.session_state.open_coding_groups:
            df.to_csv(f"coding_backups/groups_{ts}.csv", index=False)
        if st.session_state.open_coding_assignments:
            df2.to_csv(f"coding_backups/assignments_{ts}.csv", index=False)

        st.session_state.last_save_time = time.time()
        return True
    except Exception as e:
        print(f"Error saving coding state: {e}")
        return False

def initialize_coding_state():
    """Ensure open coding data is loaded from disk once."""
    if 'coding_initialized' not in st.session_state:
        # load groups
        st.session_state.open_coding_groups = load_open_coding_groups()
        # load assignments
        st.session_state.open_coding_assignments = load_open_coding_assignments()
        st.session_state.coding_initialized = True


def render_open_coding_interface(variable,
                                responses_dict,
                                open_var_options,
                                grouping_columns,
                                group_by=None):
    """
    Renders a combined "Open Coding" tab that includes:
      - managing groups
      - a coded table (Data Editor)
      - plus random samples to read/categorize
    """
    initialize_coding_state()
    auto_save_check()

    st.markdown("## 🔎 Open Coding & Samples")

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

    # Show question text
    ov_dict = open_var_options
    st.markdown(f"""
    <div class="question-box">
        <div class="question-label">Primary Question</div>
        <div class="question-text">{ov_dict.get(variable, "No question text")}</div>
    </div>
    """, unsafe_allow_html=True)

    # MANAGE GROUPS
    st.markdown("### Manage Groups")
    with st.expander("Create/Edit Groups", expanded=False):
        c1, c2 = st.columns([3,1])
        with c1:
            new_group_name = st.text_input("Group Name:")
            new_group_desc = st.text_input("Group Description (optional):")
            if st.button("Add / Update Group"):
                gname = new_group_name.strip()
                if gname:
                    existing = next((g for g in st.session_state.open_coding_groups if g['name'] == gname), None)
                    if existing:
                        existing["desc"] = new_group_desc.strip()
                        st.success(f"Updated group '{gname}'.")
                    else:
                        st.session_state.open_coding_groups.append({
                            "name": gname,
                            "desc": new_group_desc.strip()
                        })
                        st.success(f"Created new group '{gname}'.")
                    save_coding_state()
                else:
                    st.warning("Please enter a valid group name.")
        with c2:
            if st.button("💾 Save Group Changes"):
                ok = save_coding_state()
                if ok:
                    st.success("Groups saved.")
                else:
                    st.error("Error saving groups.")

        # Remove group
        del_choice = st.selectbox("Remove a Group?", ["(None)"] + [g["name"] for g in st.session_state.open_coding_groups])
        if del_choice != "(None)":
            if st.button(f"Remove Group '{del_choice}'"):
                st.session_state.open_coding_groups = [
                    g for g in st.session_state.open_coding_groups if g['name'] != del_choice
                ]
                # remove from assignments
                for txt_key, assigned_grp in list(st.session_state.open_coding_assignments.items()):
                    if assigned_grp == del_choice:
                        st.session_state.open_coding_assignments[txt_key] = "Unassigned"
                save_coding_state()
                st.success(f"Group '{del_choice}' removed.")

    # COLLECT RELEVANT DATA
    all_dfs = []
    for sid, df in responses_dict.items():
        if variable in df.columns:
            tmp = df.copy()
            tmp['surveyid'] = sid
            if 'province' in tmp.columns:
                tmp['region'] = tmp['province']
                tmp.drop('province', axis=1, inplace=True)
            elif 'state' in tmp.columns:
                tmp['region'] = tmp['state']
                tmp.drop('state', axis=1, inplace=True)
            all_dfs.append(tmp)
    if not all_dfs:
        st.warning("No valid data found for this variable.")
        return

    cdf = pd.concat(all_dfs, ignore_index=True)

    # Identify default columns
    default_cols = ['id','age','jobtitle']
    if 'region' in cdf.columns:
        default_cols.append('region')
    def_cols_present = [c for c in default_cols if c in cdf.columns]

    # Also see if variable has a base var
    base_var = variable.replace('_open','')

    # Let user pick additional columns
    grouping_cols = [g for g in grouping_columns if g not in def_cols_present]
    st.markdown("### Data Selection")
    user_cols = st.multiselect(
        "Pick additional columns to display:",
        options=grouping_cols,
        default=[]
    )

    used_cols = def_cols_present[:]
    used_cols.extend(user_cols)
    if base_var in cdf.columns and base_var not in used_cols:
        used_cols.append(base_var)
    used_cols.append(variable)
    used_cols.append('surveyid')
    used_cols = list(dict.fromkeys(used_cols))  # remove duplicates if any

    # Ensure 'surveyid' is right after 'id' if both exist
    if 'id' in used_cols and 'surveyid' in used_cols:
        used_cols.remove('surveyid')
        idx_ = used_cols.index('id')
        used_cols.insert(idx_+1, 'surveyid')

    # Build the final table
    show_table_expander = st.expander("🔎 View & Edit Full Table", expanded=False)
    with show_table_expander:
        temp_data = cdf[used_cols].copy()
        temp_data = temp_data.dropna(subset=[variable])
        temp_data = temp_data[temp_data[variable].astype(str).str.strip() != ""]

        # Filter
        st.markdown("**Global Search/Filter**")
        gl_search = st.text_input("Search across all columns:")
        show_col_filters = st.checkbox("Show column-based filters", False)
        col_filters = {}
        if show_col_filters:
            st.write("Enter text to filter each column:")
            columns_per_row = 3
            fil_cols = st.columns(columns_per_row)
            for i, c in enumerate(used_cols):
                with fil_cols[i % columns_per_row]:
                    col_filters[c] = st.text_input(f"Filter {c}", "")

        # apply
        df_for_edit = temp_data.copy()

        if gl_search.strip():
            # do a global OR search across columns
            mask = pd.Series(False, index=df_for_edit.index)
            for c in df_for_edit.columns:
                mask |= df_for_edit[c].astype(str).str.contains(gl_search, case=False, na=False)
            df_for_edit = df_for_edit[mask]

        if show_col_filters:
            for c, val in col_filters.items():
                if val.strip():
                    df_for_edit = df_for_edit[df_for_edit[c].astype(str).str.contains(val, case=False, na=False)]

        # Build a data dict for st.data_editor
        rows_data = []
        for idx, row in df_for_edit.iterrows():
            resp_text = row[variable]
            assigned_grp = st.session_state.open_coding_assignments.get(resp_text, "Unassigned")
            grp_desc = ""
            if assigned_grp != "Unassigned":
                gobj = next((g for g in st.session_state.open_coding_groups if g['name'] == assigned_grp), None)
                if gobj:
                    grp_desc = gobj.get("desc","")

            new_row = {c: row[c] for c in df_for_edit.columns}
            new_row['coded_group'] = assigned_grp
            new_row['group_desc'] = grp_desc
            rows_data.append(new_row)

        final_df = pd.DataFrame(rows_data)

        # Build config for columns
        colconf = {}
        colconf['coded_group'] = st.column_config.SelectboxColumn(
            "Coded Group",
            options=["Unassigned"] + [g["name"] for g in st.session_state.open_coding_groups],
            help="Pick group from dropdown"
        )
        colconf['group_desc'] = st.column_config.TextColumn(
            "Group Description",
            disabled=True
        )
        for c in final_df.columns:
            if c not in ['coded_group','group_desc']:
                colconf[c] = st.column_config.TextColumn(c, disabled=(c != variable))

        edited = st.data_editor(
            final_df,
            column_config=colconf,
            hide_index=True,
            use_container_width=True,
            key="open_coding_data_editor"
        )

        # sync changes back
        for i in edited.index:
            rtxt = edited.loc[i, variable]
            cgrp = edited.loc[i, 'coded_group']
            st.session_state.open_coding_assignments[rtxt] = cgrp

        st.markdown("#### Export")
        c1, c2 = st.columns(2)
        with c1:
            # current view
            cur_csv = edited.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Current View CSV",
                data=cur_csv,
                file_name=f"coded_data_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )
        with c2:
            # entire dataset with coding
            all_rows = []
            for idx0, row0 in temp_data.iterrows():
                rt0 = row0[variable]
                assigned_grp0 = st.session_state.open_coding_assignments.get(rt0, "Unassigned")
                grp_desc0 = ""
                if assigned_grp0 != "Unassigned":
                    gobj0 = next((g for g in st.session_state.open_coding_groups if g["name"] == assigned_grp0), None)
                    if gobj0: grp_desc0 = gobj0.get("desc","")

                base_ = {c: row0[c] for c in temp_data.columns}
                base_["coded_group"] = assigned_grp0
                base_["group_desc"] = grp_desc0
                all_rows.append(base_)
            full_df = pd.DataFrame(all_rows)
            full_csv = full_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Complete Dataset",
                data=full_csv,
                file_name=f"complete_coded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )

    # RENDER SAMPLES
    st.markdown("---")
    st.markdown("### Random Samples for Review")

    def build_samples_dict():
        out = {"All":[]}
        for df in responses_dict.values():
            if variable in df.columns:
                for _, row in df.iterrows():
                    val = row[variable]
                    if pd.notna(val) and str(val).strip():
                        out["All"].append({
                            "text": str(val).strip(),
                            "id": row["id"] if "id" in row else None,
                            "age": row["age"] if "age" in row else None,
                            "jobtitle": row["jobtitle"] if "jobtitle" in row else None,
                            "province": row["region"] if "region" in row else None
                        })
        return out

    samples_dict = build_samples_dict()

    # If user selected a valid group_by, let's also gather samples that way
    if group_by and group_by not in [None,'None']:
        for sid, df in responses_dict.items():
            if group_by in df.columns and variable in df.columns:
                for gval in df[group_by].dropna().unique():
                    cat_str = str(gval)
                    sub_ = df[(df[group_by]==gval) & df[variable].notna()]
                    for _, row in sub_.iterrows():
                        vt = str(row[variable]).strip()
                        if vt:
                            if cat_str not in samples_dict:
                                samples_dict[cat_str] = []
                            samples_dict[cat_str].append({
                                "text": vt,
                                "id": row["id"] if "id" in row else None,
                                "age": row["age"] if "age" in row else None,
                                "jobtitle": row["jobtitle"] if "jobtitle" in row else None,
                                "province": row["region"] if "region" in row else None
                            })

    # De-duplicate
    for cat_k, items in samples_dict.items():
        seen_txt = set()
        newlist = []
        for obj in items:
            if obj["text"] not in seen_txt:
                seen_txt.add(obj["text"])
                newlist.append(obj)
        samples_dict[cat_k] = newlist

    num_samp = st.slider("Number of samples per group", min_value=1, max_value=20, value=5)
    if st.button("🔀 Shuffle Samples"):
        st.session_state["sample_seed"] = int(time.time())
    import random
    random.seed(st.session_state.get("sample_seed", 1234))

    for cat, arr in samples_dict.items():
        st.markdown(f"#### {cat} (Total: {len(arr)})")
        if not arr:
            st.write("No data in this category.")
            continue
        # random sample
        sub_samp = random.sample(arr, min(num_samp, len(arr)))
        for i, obj in enumerate(sub_samp, 1):
            with st.expander(f"[{cat}] Sample #{i}", expanded=False):
                meta_parts = []
                if obj["id"]: meta_parts.append(f"ID: {obj['id']}")
                if obj["age"]: meta_parts.append(f"Age: {obj['age']}")
                if obj["jobtitle"]: meta_parts.append(f"Job: {obj['jobtitle']}")
                if obj["province"]: meta_parts.append(f"Region: {obj['province']}")
                if meta_parts:
                    st.markdown("*" + " | ".join(meta_parts) + "*")

                st.write(obj["text"])
                # Let user assign to a group
                assigned_grp_ = st.session_state.open_coding_assignments.get(obj["text"], "Unassigned")
                grp_list = ["Unassigned"] + [g["name"] for g in st.session_state.open_coding_groups]
                new_sel = st.selectbox(
                    "Assign to group:",
                    options=grp_list,
                    index=(grp_list.index(assigned_grp_) if assigned_grp_ in grp_list else 0),
                    key=f"sample_{cat}_{i}"
                )
                st.session_state.open_coding_assignments[obj["text"]] = new_sel

    if st.button("💾 Save All Coding"):
        ok = save_coding_state()
        if ok:
            st.success("Coding saved successfully.")
        else:
            st.error("Error saving coding.")

################################################################################
# WORDCLOUD FUNCTIONS
################################################################################

def generate_word_freq(texts, exact_words=200):
    """
    Process a list of strings, return the top N (exact_words) list of (word, freq).
    If fewer than exact_words are available, return all of them.
    """
    cleaned_texts = []
    for txt in texts:
        if isinstance(txt, str) and txt.strip():
            proc = process_text(txt, st.session_state.custom_stopwords, st.session_state.synonym_groups)
            if proc:
                cleaned_texts.append(proc)
    if not cleaned_texts:
        return []

    merged = ' '.join(cleaned_texts)
    freq_counter = Counter(merged.split())
    most_common = freq_counter.most_common(exact_words)
    return most_common


def generate_interactive_wordcloud(freq_data, highlight_words=None, title='', exact_words=200, colormap='viridis'):
    """
    Returns a Plotly figure (scatter-based wordcloud).
    - highlight_words (set): highlight in a viridis-like color (#443983), else grey
    - exact_words: top words to consider (already truncated in freq_data).
    - colormap is only used if highlight_words is empty; otherwise highlight logic is used.
    """
    if not freq_data:
        return None

    # Ensure freq_data has at most 'exact_words'
    freq_data = freq_data[:exact_words]

    words = [w for w, _ in freq_data]
    freqs = [f for _, f in freq_data]
    max_f = max(freqs) if freqs else 1
    sizes = [20 + (f / max_f) * 60 for f in freqs]

    import numpy as np
    golden_ratio = (1 + 5**0.5) / 2
    positions = []
    for i, f_val in enumerate(freqs):
        # Spiral-ish layout
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
                d_sq = (xi - xx)**2 + (yi - yy)**2
                dists.append((d_sq, j))
        dists.sort(key=lambda x: x[0])
        neigh = [words[d[1]] for d in dists[:k]]
        return ", ".join(neigh)

    custom_data = []
    color_list = []

    # If highlight set is provided, highlight in viridis-like color (#443983), else grey
    if highlight_words:
        for i, w in enumerate(words):
            highlight = (w.lower() in highlight_words)
            color_list.append("#443983" if highlight else "gray")
            custom_data.append((freqs[i], w, nearest_terms(i, 3)))
    else:
        # Use chosen colormap via matplotlib, fix the randint usage:
        import matplotlib as mpl
        selected_cmap = mpl.cm.get_cmap(colormap)
        import random
        random_state = np.random.RandomState(42)

        for i, w in enumerate(words):
            color_idx = random_state.randint(0, 256)  # <-- FIX: specify (low, high)
            r, g, b, _ = selected_cmap(color_idx / 255.0)
            color_list.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
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


def render_wordclouds(var_resps):
    """
    Updated Word Cloud UI with separate quadrant boxes:
      - 1: Select how many quadrants (1..4)
      - 2: For each quadrant, pick one or more categories
      - 3: EXACT number of words
      - 4: Choose color scheme for non-highlight
      - 5: Highlight words
      - Download CSVs for each quadrant and a combined CSV for all quadrants
    """
    st.markdown("## 🎨 Word Clouds")

    # 1) Quadrant layout
    layout_choice = st.selectbox("Number of quadrants", [1, 2, 3, 4], index=0)

    # 2) For each quadrant, let user pick categories from var_resps
    groups_available = sorted(list(var_resps.keys()))
    quadrant_selections = {}
    for q_idx in range(layout_choice):
        quadrant_label = f"Quadrant {chr(65 + q_idx)}"  # e.g. Quadrant A, B...
        quadrant_selections[quadrant_label] = st.multiselect(
            f"Select categories for {quadrant_label}",
            groups_available,
            []
        )

    # 3) EXACT number of words
    exact_words = st.slider("Exact number of words per quadrant", min_value=10, max_value=300, value=50, step=10)

    # 4) Color scheme for non-highlight
    color_schemes = ["viridis", "plasma", "inferno", "magma", "cividis", "winter", "coolwarm", "bone", "terrain", "twilight"]
    chosen_cmap = st.selectbox("Color Scheme (if no highlights)", color_schemes, index=0)

    # 5) Highlight words
    highlight_input = st.text_area("Highlight words (one per line)", "")
    highlight_set = {h.strip().lower() for h in highlight_input.split('\n') if h.strip()}

    # Gather text for each quadrant
    quadrant_texts = {}
    for quad_label, cat_list in quadrant_selections.items():
        combined_texts = []
        for cat in cat_list:
            combined_texts.extend(var_resps[cat])
        quadrant_texts[quad_label] = combined_texts

    # If everything is empty, warn and return
    nonempty_quadrants = any(len(txts) > 0 for txts in quadrant_texts.values())
    if not nonempty_quadrants:
        st.warning("No groups selected or no data to show.")
        return

    # Build freq data for each quadrant
    quadrant_freqs = {}
    for quad_label, txt_list in quadrant_texts.items():
        freq_ = generate_word_freq(txt_list, exact_words=exact_words)
        quadrant_freqs[quad_label] = freq_

    # TABS
    tabs = st.tabs(["📸 Static", "🔄 Interactive", "📊 Frequencies"])

    #######################################################
    # STATIC
    #######################################################
    with tabs[0]:
        fig_cols = 2 if layout_choice > 1 else 1
        fig_rows = (layout_choice + 1) // 2 if layout_choice > 2 else (1 if layout_choice < 3 else 2)
        fig = plt.figure(figsize=(8 * fig_cols, 6 * fig_rows))
        gs = fig.add_gridspec(fig_rows, fig_cols)

        idx = 0
        for quad_label, freq_dat in quadrant_freqs.items():
            row_ = idx // fig_cols
            col_ = idx % fig_cols
            ax_ = fig.add_subplot(gs[row_, col_])

            if freq_dat:
                # Build a WordCloud using highlight or colormap
                if highlight_set:
                    def color_func(word, *args, **kwargs):
                        return "rgb(68, 57, 131)" if word.lower() in highlight_set else "hsl(0, 0%, 55%)"
                    wc = WordCloud(
                        width=800,
                        height=600,
                        background_color='white',
                        collocations=False,
                        max_words=exact_words,
                        color_func=color_func
                    )
                else:
                    import matplotlib.cm as cm
                    selected_cmap = cm.get_cmap(chosen_cmap)
                    random_state = np.random.RandomState(42)
                    def color_func(word, font_size, position, orientation, random_state=random_state, **kwargs):
                        color_idx = random_state.randint(0, 256)
                        r, g, b, _ = selected_cmap(color_idx / 255.0)
                        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
                    wc = WordCloud(
                        width=800,
                        height=600,
                        background_color='white',
                        collocations=False,
                        max_words=exact_words,
                        color_func=color_func
                    )

                # Rebuild text from frequencies
                word_list = []
                for w, f in freq_dat:
                    word_list.extend([w] * f)
                joined_text = ' '.join(word_list)
                wc.generate(joined_text)

                ax_.imshow(wc, interpolation='bilinear')
                ax_.axis('off')
            else:
                ax_.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
            idx += 1

        st.pyplot(fig)
        # Download PNG
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            "💾 Download PNG",
            data=buf.getvalue(),
            file_name="wordcloud_quadrants.png",
            mime="image/png",
            use_container_width=True
        )
        plt.close(fig)

        # Also let user download frequencies used in this static view
        # Combined Frequencies
        with st.expander("Download Frequencies (Static View)"):
            # Combined
            all_freq_rows = []
            for quad_label, freq_d in quadrant_freqs.items():
                for w, f in freq_d:
                    all_freq_rows.append({'Quadrant': quad_label, 'Word': w, 'Frequency': f})
            if all_freq_rows:
                freqdf = pd.DataFrame(all_freq_rows)
                freq_csv = freqdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download All Quadrants CSV",
                    data=freq_csv,
                    file_name="wordcloud_all_quadrants_static.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            # Quadrant by quadrant
            for quad_label, freq_d in quadrant_freqs.items():
                st.markdown(f"**{quad_label}**")
                if freq_d:
                    df_ = pd.DataFrame(freq_d, columns=['Word', 'Frequency'])
                    freq_csv = df_.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"Download {quad_label} CSV",
                        data=freq_csv,
                        file_name=f"wordcloud_{quad_label.replace(' ','_')}_static.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.write("No data.")

    #######################################################
    # INTERACTIVE
    #######################################################
    with tabs[1]:
        rows_ = (layout_choice + 1) // 2 if layout_choice > 2 else (1 if layout_choice < 3 else 2)
        cols_ = 2 if layout_choice > 1 else 1

        fig = make_subplots(rows=rows_, cols=cols_, subplot_titles=[k for k in quadrant_freqs.keys()])
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
                    fig.add_trace(trace, row=row_, col=col_)
                fig.update_xaxes(visible=False, showgrid=False, zeroline=False, row=row_, col=col_)
                fig.update_yaxes(visible=False, showgrid=False, zeroline=False, row=row_, col=col_)

            idx += 1

        fig.update_layout(
            width=1000 + 400 * (cols_ - 1),
            height=600 * rows_,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            title="",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Combined CSV (interactive view)
        with st.expander("Download Frequencies (Interactive View)"):
            all_freq_rows = []
            for quad_label, freq_d in quadrant_freqs.items():
                for w, f in freq_d:
                    all_freq_rows.append({'Quadrant': quad_label, 'Word': w, 'Frequency': f})
            if all_freq_rows:
                freqdf = pd.DataFrame(all_freq_rows)
                freq_csv = freqdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download All Quadrants CSV",
                    data=freq_csv,
                    file_name="wordcloud_all_quadrants_interactive.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Download each quadrant's CSV
            for quad_label, freq_d in quadrant_freqs.items():
                st.markdown(f"**{quad_label}**")
                if freq_d:
                    df_ = pd.DataFrame(freq_d, columns=['Word', 'Frequency'])
                    freq_csv = df_.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"Download {quad_label} CSV",
                        data=freq_csv,
                        file_name=f"wordcloud_{quad_label.replace(' ','_')}_interactive.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.write("No data.")

    #######################################################
    # FREQUENCIES TABLE
    #######################################################
    with tabs[2]:
        st.markdown("### Frequency Tables")
        for quad_label, freq_dat in quadrant_freqs.items():
            st.subheader(quad_label)
            if freq_dat:
                df_ = pd.DataFrame(freq_dat, columns=['Word', 'Frequency'])
                st.dataframe(df_, use_container_width=True)
            else:
                st.write("No data.")

        # Combined CSV from the Frequencies Table
        with st.expander("Download Frequencies (Table View)"):
            all_freq_rows = []
            for quad_label, freq_d in quadrant_freqs.items():
                for w, f in freq_d:
                    all_freq_rows.append({'Quadrant': quad_label, 'Word': w, 'Frequency': f})
            if all_freq_rows:
                freqdf = pd.DataFrame(all_freq_rows)
                freq_csv = freqdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download All Quadrants CSV",
                    data=freq_csv,
                    file_name="wordcloud_all_quadrants_table.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            # Quadrant by quadrant
            for quad_label, freq_d in quadrant_freqs.items():
                st.markdown(f"**{quad_label}**")
                if freq_d:
                    df_ = pd.DataFrame(freq_d, columns=['Word', 'Frequency'])
                    freq_csv = df_.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"Download {quad_label} CSV",
                        data=freq_csv,
                        file_name=f"wordcloud_{quad_label.replace(' ','_')}_table.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.write("No data.")


###############################################################################
# 8) TOPIC DISCOVERY
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
# 9) SENTIMENT ANALYSES
###############################################################################
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

def create_sentiment_radar(sentiment_summary):
    """
    Build a radar chart using a summary list of dicts that looks like:
      [
        {
          'Group': ...,
          'Total': ...,
          'Positive%': ...,
          'Neutral%': ...,
          'Negative%': ...,
          'AvgCompound': ...
        },
        ...
      ]
    We'll map:
      Positive% -> radius
      Neutral%  -> radius
      Negative% -> radius
      AvgCompound -> scaled from -1..1 to 0..100 for visualization.
    """
    categories = ["Positive%", "Neutral%", "Negative%", "Compound(Scaled)"]

    fig = go.Figure()
    for row in sentiment_summary:
        grp = row["Group"]
        try:
            pos_val = float(row["Positive%"])
            neu_val = float(row["Neutral%"])
            neg_val = float(row["Negative%"])
            # Convert compound (-1..1) into 0..100
            avg_comp = float(row["AvgCompound"])
            comp_scaled = (avg_comp + 1.0) * 50.0
        except:
            # If parsing fails, skip
            continue

        fig.add_trace(go.Scatterpolar(
            r=[pos_val, neu_val, neg_val, comp_scaled],
            theta=categories,
            fill='toself',
            name=str(grp)
        ))

    fig.update_layout(
        title="Sentiment Radar Chart",
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100])
        ),
        showlegend=True,
        width=700,
        height=600
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
# 10) THEMATIC EVOLUTION
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
# 11) APP LAYOUT (No Nested Expanders, Improved Aesthetics & Error Locations)
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

        # Choose type of analysis
        st.markdown("**Choose Analysis**")
        analyses = [
            "Open Coding",
            "Word Cloud",
            "Word Analysis",
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

        # Group by
        if grp_cols:
            st.subheader("Group By (optional)")
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

    # Determine which section of the dashboard to show
    if st.session_state.selected_analysis == "Open Coding":
        st.markdown("## Open Coding")
        render_open_coding_interface(chosen_var, rdict, open_var_options, grouping_columns)

    elif st.session_state.selected_analysis == "Word Cloud":
        st.markdown("## 🎨 Word Cloud")
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; 
                    background-color: #f8f9fa; margin: 10px 0;">
            <div style="color: #2E7D32; font-size: 1.2em; margin-bottom: 10px; 
                        font-weight: bold;">
                Primary Question
            </div>
            <div style="color: #1a1a1a; font-size: 1.1em; line-height: 1.5; 
                        font-weight: 600;">
                {open_var_options.get(chosen_var, "No question text")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_wordclouds(var_resps)

    elif st.session_state.selected_analysis == "Word Analysis":
        st.markdown("## 📊 Word Analysis")
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; 
                    background-color: #f8f9fa; margin: 10px 0;">
            <div style="color: #2E7D32; font-size: 1.2em; margin-bottom: 10px; 
                        font-weight: bold;">
                Primary Question
            </div>
            <div style="color: #1a1a1a; font-size: 1.1em; line-height: 1.5; 
                        font-weight: 600;">
                {open_var_options.get(chosen_var, "No question text")}
            </div>
        </div>
        """, unsafe_allow_html=True)

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

    elif st.session_state.selected_analysis == "Topic Discovery":
        st.markdown("## 🔍 Topic Discovery")
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; 
                    background-color: #f8f9fa; margin: 10px 0;">
            <div style="color: #2E7D32; font-size: 1.2em; margin-bottom: 10px; 
                        font-weight: bold;">
                Primary Question
            </div>
            <div style="color: #1a1a1a; font-size: 1.1em; line-height: 1.5; 
                        font-weight: 600;">
                {open_var_options.get(chosen_var, "No question text")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        num_topics = st.slider("Number of Topics (K)", 2, 10, 4)
        min_topic_size = st.slider("Min Topic Size (unused for basic KMeans)", 2, 5, 2)

        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        for key_, arr_ in var_resps.items():
            st.subheader(f"{key_} ({len(arr_)} responses)")
            cleaned_txts = [tx for tx in arr_ if isinstance(tx, str) and tx.strip()]
            if len(cleaned_txts) < 20:
                st.warning("Not enough data (<20) to do topic modeling.")
                continue

            embeddings_ = emb_model.encode(cleaned_txts, show_progress_bar=False)
            kmeans_ = KMeans(n_clusters=num_topics, random_state=42).fit(embeddings_)

            from collections import defaultdict

            cluster_map = defaultdict(list)
            for doc_, lbl_ in zip(cleaned_txts, kmeans_.labels_):
                cluster_map[lbl_].append(doc_)

            stats_list = []
            cvec = CountVectorizer(max_features=10, stop_words=list(st.session_state.custom_stopwords))
            for c_ in sorted(cluster_map.keys()):
                group_docs = cluster_map[c_]
                if not group_docs:
                    continue
                X_ = cvec.fit_transform(group_docs)
                w_ = cvec.get_feature_names_out()
                fr_ = X_.sum(axis=0).A1
                top_ = sorted(zip(w_, fr_), key=lambda x: x[1], reverse=True)[:5]
                stats_list.append({
                    'Topic': c_,
                    'Count': len(group_docs),
                    'Top Terms': ", ".join([t[0] for t in top_])
                })
            st.dataframe(pd.DataFrame(stats_list), use_container_width=True)

    elif st.session_state.selected_analysis == "Sentiment Analysis":

        st.markdown("## ❤️ Sentiment Analysis")

        if not chosen_grp:

            st.warning("Please select a 'Group By' variable for sentiment comparison.")

        else:

            analyzer = SentimentIntensityAnalyzer()

            results_dict = defaultdict(lambda: {'count': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'scores': []})

            for key_, texts_ in var_resps.items():

                for txt_ in texts_:

                    if not isinstance(txt_, str) or not txt_.strip():
                        continue

                    sc = analyzer.polarity_scores(txt_)

                    results_dict[key_]['count'] += 1

                    results_dict[key_]['scores'].append(sc['compound'])

                    if sc['compound'] >= 0.05:

                        results_dict[key_]['pos'] += 1

                    elif sc['compound'] <= -0.05:

                        results_dict[key_]['neg'] += 1

                    else:

                        results_dict[key_]['neu'] += 1

            # Build summary rows

            sum_rows = []

            for gkey, stats_ in results_dict.items():

                c_ = stats_['count']

                if c_ > 0:
                    avg_c = np.mean(stats_['scores'])

                    p_ = (stats_['pos'] / c_) * 100

                    n_ = (stats_['neg'] / c_) * 100

                    u_ = (stats_['neu'] / c_) * 100

                    sum_rows.append({

                        'Group': gkey,

                        'Total': c_,

                        'Positive%': f"{p_:.1f}",

                        'Neutral%': f"{u_:.1f}",

                        'Negative%': f"{n_:.1f}",

                        'AvgCompound': f"{avg_c:.3f}"

                    })

            if not sum_rows:

                st.warning("No sentiment data found.")

            else:

                # 1) Radar Chart FIRST

                radar_fig = create_sentiment_radar(sum_rows)

                st.plotly_chart(radar_fig, use_container_width=True)

                # 2) Violin / Distribution

                points_data = []

                for grp_data, stats_ in results_dict.items():

                    for val_ in stats_['scores']:
                        points_data.append({'Group': grp_data, 'Compound': val_})

                if points_data:
                    df_vio = pd.DataFrame(points_data)

                    fig_vio = px.violin(df_vio, x='Group', y='Compound', box=True, points='all')

                    st.plotly_chart(fig_vio, use_container_width=True)

                # 3) Summaries Table

                st.dataframe(pd.DataFrame(sum_rows), use_container_width=True)

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

