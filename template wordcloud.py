import os
import pandas as pd
import string
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. PATHS & STOPWORDS (Update path logic if desired)
# ---------------------------------------------------------------------------
current_dir = os.getcwd()
qwels_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
data_path = os.path.join(qwels_path, "ANALYSIS", "DATASETS")
text_path = os.path.join(data_path, "textsets")

# Create/verify a folder to save visuals in your *current* directory
visuals_path = os.path.join(current_dir, "visuals")
if not os.path.exists(visuals_path):
    os.mkdir(visuals_path)

synonyms_dict = {}
# Custom stopwords
custom_stops = {
    "miss", "would", "able", "most", "don", "s", "d", "t", "working", "work", "shit", "almost", "day", "daily",
    "much", "really", "take", "m", "really", "see", "check", "na", "hard", "example", "still", "worker",
    "going", "sure", "need", "wouldn", "well", "stop", "sometimes", "part", "go", "mi", "improve", "instant", "live",
    "job", "jobs", "role", "roles", "around", "good", "student", "life", "fun", "para", "skills", "workforce", "etre",
    "co", "coworker", "coworkers", "sense", "will", "every", "interesting", "mostly", "skill", "aidais",
    "would", "could", "should", "might", "maybe", "like", "just", "really", "always", "opportunity", "professional",
    "nothing", "anything", "everything", "something", "someone", "somebody", "plus", "better", "suddenly",
    "anyone", "anybody", "everyone", "everybody", "thing", "things", "stuff", "lot", "abilities", "consistently",
    "dont", "don", "didn", "didn't", "couldn't", "shouldn't", "wouldn't", "re", "seeing", "lol", "nouvelles",
    "can't", "cannot", "wanna", "gonna", "sorta", "kinda", "time", "enjoy", "keep", "ok", "4th",
    "think", "feel", "felt", "get", "got", "make", "makes", "making", "way", "ways", "love", "french", "english",
    "miss", "most", "people", "able", "day", "isn", "times", "great", "find", "often", "many", "deviate", "week", "close",
    "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "donc", "car", "ni", "gens", "yellow", "puts", "gta", "mieux",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "constantly", "manque", "kept", "amount", "faut", "vivre", "ultimately",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur", "true", "small", "difference", "today", "tell",
    "à", "de", "par", "pour", "en", "dans", "sur", "avec", "feeling", "thought", "question", "point", "bon",
    "que", "qui", "quoi", "dont", "où", "être", "avoir", "faire", "dire", "eventually", "company", "realize", "realized",
    "plus", "moins", "très", "bien", "mal", "ici", "là", "oui", "non", "trop", "yet", "totally", "days", "taken",
    "au", "aux", "du", "des", "si", "même", "tout", "tous", "toute", "toutes", "ont", "especially", "align", "aline",
    "autre", "autres", "après", "avant", "puis", "enfin", "encore", "toujours", "jamais", "quite", "very",
    "rien", "personne", "quelque", "quelques", "sans", "sous", "vers", "est", "nes", "assez", "pas", "n", "jour",
    "comme", "aussi", "alors", "donc", "j", "ve", "l", "se", "ne", "doesn", "suis", "dois", "aucune", "sont", "say",
    "mes", "moi", "change", "thinking", "emploi", "travail", "ca", "never", "less", "more", "bit", "pense", "even", "paramount",
    "c", "sondage", "taking", "new", "tru", "cela", "actually", "opinion", "chose", "etc", "dint", "end", "couldn", "expect",
    "x000d", "goalx", "vie", "want", "argent", "gagne", "bonne", "ma", "peu", "must", "side", "enables", "facilitates", "rend",
    "fais", "fait", "voir", "parce", "dois", "salaire", "qualité", "peut", "ainsi", "envie", "ça", "quand",
    "tâche", "veux", "aime", "travaille", "domaine", "getting", "giving", "put", "rather", "looking",
    "whatever", "enough", "mean", "made", "within", "probably", "done", "become", "give", "given", "stay",
    "look", "whether", "back", "first", "now", "another", "due", "without", "comes", "became", "simply",
    "already", "currently", "later", "weren", "y", "important", "qu", "us",
    "away", "current", "past", "future", "pretty", "high", "higher", "low", "lower", "though", "previous",
    "certain", "left", "right", "quit", "told", "went", "took", "started", "treated", "found", "larger", "received",
    "using", "comes", "allows", "allows", "comes", "bring", "brings", "one", "pretty", "wasn", "found", "extra", "though",
    "choosing", "level", "levels", "kind", "ago", "last", "update", "updated", "become", "became", "gets", "word",
    "getting", "gaining", "wanted", "basically", "mainly", "mostly", "specifically", "generally", "said",
    "little", "caused", "long", "employeur", "disagree", "becomes", "e", "fois", "lorsque",
    "seem", "seemed", "seems", "somewhat", "somehow", "somewhere", "anywhere", "nowhere", "either", "neither",
    "among", "amongst", "beside", "besides", "beyond", "during", "except", "despite", "unless", "until",
    "upon", "within", "throughout", "across", "along", "around", "through", "toward", "towards", "onto",
    "into", "onto", "inside", "outside", "near", "nearby", "far", "further", "while", "whilst", "whereas",
    "meanwhile", "nowadays", "perhaps", "possibly", "probably", "presumably", "surely", "certainly", "definitely",
    "absolutely", "completely", "totally", "entirely", "fully", "rather", "quite", "fairly", "somewhat",
    "depuis", "pendant", "selon", "suivant", "malgré", "outre", "parmi", "durant", "derrière", "devant",
    "entre", "envers", "hormis", "jusque", "moyennant", "nonobstant", "sauf", "selon", "voici", "voilà",
    "certes", "cependant", "néanmoins", "pourtant", "toutefois", "auprès", "dedans", "dehors", "dessous",
    "dessus", "ensemble", "partout", "pourquoi", "comment", "quand", "combien", "lequel", "laquelle",
    "lesquels", "lesquelles", "duquel", "desquels", "desquelles", "auquel", "auxquels", "auxquelles",
    "celui", "celle", "ceux", "celles", "chaque", "plusieurs", "beaucoup", "tellement", "autant",
    "parfois", "souvent", "rarement", "maintenant", "bientôt", "aussitôt", "tantôt", "ensuite", "jadis",
    "full", "per", "11", "ll", "know", "clear", "aspect", "gives", "depends", "receive", "set", "poste", "use",
    "employment", "bureau", "commence", "difficile", "titre", "gave", "ran", "usually", "goes", "avons", "concret",
    "immediately", "surveys", "aide", "idea", "issue", "come", "technical", "suivi", "must", "takes", "turn", "place",
    "dossier", "usually", "typically", "events", "station", "matter", "seriously", "try", "survey", "questions", "question",
    "working", "avais", "continue", "stop",
    
    # Common expressions
    "gonna", "wanna", "gotta", "sorta", "kinda", "dunno", "lemme", "gimme", "ain't", "y'all",
    "allons", "vais", "vas", "allez", "vont", "dois", "doit", "devons", "devez", "doivent",

    # Time-related
    "today", "tomorrow", "yesterday", "morning", "evening", "night", "week", "month", "year",
    "aujourd'hui", "demain", "hier", "matin", "soir", "nuit", "semaine", "mois", "année",

    # Quantities
    "many", "much", "few", "several", "some", "any", "all", "none", "both", "either",
    "beaucoup", "peu", "plusieurs", "quelques", "certains", "tous", "toutes", "aucun", "aucune",

    # Additional common words
    "anyway", "anyhow", "elsewhere", "everywhere", "whenever", "whoever", "whatever", "whichever",
    "ailleurs", "partout", "quiconque", "quelconque", "n'importe", "parfois", "désormais"
}
all_stops = set(STOPWORDS).union(custom_stops)


def remove_punctuation_and_stopwords(text: str) -> str:
    """
    Removes punctuation, splits text, and removes any words found in all_stops.
    """
    # Ensure text is lowercase before processing
    text = text.lower()

    # Remove punctuation and replace with space
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Split by whitespace and remove extra spaces
    words = [word.strip() for word in text.split()]

    # Filter out stopwords and empty strings
    filtered_words = [w for w in words if w and w not in all_stops]

    return " ".join(filtered_words)


def group_synonyms(text: str) -> str:
    """
    Replace words in 'text' that appear in synonyms_dict with the main key.
    Ensures synonyms (e.g., 'income', 'salary', etc.) all become 'money'.
    This combines their counts under a single main term in the final wordcloud.
    """
    # Convert text to lowercase and split into tokens
    tokens = text.lower().split()

    # Build a flat lookup dictionary for faster processing
    synonym_lookup = {}
    for main_term, synonyms in synonyms_dict.items():
        # Add all synonyms AND the main term itself to map to the main term
        for synonym in synonyms:
            synonym_lookup[synonym.lower()] = main_term
        # Explicitly add the main term to map to itself
        synonym_lookup[main_term.lower()] = main_term

    # Map tokens to their main terms
    mapped_tokens = []
    for token in tokens:
        # Use get() with the token itself as default if not found
        mapped_token = synonym_lookup.get(token, token)
        mapped_tokens.append(mapped_token)

    return " ".join(mapped_tokens)


def tokenize_ngrams(text: str, n: int = 1) -> str:
    """
    Create n-grams from a string (default: unigrams).
    Ensures proper handling of text by splitting on whitespace.
    """
    # Split text into words, handling multiple spaces
    words = [w for w in text.split() if w]

    # Create n-grams
    if n == 1:
        return " ".join(words)

    ngrams = zip(*[words[i:] for i in range(n)])
    return " ".join("_".join(ngram) for ngram in ngrams)


def highlight_color_func_generator(highlight_words: set, use_viridis: bool = False):
    """
    Creates a color_func for WordCloud that highlights specific terms.

    Parameters:
    - highlight_words: set of words to highlight
    - use_viridis: bool, if True uses viridis colormap for highlights, if False uses red
    """
    if not highlight_words:
        return None

    if use_viridis:
        from matplotlib import colormaps
        import numpy as np

        # Get viridis colormap
        viridis = colormaps['viridis']

        # Create evenly spaced colors for highlighted words
        highlight_words_list = sorted(list(highlight_words))  # Sort for consistency
        n_words = len(highlight_words_list)
        color_positions = np.linspace(0, 0.9, n_words)  # Use 0.9 to avoid too-dark colors

        # Create color mapping dictionary
        highlight_colors = {}
        for word, pos in zip(highlight_words_list, color_positions):
            rgba = viridis(pos)
            # Convert to RGB string
            highlight_colors[word] = f"rgb({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)})"

    def highlight_color_func(word, font_size, position, orientation, font_path, random_state):
        word_lower = word.lower()
        if word_lower in highlight_words:
            if use_viridis:
                return highlight_colors[word_lower]
            else:
                return "hsl(0, 100%, 40%)"  # bright red
        return "hsl(0, 0%, 50%)"  # medium gray for non-highlighted words

    return highlight_color_func

def generate_wordcloud_figure(
        filename,
        sheet_name,
        rating_var=None,
        open_text_var=None,
        categories_list=None,
        main_title="",
        subtitle="",
        source_text="",
        output_filename=None,
        open_text_var2=None,
        highlight_words=None,
        use_viridis_for_highlights=False
):
    """
    Creates either:
      - A single wordcloud if no categories are provided, or
      - A 2x2 subplot of wordclouds for different categories of the same variable.

    By default:
      - Collocations are disabled (collocations=False) so only single tokens show up.
      - synonyms_dict is applied so synonyms merge under the same key.
      - highlight_words can be used to color particular words in red.
      - The figure size is set large and the WordCloud is scaled to better fill space.

      Parameters:
      - filename: str -> name of the Excel file (e.g. 'MESSI_CAN_2024_MAY.xlsx')
      - sheet_name: str -> name of the sheet (e.g. 'REDUCED')
      - rating_var: str or None -> rating variable (e.g. 'improve'). If None, no categories are used
      - open_text_var: str or None -> name of the primary open-text column
      - categories_list: list or None -> up to 4 categories to display in separate subplots
      - main_title: str -> main title of the figure
      - subtitle: str -> subtitle of the figure
      - source_text: str -> text to display at the bottom (data source or notes)
      - output_filename: str or None -> if given, saves the figure to this filename in the 'visuals' folder
      - open_text_var2: str or None -> a second open-text column to combine with open_text_var
      - highlight_words: iterable or None -> words you want highlighted in a special color
    """

    # Convert highlight_words to a set if it's provided
    if highlight_words is not None:
        highlight_words = set(highlight_words)
    else:
        highlight_words = set()

    # Prepare our color_func with the new viridis option
    color_func = highlight_color_func_generator(highlight_words, use_viridis_for_highlights)

    file_path = os.path.join(text_path, filename)
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if not open_text_var:
        print("No open_text_var specified. Cannot generate a wordcloud.")
        return

    # -----------------------------------------------------------------------
    # A) CLEAN & FILTER OPEN-TEXT RESPONSES
    # -----------------------------------------------------------------------
    # If we have a second open-text column, combine them.
    if open_text_var2:
        df_subset = df.dropna(subset=[open_text_var, open_text_var2], how='all').copy()
        df_subset["combined_text"] = (
            df_subset[open_text_var].fillna("") + " " + df_subset[open_text_var2].fillna("")
        )
        df_subset["combined_text"] = df_subset["combined_text"].str.strip().str.lower()
        # Exclude any row containing "__DK__" in combined_text
        df_subset = df_subset[~df_subset["combined_text"].str.contains("__dk__", na=False)]
        # We'll call this final_text
        df_subset.rename(columns={"combined_text": "final_text"}, inplace=True)
    else:
        df_subset = df[df[open_text_var].notna()].copy()
        df_subset = df_subset[df_subset[open_text_var].str.strip().ne("__DK__")].copy()
        df_subset[open_text_var] = df_subset[open_text_var].str.lower()
        df_subset.rename(columns={open_text_var: "final_text"}, inplace=True)

    total_responses = len(df_subset)

    # Prepare our color_func if highlight_words provided
    color_func = highlight_color_func_generator(highlight_words)

    # -----------------------------------------------------------------------
    # B) SINGLE WORDCLOUD (no rating_var/categories_list)
    # -----------------------------------------------------------------------
    if not rating_var or not categories_list:
        if total_responses == 0:
            print("No valid text found to generate wordcloud.")
            return

        # 1) Remove punctuation & stopwords
        df_subset["clean_text"] = df_subset["final_text"].apply(remove_punctuation_and_stopwords)
        # 2) Group synonyms so synonyms get merged
        df_subset["grouped_text"] = df_subset["clean_text"].apply(group_synonyms)
        # 3) Make unigrams
        df_subset["ngram_text"] = df_subset["grouped_text"].apply(lambda x: tokenize_ngrams(x, n=1))
        # Combine all
        all_text = " ".join(df_subset["ngram_text"])

        # Increase width/height & tweak max_font_size to fill more space
        wc = WordCloud(
            width=2000,
            height=1200,
            background_color="white",
            collocations=False,
            stopwords=all_stops,
            colormap="viridis",
            color_func=highlight_color_func_generator(highlight_words, use_viridis_for_highlights),
            max_words=200,
            max_font_size=300,
            prefer_horizontal=1,
            scale=2
        ).generate(all_text)

        # Create a figure so we can place title/subtitle in top-left corner
        fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")

        # Title & Subtitle at top-left
        fig.text(
            0.05, 0.92,
            main_title,
            fontsize=22,
            fontweight='bold',
            ha='left'
        )
        fig.text(
            0.05, 0.90,
            subtitle,
            fontsize=18,
            ha='left',
            va='top',
            color='#404040'
        )

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        # Footnote
        fig.text(
            0.05, 0.03,
            (
                f"Note: {total_responses} open-text responses used. '__DK__' removed.\n"
                f"Source: {source_text}"
            ),
            fontsize=18,
            ha='left',
            va='bottom',
            color='#404040'
        )

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(
            top=0.88,
            bottom=0.05,
            left=0.07,
            right=0.95,
            wspace=0.1,
            hspace=0.1
        )

        if output_filename:
            save_path = os.path.join(visuals_path, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()
        return

    # -----------------------------------------------------------------------
    # C) MULTI WORDCLOUD (2x2 by categories)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten()
    fig.patch.set_facecolor('white')

    # Title & Subtitle at top-left
    fig.text(
        0.05, 0.96,
        main_title,
        fontsize=22,
        fontweight='bold',
        ha='left'
    )
    fig.text(
        0.05, 0.94,
        subtitle,
        fontsize=18,
        ha='left',
        va='top',
        color='#404040'
    )

    # Loop through categories (up to 4)
    for i, cat in enumerate(categories_list):
        ax = axes[i]
        subset_cat = df_subset[df_subset[rating_var] == cat].copy()

        if subset_cat.empty:
            ax.axis("off")
            ax.set_title(
                f"Those who said '{cat}'\n(No valid text)",
                fontsize=15,
                fontweight='bold'
            )
            continue

        # 1) Remove punctuation & stopwords
        subset_cat["clean_text"] = subset_cat["final_text"].apply(remove_punctuation_and_stopwords)
        # 2) Group synonyms
        subset_cat["grouped_text"] = subset_cat["clean_text"].apply(group_synonyms)
        # 3) Make unigrams
        subset_cat["ngram_text"] = subset_cat["grouped_text"].apply(lambda x: tokenize_ngrams(x, 1))
        cat_text = " ".join(subset_cat["ngram_text"]).strip()

        if not cat_text:
            ax.axis("off")
            ax.set_title(
                f"Those who said '{cat}'\n(No valid text)",
                fontsize=15,
                fontweight='bold'
            )
            continue

        wc = WordCloud(
            width=1500,
            height=1000,
            background_color="white",
            collocations=False,
            stopwords=all_stops,
            colormap="viridis",
            color_func=highlight_color_func_generator(highlight_words, use_viridis_for_highlights),
            max_words=100,
            max_font_size=200,
            prefer_horizontal=1,
            scale=2
        ).generate(cat_text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Those who said '{cat}'\nTotal responses: {len(subset_cat)}",
            fontsize=15,
            fontweight='bold'
        )

    # Turn off extra subplots if fewer than 4 categories
    for j in range(len(categories_list), 4):
        axes[j].axis("off")

    # Footnote
    fig.text(
        0.05, 0.03,
        (
            f"Note: {total_responses} open-text responses used. '__DK__' responses removed.\n"
            f"Source: {source_text}"
        ),
        fontsize=18,
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

    # Save if requested
    if output_filename:
        save_path = os.path.join(visuals_path, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def generate_comparison_wordclouds(
        filename,
        sheet_name,
        open_text_var1,
        open_text_var2,
        main_title="",
        subtitle="",
        source_text="",
        label1="Variable 1",
        label2="Variable 2",
        output_filename=None,
        highlight_words=None,
        use_viridis_for_highlights=False
):
    """
    Creates two wordclouds side by side comparing different variables.

    Parameters:
    - filename: str -> name of the Excel file
    - sheet_name: str -> name of the sheet
    - open_text_var1: str -> name of the first open-text column
    - open_text_var2: str -> name of the second open-text column
    - main_title: str -> main title of the figure
    - subtitle: str -> subtitle of the figure
    - source_text: str -> text to display at the bottom
    - label1: str -> label for the first wordcloud
    - label2: str -> label for the second wordcloud
    - output_filename: str or None -> if given, saves the figure
    - highlight_words: iterable or None -> words to highlight
    - use_viridis_for_highlights: bool -> whether to use viridis colormap for highlights
    """

    # Convert highlight_words to a set if provided
    if highlight_words is not None:
        highlight_words = set(highlight_words)
    else:
        highlight_words = set()

    # Prepare color_func with viridis option
    color_func = highlight_color_func_generator(highlight_words, use_viridis_for_highlights)

    file_path = os.path.join(text_path, filename)
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('white')

    # Title & Subtitle at top
    fig.text(
        0.05, 0.96,
        main_title,
        fontsize=22,
        fontweight='bold',
        ha='left'
    )
    fig.text(
        0.05, 0.94,
        subtitle,
        fontsize=18,
        ha='left',
        va='top',
        color='#404040'
    )

    # Process each variable
    for i, (ax, var, label) in enumerate([(ax1, open_text_var1, label1), (ax2, open_text_var2, label2)]):
        # Filter and clean data
        df_subset = df[df[var].notna()].copy()
        df_subset = df_subset[df_subset[var].str.strip().ne("__DK__")].copy()
        total_responses = len(df_subset)

        if total_responses == 0:
            ax.axis("off")
            ax.set_title(
                f"{label}\n(No valid text)",
                fontsize=15,
                fontweight='bold'
            )
            continue

        # Clean text
        df_subset["clean_text"] = df_subset[var].apply(remove_punctuation_and_stopwords)
        df_subset["grouped_text"] = df_subset["clean_text"].apply(group_synonyms)
        df_subset["ngram_text"] = df_subset["grouped_text"].apply(lambda x: tokenize_ngrams(x, n=1))
        all_text = " ".join(df_subset["ngram_text"])

        wc = WordCloud(
            width=1500,
            height=1000,
            background_color="white",
            collocations=False,
            stopwords=all_stops,
            colormap="viridis",
            color_func=highlight_color_func_generator(highlight_words, use_viridis_for_highlights),
            max_words=150,
            max_font_size=200,
            prefer_horizontal=1,
            scale=2
        ).generate(all_text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"{label}\nTotal responses: {total_responses}",
            fontsize=15,
            fontweight='bold'
        )

    # Footnote
    fig.text(
        0.05, 0.03,
        f"Note: '__DK__' responses removed.\nSource: {source_text}",
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

    if output_filename:
        save_path = os.path.join(visuals_path, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

# ---------------------------------------------------------------------------
# EXAMPLE USAGE:
# ---------------------------------------------------------------------------

generate_wordcloud_figure(
    filename="MESSI_US_2024_JUNE.xlsx",
    sheet_name="MISSMOST",
    open_text_var="missmost_open",
    main_title="What Would Be Missed Most About Work?",
    subtitle="'If you were to suddenly stop working, what do you think you'd miss the most?'",
    source_text="2024 Measuring Economic Sentiment and Social Inequality Survey (MESSI-US)",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose", "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family", "flexibility", "underpaid"},
    use_viridis_for_highlights=True,
    output_filename="missmost_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_US_2024_JUNE.xlsx",
    sheet_name="JOBFUN",
    rating_var="jobfun",
    open_text_var="jobfun_open",
    categories_list=["often", "sometimes", "rarely", "never"],
    main_title="What is Fun About Work?",
    subtitle="'How often would you say your job is fun? Please briefly tell us why you answered that way.'",
    source_text="2024 Measuring Economic Sentiment and Social Inequality Survey (MESSI-US)",
    output_filename="jobfun_categories_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    rating_var="improve",
    open_text_var="improve_open",
    categories_list=[" strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="The Chance to Improve",
    subtitle="'My job gives me a chance to improve my skills.'",
    source_text="2024 Measuring Economic Sentiment and Social Inequality Survey (MESSI-CAN)",
    output_filename="improve_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    rating_var="opdevel",
    open_text_var="opdevel_open",
    categories_list=["very true", "somewhat true", "not too true", "not at all true"],
    main_title="Opportunity to Develop My Abilities",
    subtitle="'I have an opportunity to develop my own special abilities.'",
    source_text="2024 Measuring Economic Sentiment and Social Inequality Survey (MESSI-CAN)",
    output_filename="opdevel_wordcloud.png"
)

# A job is just a way of earning money—no more.
generate_wordcloud_figure(
    filename="QES_US_2023.xlsx",
    sheet_name="ALL",
    rating_var="a_wrkearn",
    open_text_var="a_wrkearn_open",
    categories_list=["strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="When Work is Only about the Money.",
    subtitle="A job is just a way of earning money—no more.",
    source_text="2022 Quality of Employment Survey (QES-US)",
    output_filename="wrkearn_wordcloud.png"
)

# If you had enough money to live comfortably for the rest of your life, would you continue to work or stop working?
generate_wordcloud_figure(
    filename="QES_US_2022.xlsx",
    sheet_name="ALL",
    open_text_var="stopwork_open",
    main_title="Would you continue to work or stop working...",
    subtitle="if you had enough to live comfortably?",
    source_text="2022 Quality of Employment Survey (QES-US)",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose",
                     "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family",
                     "flexibility"},
    use_viridis_for_highlights=True,
    output_filename="stopwork_wordcloud.png"
)

generate_comparison_wordclouds(
    filename="QES_US_2022.xlsx",
    sheet_name="ALL",
    open_text_var1="stopwork_open",
    open_text_var2="continuework_open",
    main_title="Would you continue to work or stop working...",
    subtitle="if you had enough to live comfortably?",
    source_text="QES_US_2022",
    label1="Continue",
    label2="Stop",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose", "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family", "flexibility", "underpaid"},
    use_viridis_for_highlights=True,
    output_filename="stopcontinue_wordcloud.png"
)

# What I do at work is more important to me than the money I earn.
generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    rating_var="workimpt",
    open_text_var="workimpt_open",
    categories_list=[" strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="When Work is More Important than Money",
    subtitle="'What I do at work is more important to me than the money I earn.'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    output_filename="wrkimpt_wordcloud.png"
)

# I would enjoy having a paid job even if I didn’t need the money.
generate_wordcloud_figure(
    filename="QES_US_2023.xlsx",
    sheet_name="WKENJOY",
    rating_var="wrkenjoy",
    open_text_var="wrkenjoy_open",
    categories_list=["strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="Is Enjoyment of Work Possible Without Money?",
    subtitle="'I would enjoy having a paid job even if I didn’t need the money.'",
    source_text="2023 Quality of Employment Survey (QES-US 2023)",
    output_filename="wrkenjoy_wordcloud.png"
)

# How taking the survey changed the way you think or feel about work.
generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    open_text_var="changefeel_open",
    main_title="When a Little More Makes the Difference",
    subtitle="'A little more made a difference'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose",
                     "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family",
                     "flexibility"},
    use_viridis_for_highlights=True,
    output_filename="changefeel_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    open_text_var="changethink_open",
    main_title="When a Little More Makes the Difference",
    subtitle="'A little more made a difference'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose",
                     "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family",
                     "flexibility"},
    use_viridis_for_highlights=True,
    output_filename="changethink_wordcloud.png"
)

# A little more made a difference/A little less made a difference.
generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    open_text_var="littlemore_open",
    main_title="When a Little More Makes the Difference",
    subtitle="'A little more made a difference'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose",
                     "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family",
                     "flexibility"},
    use_viridis_for_highlights=True,
    output_filename="littlemore_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    open_text_var="littleless_open",
    main_title="When a Little Less Makes the Difference",
    subtitle="'A little less made a difference.'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    highlight_words={"money", "helping", "acknowledgement", "relationships", "results", "growth", "novelty", "purpose",
                     "kindness", "stress", "skill", "satisfaction", "fulfillment", "boss", "routine", "family",
                     "flexibility"},
    use_viridis_for_highlights=True,
    output_filename="littleless_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    rating_var="affected",
    open_text_var="affected_open",
    categories_list=[" strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="Affected",
    subtitle="'A lot are affected by the work I do.'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    output_filename="affected_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    rating_var="best",
    open_text_var="best_open",
    categories_list=["very true", "somewhat true", "not too true", "not at all true"],
    main_title="Best",
    subtitle="'Opportunity to do my best.'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    output_filename="best_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    rating_var="personal_interest",
    open_text_var="interest_open",
    categories_list=["very true", "somewhat true", "not too true", "not at all true"],
    main_title="When Workers Take an Interest",
    subtitle="'Coworkers take a personal interest in you.'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    output_filename="personal_interest_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    open_text_var="conscience_open",
    main_title="Conscience",
    subtitle="'I sometimes have a bad conscience at work'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    output_filename="conscience_wordcloud.png"
)

generate_wordcloud_figure(
    filename="MESSI_CAN_2024_MAY.xlsx",
    sheet_name="WHOLE",
    open_text_var="results_open",
    main_title="When results matter",
    subtitle="'I can see the results of my work.'",
    source_text="2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)",
    output_filename="results_wordcloud.png"
)

# How much can you tell about a person by knowing what they do for a living? [FT] knowyes QES_US_2022
generate_wordcloud_figure(
    filename="QES_US_2022.xlsx",
    sheet_name="ALL",
    open_text_var="knowyes_open",
    main_title="The Features of Work That Stand Out",
    subtitle="'How much can you tell about a person by knowing what they do for a living?'",
    source_text="2022 Quality of Employment Survey (QES-US 2022)",
    output_filename="workknow_wordcloud.png"
)

# How much does your job help you understand the sort of person you really are? jobself_open QES_2023
generate_wordcloud_figure(
    filename="QES_US_2023.xlsx",
    sheet_name="ALL",
    rating_var="jobself",
    open_text_var="jobself_open",
    categories_list=["a lot", "somewhat", "a little", "not at all"],
    main_title="Work's Contribution to Self-Realization",
    subtitle="'How much does your job help you understand the sort of person you really are?'",
    source_text="2023 Quality of Employment Survey (QES-US 2023)",
    output_filename="jobself_wordcloud.png"
)

# The most important things that happen to you involve your job. jobimpt_open QES 2023
generate_wordcloud_figure(
    filename="QES_US_2023.xlsx",
    sheet_name="ALL",
    rating_var="jobimpt",
    open_text_var="jobimpt_open",
    categories_list=["strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="When Work Matters... And When it Doesn't",
    subtitle="'The most important things that happen to you involve your job.'",
    source_text="2023 Quality of Employment Survey (QES-US 2023)",
    output_filename="jobimpt_wordcloud.png"
)

# The work I do on my job is meaningful to me. [FT] meaning_open trends_2023
generate_wordcloud_figure(
    filename="TRENDS_CAN_2023_SEP.xlsx",
    sheet_name="ALL",
    rating_var="meaning",
    open_text_var="meaning_open",
    categories_list=["strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="Work Meaning and its Opposite",
    subtitle="'The work I do on my job is meaningful to me.'",
    source_text="2023 Trends Canada (TRENDS-CAN 2023)",
    output_filename="meaning_wordcloud.png"
)

# I really look forward to going to work most days. [FT] lookforward_open trends_2023
generate_wordcloud_figure(
    filename="TRENDS_CAN_2023_SEP.xlsx",
    sheet_name="ALL",
    rating_var="lookforward",
    open_text_var="lookforward_open",
    categories_list=["strongly agree", "agree", "disagree", "strongly disagree"],
    main_title="Looking Forward and Away from Work",
    subtitle="'I really look forward to going to work most days.'",
    source_text="2023 Trends Canada (TRENDS-CAN 2023)",
    output_filename="lookforward_wordcloud.png"
)