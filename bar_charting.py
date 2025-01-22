import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from matplotlib import colormaps
import re
import string
from collections import Counter

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
    "miss", "most", "people", "able", "day", "isn", "times", "great", "find", "often", "many", "deviate", "week",
    "close", "i'm", "we're", "it's", "ask", "don't", "common", "interested", "nos",
    "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "donc", "car", "ni", "gens", "yellow", "puts", "gta",
    "mieux",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "constantly", "manque", "kept", "amount", "faut", "vivre",
    "ultimately",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur", "true", "small", "difference", "today",
    "tell",
    "à", "de", "par", "pour", "en", "dans", "sur", "avec", "feeling", "thought", "question", "point", "bon",
    "que", "qui", "quoi", "dont", "où", "être", "avoir", "faire", "dire", "eventually", "company", "realize",
    "realized",
    "plus", "moins", "très", "bien", "mal", "ici", "là", "oui", "non", "trop", "yet", "totally", "days", "taken",
    "au", "aux", "du", "des", "si", "même", "tout", "tous", "toute", "toutes", "ont", "especially", "align", "aline",
    "autre", "autres", "après", "avant", "puis", "enfin", "encore", "toujours", "jamais", "quite", "very",
    "rien", "personne", "quelque", "quelques", "sans", "sous", "vers", "est", "nes", "assez", "pas", "n", "jour",
    "comme", "aussi", "alors", "donc", "j", "ve", "l", "se", "ne", "doesn", "suis", "dois", "aucune", "sont", "say",
    "mes", "moi", "change", "thinking", "emploi", "travail", "ca", "never", "less", "more", "bit", "pense", "even",
    "paramount",
    "c", "sondage", "taking", "new", "tru", "cela", "actually", "opinion", "chose", "etc", "dint", "end", "couldn",
    "expect",
    "x000d", "goalx", "vie", "want", "argent", "gagne", "bonne", "ma", "peu", "must", "side", "enables", "facilitates",
    "rend",
    "fais", "fait", "voir", "parce", "dois", "salaire", "qualité", "peut", "ainsi", "envie", "ça", "quand",
    "tâche", "veux", "aime", "travaille", "domaine", "getting", "giving", "put", "rather", "looking",
    "whatever", "enough", "mean", "made", "within", "probably", "done", "become", "give", "given", "stay",
    "look", "whether", "back", "first", "now", "another", "due", "without", "comes", "became", "simply",
    "already", "currently", "later", "weren", "y", "important", "qu", "us",
    "away", "current", "past", "future", "pretty", "high", "higher", "low", "lower", "though", "previous",
    "certain", "left", "right", "quit", "told", "went", "took", "started", "treated", "found", "larger", "received",
    "using", "comes", "allows", "allows", "comes", "bring", "brings", "one", "pretty", "wasn", "found", "extra",
    "though",
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
    "dossier", "usually", "typically", "events", "station", "matter", "seriously", "try", "survey", "questions",
    "question",
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


def generate_ranked_wordcloud_bars(
        df,
        text_column,
        rank_column,
        rank_categories,
        main_title="",
        subtitle="",
        source_text="",
        all_stops=None,
        synonyms_dict=None,
        highlight_words=None,
        output_filename=None
):
    def remove_punctuation_and_stopwords(text):
        """Remove punctuation and stopwords, handling contractions carefully"""
        if not isinstance(text, str):
            return ""
        text = text.lower()

        # Pre-process common contractions before removing punctuation
        text = re.sub(r"i'm|i've|it's|that's|there's|what's|can't|won't|isn't|aren't|here's|let's", "", text)
        text = re.sub(r"'s\b", "", text)  # Remove possessive 's

        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # Split and filter words
        words = [word.strip() for word in text.split()]
        if all_stops:
            words = [w for w in words if w and w not in all_stops]

        return " ".join(words)

    def get_exact_word_frequencies(text, n_words):
        """Get exactly n_words most frequent words with controlled frequencies"""
        words = text.split()
        if not words:
            return {}

        # Get raw frequencies
        word_freq = Counter(words)

        # Get top n words
        n = min(n_words, len(word_freq))
        top_words = word_freq.most_common(n)

        if not top_words:
            return {}

        # Create artificial frequencies that decrease linearly
        # This ensures WordCloud will show all words
        return {word: 100 - (i * (90 / n)) for i, (word, _) in enumerate(top_words)}

    # Set up the figure
    fig, ax = plt.subplots(figsize=(24, 16), facecolor='white')

    # Calculate proportions
    total_responses = len(df)
    value_counts = df[rank_column].value_counts()
    proportions = value_counts / total_responses
    proportions = proportions.reindex(rank_categories)
    max_proportion = max(proportions)

    # Setup visualization parameters
    y_pos = np.arange(len(rank_categories))
    bar_height = 0.8
    min_proportion_for_wordcloud = 0.01
    viridis = colormaps['viridis']
    colors = [viridis(1 - i / (len(rank_categories) - 1)) for i in range(len(rank_categories))]

    # Create base bars
    bars = ax.barh(y_pos, proportions, height=bar_height,
                   color=colors, alpha=0.4, edgecolor='black', linewidth=3)

    # Process each category
    for idx, (category, proportion) in enumerate(proportions.items()):
        if pd.isna(proportion):
            continue

        category_df = df[df[rank_column] == category].copy()

        if not category_df.empty and proportion >= min_proportion_for_wordcloud:
            # Clean and combine all texts
            category_df['clean_text'] = category_df[text_column].apply(remove_punctuation_and_stopwords)
            clean_text = ' '.join(category_df['clean_text'])

            # Calculate desired number of words based on proportion
            # Reduced from 200 to 100 to show fewer words
            desired_words = max(round(proportion * 100), 1)

            # Get word frequencies with controlled distribution
            word_frequencies = get_exact_word_frequencies(clean_text, desired_words)

            print(f"\nCategory: {category}")
            print(f"Proportion: {proportion:.1%}")
            print(f"Desired words: {desired_words}")
            print(f"Actual words: {len(word_frequencies)}")

            if word_frequencies:
                try:
                    # Force WordCloud to use our exact frequencies
                    wc = WordCloud(
                        width=int(1800 * proportion / max_proportion),
                        height=250,
                        background_color=None,
                        mode='RGBA',
                        color_func=lambda *args, **kwargs: 'black',
                        max_words=len(word_frequencies),
                        min_font_size=18,  # Match category label size
                        max_font_size=32,
                        prefer_horizontal=0.9,
                        relative_scaling=0.6,
                        collocations=False,
                        scale=2,
                        repeat=False,
                        normalize_plurals=False,
                        margin=1
                    ).generate_from_frequencies(word_frequencies)

                    # Position word cloud
                    horizontal_padding = 0.03
                    vertical_padding = 0.1
                    bar_center = y_pos[idx]
                    bar_width = proportion

                    cloud_extent = [
                        horizontal_padding * bar_width,
                        bar_width * (1 - horizontal_padding),
                        bar_center - (bar_height / 2 * (1 - vertical_padding)),
                        bar_center + (bar_height / 2 * (1 - vertical_padding))
                    ]

                    ax.imshow(wc, extent=cloud_extent, aspect='auto', zorder=2)

                except ValueError as e:
                    print(f"Could not generate word cloud for {category}: {e}")

        # Add percentage label
        ax.text(proportion + 0.02, y_pos[idx],
                f"{proportion:.1%}",
                va='center', fontsize=24, fontweight='bold')

    # [Rest of the visualization code remains the same...]
    ax.set_xlim(0, max_proportion * 1.15)
    ax.set_ylim(-0.5, len(rank_categories) - 0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax.tick_params(axis='x', labelsize=16, width=2)
    ax.set_xlabel('Proportion of Responses', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rank_categories, fontsize=18, fontweight='bold')

    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add titles and source
    fig.text(0.2, 0.95, main_title, fontsize=24, fontweight='bold', ha='left')
    fig.text(0.2, 0.91, subtitle, fontsize=20, color='#404040', ha='left')
    source_text_full = (
        f"Source: {source_text}\n"
        f"n = {total_responses:,} responses\n"
        "Note: Bar width shows the proportion of responses in each category. "
        "Word clouds show the most common words in open-ended responses, "
        "with word size indicating frequency and number of words scaled by category size."
    )
    fig.text(0.2, 0.02, source_text_full, fontsize=14, color='#404040', ha='left')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    return fig

if __name__ == "__main__":
    # Here we'll recreate several of the visualizations from your script
    # using the new proportional bar + wordcloud approach

    # Set up paths as in original script
    import os

    current_dir = os.getcwd()
    qwels_path = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(qwels_path, "QWELS", "ANALYSIS", "DATASETS")
    text_path = os.path.join(data_path, "textsets")

    # Create visuals folder if it doesn't exist
    visuals_path = os.path.join(current_dir, "visuals")
    if not os.path.exists(visuals_path):
        os.mkdir(visuals_path)

    # Example 1: Job Fun Analysis
    df_jobfun = pd.read_excel(os.path.join(text_path, "APP_FILE.xlsx"),
                              sheet_name="messi_us_2024")

    fig1 = generate_ranked_wordcloud_bars(
        df=df_jobfun,
        text_column='jobfun_open',
        rank_column='jobfun',
        rank_categories=['often', 'sometimes', 'rarely', 'never'],
        main_title='What is Fun About Work?',
        subtitle='How often would you say your job is fun? Please briefly tell us why you answered that way.',
        source_text='2024 Measuring Economic Sentiment and Social Inequality Survey (MESSI-US)',
        all_stops=all_stops,
        output_filename=os.path.join(visuals_path, 'jobfun_ranked_wordcloud.png')
    )

    # Example 2: Skills Improvement Analysis
    df_improve = pd.read_excel(os.path.join(text_path, "MESSI_CAN_2024_MAY.xlsx"),
                               sheet_name="WHOLE")

    fig2 = generate_ranked_wordcloud_bars(
        df=df_improve,
        text_column='improve_open',
        rank_column='improve',
        rank_categories=[' strongly agree', 'agree', 'disagree', 'strongly disagree'],
        main_title='The Chance to Improve',
        subtitle='My job gives me a chance to improve my skills.',
        source_text='2024 Measuring Economic Sentiment and Social Inequality Survey (MESSI-CAN)',
        all_stops=all_stops,
        output_filename=os.path.join(visuals_path, 'improve_ranked_wordcloud.png')
    )

    # Example 3: Work Enjoyment Without Money
    df_enjoy = pd.read_excel(os.path.join(text_path, "QES_US_2023.xlsx"),
                             sheet_name="WKENJOY")

    fig3 = generate_ranked_wordcloud_bars(
        df=df_enjoy,
        text_column='wrkenjoy_open',
        rank_column='wrkenjoy',
        rank_categories=['strongly agree', 'agree', 'disagree', 'strongly disagree'],
        main_title='Is Enjoyment of Work Possible Without Money?',
        subtitle='I would enjoy having a paid job even if I didnt need the money.',
        source_text = '2023 Quality of Employment Survey (QES-US 2023)',
        all_stops = all_stops,
        output_filename = os.path.join(visuals_path, 'wrkenjoy_ranked_wordcloud.png')
    )

    # Example 4: Job Meaning Analysis
    df_meaning = pd.read_excel(os.path.join(text_path, "TRENDS_CAN_2023_SEP.xlsx"),
                               sheet_name="ALL")

    fig4 = generate_ranked_wordcloud_bars(
        df=df_meaning,
        text_column='meaning_open',
        rank_column='meaning',
        rank_categories=['strongly agree', 'agree', 'disagree', 'strongly disagree'],
        main_title='Work Meaning and its Opposite',
        subtitle='The work I do on my job is meaningful to me.',
        source_text='2023 Trends Canada (TRENDS-CAN 2023)',
        all_stops=all_stops,
        output_filename=os.path.join(visuals_path, 'meaning_ranked_wordcloud.png')
    )

    # Example 5: Personal Interest Analysis
    df_interest = pd.read_excel(os.path.join(text_path, "MESSI_CAN_2024_MAY.xlsx"),
                                sheet_name="WHOLE")

    fig5 = generate_ranked_wordcloud_bars(
        df=df_interest,
        text_column='interest_open',
        rank_column='personal_interest',
        rank_categories=['very true', 'somewhat true', 'not too true', 'not at all true'],
        main_title='When Workers Take an Interest',
        subtitle='Coworkers take a personal interest in you.',
        source_text='2024 Measuring Economic Sentiments and Social Inequality (MESSI-CAN 2024)',
        all_stops=all_stops,
        output_filename=os.path.join(visuals_path, 'personal_interest_ranked_wordcloud.png')
    )

    plt.show()