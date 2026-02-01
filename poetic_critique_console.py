import torch
import nltk
import textstat
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, set_seed, AutoTokenizer
import time
import os

# --- 1. CONFIGURATION ---
# Using Gemma-2B-it for the best speed/quality compromise on CPU.
MODEL_NAME = "google/gemma-2b-it"
INITIAL_PROMPT = (
    "You are a melancholic, nature-focused Romantic poet from Romania, "
    "like Mihai Eminescu. Write the next four-line stanza of a poem. "
)

# Critic AI Thresholds (Tuned for Romantic Melancholy)
ROMANTIC_SYMBOLS = {
    'moon', 'star', 'lake', 'willow', 'shadow', 'grief', 'soul', 'tear',
    'night', 'dream', 'heart', 'pale', 'silent', 'void', 'cold'
}
SYMBOL_RECURRENCE_THRESHOLD = 0.15
SENTIMENT_STAGNATION_THRESHOLD = 0.15
MAX_MELANCHOLY_THRESHOLD = -0.50
MAX_READABILITY_GRADE = 7.0
MAX_HISTORY_LENGTH = 1000

# --- 2. INITIALIZATION ---
set_seed(42)
analyzer = SentimentIntensityAnalyzer()
device = -1
print(f"Initializing Generative AI on CPU with model: {MODEL_NAME}...")

try:
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=device,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    )
except Exception as e:
    print(f"FATAL ERROR LOADING GEMMA: Check authentication/license. Error: {e}")
    exit()


# --- 3. CRITIC AI FUNCTION (Phase 2) ---

def get_critic_feedback(new_segment: str, recent_segments_data: list) -> tuple:
    """Analyzes a new poem segment against three Romantic Flaws."""
    constraints = []
    metrics = {}

    # 1. Calculate Sentiment and Readability
    scores = analyzer.polarity_scores(new_segment)
    new_compound_sentiment = scores['compound']
    metrics['sentiment_compound'] = new_compound_sentiment
    metrics['readability_grade'] = textstat.flesch_kincaid_grade(new_segment)

    # --- Flaw 3: Flowery Verbosity ---
    if metrics['readability_grade'] > MAX_READABILITY_GRADE:
        constraints.append("C_use_direct_metaphors_and_short_sentences")

    # --- Flaw 2: Melancholy Monotony & Extreme Emotion FIX ---
    if recent_segments_data:
        previous_compound_sentiments = [data[1] for data in recent_segments_data]
        avg_sentiment = sum(previous_compound_sentiments) / len(previous_compound_sentiments)
        sentiment_change = abs(new_compound_sentiment - avg_sentiment)
        metrics['sentiment_change'] = sentiment_change
        metrics['avg_prev_sentiment'] = avg_sentiment

        # Check A: Too little change (stagnation)
        if sentiment_change < SENTIMENT_STAGNATION_THRESHOLD:
            constraints.append("C_dramatically_shift_sentiment")

        # Check B: TOO EXTREME (The new check for the soft critic)
        # If the poem is excessively dark, force a shift toward hope or conflict.
        if new_compound_sentiment < MAX_MELANCHOLY_THRESHOLD:
            constraints.append("C_introduce_dramatic_conflict_or_hope")

    # --- Flaw 1: Symbolic Stagnation ---
    tokens = [w.lower() for w in word_tokenize(new_segment) if w.isalpha()]
    symbol_tokens = [t for t in tokens if t in ROMANTIC_SYMBOLS]

    if len(tokens) > 0:
        symbol_ratio = len(symbol_tokens) / len(tokens)
        metrics['symbol_ratio'] = symbol_ratio

        if symbol_ratio > SYMBOL_RECURRENCE_THRESHOLD:
            constraints.append("C_introduce_new_cosmic_symbol")

    return constraints, metrics


# --- 4. GENERATIVE AI FUNCTION (Phase 1) ---

def generate_poem_segment(current_poem: str, constraints: list) -> str:
    """Generates the next stanza based on the poem history and critic constraints."""

    constraint_str = ", ".join(constraints) if constraints else "Continue normally."

    last_stanza = current_poem.strip().split('\n')[-4:] if current_poem.strip() else ["(The poem begins now.)"]
    context = "\n".join(last_stanza)

    full_prompt = (
        f"{INITIAL_PROMPT} "
        f"The previous lines were: '{context}' "
        f"Your instruction: '{constraint_str}'. "
        "Write ONLY the next 4-line stanza and nothing else."
    )

    # Generate text
    output = generator(
        full_prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        return_full_text=False
    )

    raw_text = output[0]['generated_text'].strip()
    stanza = raw_text.split('\n\n')[0].strip()
    stanza = stanza.replace(INITIAL_PROMPT, '').strip()

    if stanza.startswith("<start_of_turn>"):
        stanza = stanza.split("<start_of_turn>")[1].strip()
    return stanza


# --- 5. MAIN LOOP (Phase 3) ---

def main_installation_loop():
    print("--- THE SELF-CORRECTING LYRICAL NARRATIVE ---")
    print(f"Model: {MODEL_NAME} | Running on: CPU\n")
    current_poem = ""
    history_data = []
    current_constraints = []

    # Start the loop
    for i in range(1, 100):
        print(f"\n======== GENERATION {i} ========")
        new_segment = generate_poem_segment(current_poem, current_constraints)

        if not new_segment or len(new_segment.split()) < 5:
            print("   (Model failed to produce a valid stanza. Substituting fallback text.)")
            new_segment = "A silent shadow where no thought resides,\nThe empty canvas where the moon presides."

        current_poem += "\n" + new_segment
        print("\n[POEM NARRATIVE]:")
        print(f"  {new_segment.replace('\n', '\n  ')}")

        temp_scores = analyzer.polarity_scores(new_segment)
        new_sentiment = temp_scores['compound']

        history_data.append((new_segment, new_sentiment))
        recent_history = history_data[-5:]

        new_constraints, metrics = get_critic_feedback(new_segment, recent_history[:-1])

        print("\n--- CRITIC AI ANALYSIS ---")
        print(
            f"Sentiment: {metrics.get('sentiment_compound'):.2f} (Avg Prev: {metrics.get('avg_prev_sentiment', 0):.2f})")
        print(f"Readability: Grade {metrics.get('readability_grade'):.1f}")
        print(f"Symbol Ratio: {metrics.get('symbol_ratio', 0):.2f}")

        if new_constraints:
            print("\n[!] CRITIC DEMANDS REVISION:")
            for c in new_constraints:
                print(f"   => {c}")
        else:
            print("\n[OK] CRITIC APPROVES: No constraints added.")

        current_constraints = new_constraints

        time.sleep(10)


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main_installation_loop()