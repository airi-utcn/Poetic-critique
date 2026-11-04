from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import textstat
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer
import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: No HF_TOKEN found in environment!")

app = Flask(__name__)
CORS(app)

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

MODEL_NAME = "google/gemma-2b-it"
INITIAL_PROMPT = (
    "You are a melancholic, nature-focused Romantic poet from Romania, "
    "like Mihai Eminescu. Write the next four-line stanza of a poem. "
)

ROMANTIC_SYMBOLS = {
    'moon', 'star', 'lake', 'willow', 'shadow', 'grief', 'soul', 'tear',
    'night', 'dream', 'heart', 'pale', 'silent', 'void', 'cold'
}

analyzer = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=MODEL_NAME, tokenizer=tokenizer, device=-1)


@app.route('/compute', methods=['POST'])
def compute():
    data = request.json or {}
    context = data.get('context', '')
    constraints = data.get('constraints', [])

    constraint_str = ", ".join(constraints) if constraints else "Continue normally."
    last_stanza = context.strip().split('\n')[-4:] if context.strip() else ["(The poem begins now.)"]
    sub_context = "\n".join(last_stanza)

    full_prompt = (
        f"{INITIAL_PROMPT} "
        f"The previous lines were: '{sub_context}' "
        f"Your instruction: '{constraint_str}'. "
        "Write ONLY the next 4-line stanza and nothing else."
    )

    # return_full_text=False natively strips the prompt from the output
    output = generator(full_prompt, max_new_tokens=80, do_sample=True, temperature=1.0, top_k=50, return_full_text=False)
    stanza = output[0]['generated_text'].strip()

    vader_scores = analyzer.polarity_scores(stanza)
    compound = float(vader_scores['compound'])
    readability = float(textstat.flesch_kincaid_grade(stanza))

    tokens = [w.lower() for w in word_tokenize(stanza) if w.isalpha()]
    symbol_tokens = [t for t in tokens if t in ROMANTIC_SYMBOLS]
    symbol_ratio = float(len(symbol_tokens) / len(tokens)) if tokens else 0.0

    return jsonify({
        "stanza": stanza,
        "sentiment_compound": compound,
        "readability_grade": readability,
        "symbol_ratio": symbol_ratio
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)