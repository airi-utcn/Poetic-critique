from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import textstat
import torch
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)
CORS(app)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EMINESCU_ADAPTER_PATH = "./eminescu_adapter_qwen"
YUAN_ADAPTER_PATH = "./yuan_adapter_qwen"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

ROMANTIC_LEXICON = {
    'always', 'around', 'away', 'big', 'blue', 'bright', 'cold', 'dark', 'day', 'dear',
    'death', 'deep', 'dream', 'dreams', 'eyes', 'face', 'feel', 'full', 'gently', 'give',
    'good', 'hair', 'hand', 'head', 'hear', 'heart', 'let', 'life', 'light', 'long',
    'look', 'love', 'lovely', 'many', 'mind', 'moon', 'much', 'nice', 'night', 'old',
    'pain', 'people', 'sea', 'shadow', 'shining', 'sky', 'small', 'soul', 'star', 'stars',
    'sweet', 'things', 'time', 'vain', 'want', 'waves', 'white', 'without', 'world', 'yet'
}

GONGAN_LEXICON = {
    'alone', 'autumn', 'bamboo', 'birds', 'boat', 'book', 'books', 'breeze', 'bridge', 'brush',
    'calm', 'clouds', 'cold', 'crane', 'dawn', 'drink', 'drunk', 'dust', 'east', 'empty',
    'evening', 'flowers', 'forest', 'friend', 'friends', 'garden', 'gate', 'green', 'guest', 'hills',
    'home', 'idle', 'ink', 'inn', 'jade', 'lake', 'lamp', 'laugh', 'leaves', 'leisure',
    'light', 'lone', 'monk', 'moon', 'mountain', 'night', 'old', 'path', 'pavilion', 'pine',
    'rain', 'river', 'road', 'sleep', 'snow', 'spring', 'stone', 'stream', 'tea', 'wind', 'wine'
}

EMINESCU_CONSTRAINT_MAP = {
    "C_use_direct_metaphors": (
        "Temper syntactic complexity; favor lucid, paratactic lines and direct, unadorned metaphors over labyrinthine clauses."
    ),
    "C_deepen_syntax": (
        "Elevate the prosodic architecture through periodic syntax, layered subordinate clauses, and resonant literary diction."
    ),
    "C_dramatically_shift_sentiment": (
        "Execute a decisive emotional volta, pivoting sharply away from the prevailing affective register of the preceding stanza."
    ),
    "C_introduce_hope": (
        "Infuse the stanza with luminous transcendence, consolation, and upward spiritual yearning."
    ),
    "C_introduce_melancholy": (
        "Temper the brightness with elegiac gravity, mortal transience, and profound Romantic longing."
    ),
    "C_introduce_new_cosmic_symbol": (
        "Exercise imagistic restraint; pare back dense symbolic repetition in favor of spare, disciplined poetic focus."
    ),
    "C_enrich_romantic_imagery": (
        "Anchor the verse in archetypal Romantic emblems—the moon, stars, shadow, night, sea, waves, and the yearning soul."
    )
}

YUAN_CONSTRAINT_MAP = {
    "C_use_direct_metaphors": (
        "Strip away heavy rhetorical ornament and tangled hypotaxis; write with unforced poetic lucidity and crisp, uncluttered lines."
    ),
    "C_deepen_syntax": (
        "Enrich the poetic cadence with poised, layered phrasing and reflective depth while preserving natural lyrical grace."
    ),
    "C_dramatically_shift_sentiment": (
        "Introduce a pronounced tonal turn, pivoting between serene detachment, wry philosophical wit, and sudden lyrical delight."
    ),
    "C_introduce_hope": (
        "Illuminate the verse with buoyant clarity, lighthearted ease, and unburdened delight in the immediate world."
    ),
    "C_introduce_melancholy": (
        "Subdue the exuberance with quiet, contemplative stillness, solitude, and subtle autumnal wistfulness."
    ),
    "C_introduce_new_cosmic_symbol": (
        "Practice classical restraint; pare the imagery down to two or three spare, evocative details rather than crowding the line with nouns."
    ),
    "C_enrich_romantic_imagery": (
        "Ground the lyric in concrete Gong'an motifs—bamboo, clouds, cold, jade, river, wine, monk, inn, or pavilion."
    )
}

analyzer = SentimentIntensityAnalyzer()

print(f"Loading Base Model onto {DEVICE.upper()} ({DTYPE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    low_cpu_mem_usage=True
).to(DEVICE)

print("Attaching Eminescu and Yuan Hongdao LoRA Adapters...")
model = PeftModel.from_pretrained(base_model, EMINESCU_ADAPTER_PATH, adapter_name="eminescu")
model.load_adapter(YUAN_ADAPTER_PATH, adapter_name="yuan")
model.set_adapter("eminescu")
model.to(DEVICE)
model.eval()
print(f"AI Engine Ready on {DEVICE.upper()}.")


@app.route('/compute', methods=['POST'])
def compute():
    data = request.json or {}
    context = data.get('context', '')
    poet_mode = data.get('poetMode', 'eminescu')
    if poet_mode not in ('eminescu', 'yuan'):
        poet_mode = 'eminescu'

    model.set_adapter(poet_mode)

    raw_constraints = data.get('constraints', [])
    if isinstance(raw_constraints, str):
        if raw_constraints == "Continue normally." or not raw_constraints.strip():
            constraints = []
        else:
            constraints = [c.strip() for c in raw_constraints.split(",") if c.strip()]
    else:
        constraints = raw_constraints or []

    active_map = YUAN_CONSTRAINT_MAP if poet_mode == "yuan" else EMINESCU_CONSTRAINT_MAP

    if not constraints:
        natural_instruction = (
            "Continue the poem with unforced Xingling sensibility, lyrical clarity, and vivid sensory observation."
            if poet_mode == "yuan"
            else "Continue the poem naturally in an elegiac, nature-infused Romantic register."
        )
    else:
        translated_phrases = [active_map.get(tag, tag) for tag in constraints]
        natural_instruction = "CRITIC'S FEEDBACK: " + " ".join(translated_phrases)

    clean_lines = [line.strip() for line in context.strip().split('\n') if line.strip()]
    last_stanza = clean_lines[-4:] if clean_lines else ["(The poem begins now.)"]
    sub_context = "\n".join(last_stanza)

    if poet_mode == "yuan":
        persona_prompt = (
            "You are Yuan Hongdao, a late Ming dynasty Gong'an school poet composing verse with spontaneous Xingling "
            "(innate poetic sensibility), candid wit, and unforced lyrical clarity. Write the next four-line stanza of a poem."
        )
    else:
        persona_prompt = (
            "You are a melancholic, nature-focused Romantic poet from Romania, in the tradition of Mihai Eminescu. "
            "Write the next four-line stanza of a poem."
        )

    full_prompt = (
        "<|im_start|>user\n"
        f"{persona_prompt}\n"
        f"The previous lines were:\n'{sub_context}'\n\n"
        f"Instruction: {natural_instruction}\nWrite ONLY the next 4-line stanza and nothing else.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE)

    terminators = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and isinstance(im_end_id, int):
        terminators.append(im_end_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=110,
            do_sample=True,
            temperature=0.80,
            top_k=50,
            top_p=0.90,
            repetition_penalty=1.18,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
    final_lines = lines[:4]

    if len(final_lines) == 4:
        last_line = final_lines[3]
        if not last_line.endswith(('.', '!', '?', '—', '–', '...', '“', '”', '"')):
            final_lines[3] = last_line.rstrip(",;") + "."

    stanza = "\n".join(final_lines)

    vader_scores = analyzer.polarity_scores(stanza)
    compound = float(vader_scores['compound'])
    readability = float(textstat.flesch_kincaid_grade(stanza))

    active_lexicon = GONGAN_LEXICON if poet_mode == "yuan" else ROMANTIC_LEXICON
    tokens = [w.lower() for w in word_tokenize(stanza) if w.isalpha()]
    symbol_tokens = [t for t in tokens if t in active_lexicon]
    symbol_ratio = float(len(symbol_tokens) / len(tokens)) if tokens else 0.0

    return jsonify({
        "stanza": stanza,
        "sentiment_compound": compound,
        "readability_grade": readability,
        "symbol_ratio": symbol_ratio
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)