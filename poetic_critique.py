import torch
import nltk
import textstat
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, set_seed, AutoTokenizer
import time
import os
import streamlit as st
import base64
import random

# --- 1. CONFIGURATION ---
MODEL_NAME = "google/gemma-2b-it"
INITIAL_PROMPT = (
    "You are a melancholic, nature-focused Romantic poet from Romania, "
    "like Mihai Eminescu. Write the next four-line stanza of a poem. "
)

ROMANTIC_SYMBOLS = {
    'moon', 'star', 'lake', 'willow', 'shadow', 'grief', 'soul', 'tear',
    'night', 'dream', 'heart', 'pale', 'silent', 'void', 'cold'
}
SYMBOL_RECURRENCE_THRESHOLD = 0.15
MAX_MELANCHOLY_THRESHOLD = -0.50
MAX_READABILITY_GRADE = 8.0
SENTIMENT_STAGNATION_THRESHOLD = 0.10

CRITIC_MESSAGES = {
    "C_use_direct_metaphors": "The prose is tangled...",
    "C_dramatically_shift_sentiment": "The river flows too evenly...",
    "C_introduce_hope": "The darkness is absolute...",
    "C_introduce_new_cosmic_symbol": "The stars are worn thin..."
}

# --- 2. ASSET LOADING ---
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Assets
LETTER_IMG_PATH = "C:/Users/BiancaS/Documents/ai/poems i’ll never send..png"
PAPER_BG_PATH = "C:/Users/BiancaS/.gemini/antigravity/brain/f6f39339-a894-4801-ac07-ca6990199784/kissed_paper_texture_1768053228973.png"
# Updated to user's local file
KISS_MARK_PATH = "C:/Users/BiancaS/Documents/ai/kiss.png"
BEST_POEMS_FILE = "C:/Users/BiancaS/Documents/ai/best.txt"

img_letter = get_base64_image(LETTER_IMG_PATH)
img_paper = get_base64_image(PAPER_BG_PATH)
img_kiss = get_base64_image(KISS_MARK_PATH)

# --- 3. INITIALIZATION ---
@st.cache_resource
def initialize_models():
    analyzer = SentimentIntensityAnalyzer()
    device = -1 
    try:
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=device,
            tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        )
    except Exception as e:
        generator = None
    return generator, analyzer

if 'app_state' not in st.session_state:
    st.session_state.app_state = 'landing'
if 'history_data' not in st.session_state:
    st.session_state.history_data = [] 
if 'last_constraints' not in st.session_state:
    st.session_state.last_constraints = []
if 'poem_full_text' not in st.session_state:
    st.session_state.poem_full_text = "" 

# --- 4. CORE LOGIC ---
def get_critic_feedback(new_segment, recent_segments_data, analyzer):
    constraints = []
    metrics = {}
    
    scores = analyzer.polarity_scores(new_segment)
    new_compound_sentiment = scores['compound']
    metrics['sentiment_compound'] = new_compound_sentiment
    metrics['readability_grade'] = textstat.flesch_kincaid_grade(new_segment)

    if metrics['readability_grade'] > MAX_READABILITY_GRADE:
        constraints.append("C_use_direct_metaphors")

    if recent_segments_data:
        previous_compound_sentiments = [data[1] for data in recent_segments_data]
        avg_sentiment = sum(previous_compound_sentiments) / len(previous_compound_sentiments)
        sentiment_change = abs(new_compound_sentiment - avg_sentiment)
        metrics['avg_prev_sentiment'] = avg_sentiment

        if sentiment_change < SENTIMENT_STAGNATION_THRESHOLD:
            constraints.append("C_dramatically_shift_sentiment")
        if new_compound_sentiment < MAX_MELANCHOLY_THRESHOLD:
            constraints.append("C_introduce_hope")

    tokens = [w.lower() for w in word_tokenize(new_segment) if w.isalpha()]
    symbol_tokens = [t for t in tokens if t in ROMANTIC_SYMBOLS]
    if len(tokens) > 0:
        symbol_ratio = len(symbol_tokens) / len(tokens)
        metrics['symbol_ratio'] = symbol_ratio
        if symbol_ratio > SYMBOL_RECURRENCE_THRESHOLD:
            constraints.append("C_introduce_new_cosmic_symbol")

    return constraints, metrics

def generate_poem_segment(current_poem_context, constraints, generator):
    constraint_str = ", ".join(constraints) if constraints else "Continue normally."
    last_stanza = current_poem_context.strip().split('\n')[-4:] if current_poem_context.strip() else ["(The poem begins now.)"]
    context = "\n".join(last_stanza)

    full_prompt = (
        f"{INITIAL_PROMPT} "
        f"The previous lines were: '{context}' "
        f"Your instruction: '{constraint_str}'. "
        "Write ONLY the next 4-line stanza and nothing else."
    )

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

def render_frame(placeholder_poem, placeholder_critique, stanzas, metrics, constraints, kiss_img):
    # RENDER HTML
    poem_html = '<div class="poem-content-area">'
    
    if stanzas:
        for i, (stanza, _) in enumerate(reversed(stanzas)):
            is_newest = (i == 0)
            css_class = "stanza newest-stanza" if is_newest else "stanza"
            
            # Randomized but Left-Aligned Kisses
            kisses_html = ""
            num_kisses = random.choice([0, 1, 1, 2])
            for _ in range(num_kisses):
                top = random.randint(-10, 60)
                # Position on the LEFT side of the poetry block
                left = random.randint(-25, 5) 
                rot = random.randint(-45, 10)
                kisses_html += f'<img src="data:image/png;base64,{kiss_img}" class="kiss-mark" style="top:{top}%; left:{left}%; transform:rotate({rot}deg);">'

            poem_html += f'<div class="{css_class}">{kisses_html}{stanza.replace(chr(10), "<br>")}</div>'
    
    poem_html += '</div>'
    
    with placeholder_poem.container():
        st.markdown(poem_html, unsafe_allow_html=True)
        
    with placeholder_critique.container():
        sent_val = metrics.get('sentiment_compound', 0.0)
        read_val = metrics.get('readability_grade', 0.0)
        
        critique_html = f"""
        <div class="critique-overlay">
            <h3 style="color:#8b0000; margin-top:0;">Critic's Note</h3>
            <strong>Sentiment:</strong> {sent_val:.2f}<br>
            <strong>Readability:</strong> {read_val:.1f}<br>
            <hr style="border:1px dashed #ccc;">
            <em style="color:#555;">"{'<br>'.join([CRITIC_MESSAGES[c] for c in constraints]) if constraints else 'Eminescu would weep.'}"</em>
        </div>
        """
        st.markdown(critique_html, unsafe_allow_html=True)

# --- 5. UI IMPLEMENTATION ---
st.set_page_config(layout="wide", page_title="Letters I'll Never Send")

# CSS INJECTION
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

/* GLOBAL BACKGROUND COLOR - DARK THEME (#220902) */
.stApp {{
    background-color: #220902; 
}}

/* === LANDING PAGE === */
.landing-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: 80vh;
    flex-direction: column;
}}

.letter-image-container {{
    margin-bottom: 30px;
    display: flex;
    justify-content: center;
    width: 400px; /* RESIZED SMALLER */
    max-width: 100%;
}}

.letter-img {{
    width: 100%;
    height: auto;
    /* SHADOW REMOVED as requested to blend exactly */
    /* filter: drop-shadow(0 0 15px rgba(0,0,0,0.5)); */
    border-radius: 0;
}}

/* === MAIN POEM APP STYLES === */
.letter-background {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background-image: url("data:image/png;base64,{img_paper}");
    background-size: cover;
    background-position: center;
    z-index: 0; 
}}

.poem-content-area {{
    width: 60%;
    margin: 0 auto;
    padding-top: 100px;
    padding-bottom: 50px;
    font-family: 'Dancing Script', cursive;
    font-size: 28px;
    line-height: 1.6;
    color: #4a0404; 
    position: relative; 
    text-shadow: 0 0 1px rgba(0,0,0,0.1);
    z-index: 1;
}}

.stanza {{
    margin-bottom: 40px;
    position: relative;
    padding: 20px;
}}

.newest-stanza {{
    font-weight: bold;
    color: #6a0404;
    transform: scale(1.02);
}}

/* Scattered Kisses */
.kiss-mark {{
    position: absolute;
    width: 60px; /* Slightly smaller to fit margin */
    height: auto;
    opacity: 0.7;
    z-index: -1;
    pointer-events: none;
}}

/* Sticky Note Critique */
.critique-overlay {{
    position: fixed;
    right: 80px;
    top: 200px;
    width: 220px;
    background: #fffef0;
    padding: 20px;
    border: 1px solid #dcd0c0;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    transform: rotate(3deg);
    font-family: 'Playfair Display', serif;
    font-size: 14px;
    color: #333;
    z-index: 100;
}}

/* BUTTON STYLING */
div.stButton > button {{
    background-color: #4a0404;
    color: #e0d0b8;
    font-family: 'Playfair Display';
    border: 1px solid #5a2e2e;
    border-radius: 5px;
    padding: 10px 25px;
    font-size: 18px;
    margin: 0 auto;
    display: block;
}}
div.stButton > button:hover {{
    background-color: #6a0404;
    color: white;
}}

/* Hide Sidebar elements partially */
[data-testid="stSidebar"] {{
    background-color: #1a0500;
    color: #e0d0b8;
}}
</style>
""", unsafe_allow_html=True)

# --- ROUTING ---

if st.session_state.app_state == 'landing':
    
    # 1. SPECIAL CSS FOR LANDING BUTTON (Centered & Styled)
    st.markdown("""
    <style>
    /* Force centering of the button container and button itself */
    .stMain .block-container div[data-testid="stVerticalBlock"] > div.stButton {
        display: flex;
        justify-content: center;
        width: 100%;
        text-align: center;
    }
    
    .stMain div.stButton > button {
        background-color: #e5e1d9 !important;
        color: black !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 16px !important; 
        padding: 8px 24px !important;
        border: none !important;
        border-radius: 5px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        margin-top: 10px; 
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    .stMain div.stButton > button:hover {
        background-color: #d4d0c8 !important;
        color: black !important;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

    # LANDING PAGE CONTENT
    st.markdown("""
        <style>
        .block-container {
            padding-top: 1rem !important; 
        }
        
        /* STRICT CENTERING FOR MIDDLE COLUMN */
        /* Target the vertical block inside the specific column structure */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-of-type(2) [data-testid="stVerticalBlock"] {
            align-items: center !important;
            display: flex !important;
            flex-direction: column !important;
        }

        /* Standardize Button Style */
        .stMain div.stButton > button {
            background-color: #e5e1d9 !important;
            color: black !important;
            font-family: 'Playfair Display', serif !important;
            font-size: 16px !important; 
            padding: 8px 24px !important;
            border: none !important;
            border-radius: 5px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            margin-top: 10px;
        }
        
        .stMain div.stButton > button:hover {
            background-color: #d4d0c8 !important;
            color: black !important;
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; width: 100%; margin-top: 5vh; margin-bottom: 5px;">
            <div class="letter-image-container" style="width: 400px; max-width:100%;">
                <img src="data:image/png;base64,{img_letter}" class="letter-img">
            </div>
        </div>
        """, unsafe_allow_html=True)
        

        if st.button("open the collection"):
            st.session_state.app_state = 'letter'
            st.rerun()

elif st.session_state.app_state == 'letter':
    
    # MAIN POEM VIEW
    st.markdown("""
    <style>
    .stApp {
        background-color: #e5e1d9 !important;
        background-image: none !important;
    }
    
    /* Global kill-switch for animations/transitions */
    * {
        transition: none !important;
        animation: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    generator, analyzer = initialize_models()
    
    with st.sidebar:
        st.title("Ink Flow")
        speed = st.slider("Writing Speed (s)", 5, 60, 15)
        st.markdown("---")
        if st.button("Stats Explained"):
            st.session_state.app_state = 'stats'
            st.rerun()
        if st.button("The Volume"):
            st.session_state.app_state = 'volume'
            st.rerun()
        if st.button("Close Letter"):
            st.session_state.app_state = 'landing'
            st.rerun()
            
    placeholder_poem = st.empty()
    placeholder_critique = st.empty()
    
    # --- INITIAL RENDER (Fix for Blank Page) ---
    if st.session_state.history_data:
        last_seg = st.session_state.history_data[-1][0]
        temps = analyzer.polarity_scores(last_seg)
        init_metrics = {
            'sentiment_compound': temps['compound'],
            'readability_grade': textstat.flesch_kincaid_grade(last_seg)
        }
        render_frame(
            placeholder_poem, 
            placeholder_critique, 
            st.session_state.history_data, 
            init_metrics, 
            st.session_state.last_constraints, 
            img_kiss
        )

    while True:
        # Generate
        new_segment = generate_poem_segment(
            st.session_state.poem_full_text,
            st.session_state.last_constraints,
            generator
        )
        if not new_segment: new_segment = "..."

        temp_scores = analyzer.polarity_scores(new_segment)
        new_sentiment = temp_scores['compound']
        
        st.session_state.history_data.append((new_segment, new_sentiment))
        st.session_state.poem_full_text += "\n\n" + new_segment
        
        recent_history = st.session_state.history_data[-5:]
        new_constraints, metrics = get_critic_feedback(new_segment, recent_history[:-1], analyzer)
        st.session_state.last_constraints = new_constraints
        
        # AUTO-SAVE PERFECT POEMS
        # Filter out placeholder "..." and very short failures
        if not new_constraints and len(new_segment) > 10 and "..." not in new_segment:
            try:
                with open(BEST_POEMS_FILE, "a", encoding="utf-8") as f:
                    f.write(new_segment + "\n---\n")
            except Exception:
                pass

        render_frame(
            placeholder_poem, 
            placeholder_critique, 
            st.session_state.history_data, 
            metrics, 
            new_constraints, 
            img_kiss
        )
            
        time.sleep(speed)

elif st.session_state.app_state == 'stats':
    # STATS EXPLAINED VIEW
    st.markdown("""
    <style>
    .stApp {
        background-color: #e5e1d9 !important;
        background-image: none !important;
    }
    
    .stats-container {
        width: 60%;
        margin: 0 auto;
        padding-top: 50px;
        font-family: 'Playfair Display', serif;
        color: #4a0404;
    }
    
    h1 {
        font-family: 'Dancing Script', cursive;
        color: #4a0404; 
        text-align: center;
        font-size: 3em;
        margin-bottom: 40px;
    }
    
    h2 {
        border-bottom: 2px solid #dcd0c0;
        padding-bottom: 10px;
        margin-top: 40px;
        color: #6a0404;
    }
    
    .explanation-box {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #8b0000;
    }
    
    .metric-title {
        font-weight: bold;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Ink Flow")
        if st.button("Back to Poem"):
            st.session_state.app_state = 'letter'
            st.rerun()
        if st.button("The Volume"):
            st.session_state.app_state = 'volume'
            st.rerun()

    st.markdown("""
<div class="stats-container">
<h1>The Critic's Rubric</h1>

<h2>I. The Metrics</h2>
<div class="explanation-box">
<div class="metric-title">Sentiment</div>
<p>A measure of the soul's temperature. It ranges from <strong>-1.0 (Absolute Despair)</strong> to <strong>+1.0 (Blinding Joy)</strong>. 
The Critic observes this ensuring the poem doesn't stagnate in one emotion for too long.</p>
</div>
<div class="explanation-box">
<div class="metric-title">Readability</div>
<p>Calculated via the <em>Flesch-Kincaid</em> grade. It determines the complexity of the language. 
If the score rises too high, the verses have become tangled thickets, difficult for the heart to navigate.</p>
</div>

<h2>II. The Directives</h2>
<p>When the Critic speaks, he demands specific corrections to the poet's flow:</p>

<div class="explanation-box">
<div class="metric-title">"The prose is tangled..."</div>
<p><strong>Trigger:</strong> Readability is too high.</p>
<p><strong>Meaning:</strong> The imagery has become too abstract or wordy. The Critic demands <em>Direct Metaphors</em>—clarity that strikes true.</p>
</div>

<div class="explanation-box">
<div class="metric-title">"The river flows too evenly..."</div>
<p><strong>Trigger:</strong> Sentiment has not changed significantly over recent stanzas.</p>
<p><strong>Meaning:</strong> The poem risks monotony. A <em>Dramatic Shift</em> is required—a sudden plunge into grief or a burst of ecstasy.</p>
</div>

<div class="explanation-box">
<div class="metric-title">"The darkness is absolute..."</div>
<p><strong>Trigger:</strong> Sentiment has dipped below -0.5 (Deep Melancholy).</p>
<p><strong>Meaning:</strong> The despair is overwhelming. The Critic asks for a <em>Glimmer of Hope</em> to contrast the shadows.</p>
</div>

<div class="explanation-box">
<div class="metric-title">"The stars are worn thin..."</div>
<p><strong>Trigger:</strong> Reuse of the same symbols (moon, night, tears) too frequently.</p>
<p><strong>Meaning:</strong> The imagery is stale. You must introduce a <em>New Cosmic Symbol</em> to refresh the poetic universe.</p>
</div>

</div>
""", unsafe_allow_html=True)
elif st.session_state.app_state == 'volume':
    # THE GILDED VOLUME PAGE
    st.markdown("""
    <style>
    .stApp {
        background-color: #e5e1d9 !important;
        background-image: none !important;
    }
    
    .volume-container {
        width: 60%;
        margin: 0 auto;
        padding-top: 50px;
        padding-bottom: 50px;
    }
    
    h1 {
        font-family: 'Dancing Script', cursive;
        color: #4a0404; 
        text-align: center;
        font-size: 3em;
        margin-bottom: 40px;
    }
    
    .praise-stanza {
        font-family: 'Dancing Script', cursive;
        font-size: 24px;
        line-height: 1.6;
        color: #4a0404;
        background-color: rgba(255, 255, 255, 0.4);
        padding: 30px;
        margin-bottom: 30px;
        border-radius: 5px;
        border: 1px solid #dcd0c0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        position: relative;
    }
    
    .praise-stanza::before {
        content: "Drafted";
        position: absolute;
        top: 10px;
        right: 15px;
        font-family: 'Playfair Display', serif;
        font-size: 12px;
        color: #8b0000;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Ink Flow")
        if st.button("Back to Poem"):
            st.session_state.app_state = 'letter'
            st.rerun()
            
    st.markdown("""
    <div class="volume-container">
    <h1>The Gilded Volume</h1>
    """, unsafe_allow_html=True)
    
    if os.path.exists(BEST_POEMS_FILE):
        try:
            with open(BEST_POEMS_FILE, "r", encoding="utf-8") as f:
                content = f.read()

            poems = [p.strip() for p in content.split("\n---\n") if p.strip()]
            
            if not poems:
                 st.markdown("<p style='text-align:center; font-family:Playfair Display;'>The pages are yet blank.</p>", unsafe_allow_html=True)
            else:
                for p in reversed(poems): # Show newest first
                    st.markdown(f'<div class="praise-stanza">{p.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Could not read the volume: {e}")
    else:
        st.markdown("<p style='text-align:center; font-family:Playfair Display;'>The pages are yet blank.</p>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
