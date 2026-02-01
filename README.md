# 🌹 ECHOES OF THE EVENING STAR
### *AI & The Self-Correction of Romanian Romanticism*

**An Interactive Art Installation | 2025**

---

## 🎭 The Concept
**Echoes of the Evening Star** stages a recursive paradox: *Can a machine truly master a specific cultural style, and then learn to critique itself out of cliché?*

The system operates as an endless cycle of computational struggle between two warring AI entities:

1.  **The Poet:** A Generative AI (`Gemma-2B`) steeped in the vocabulary and profound melancholy of Mihai Eminescu. It operates on digital impulse, echoing the high-Romantic style of *Luceafărul*.
2.  **The Critic:** An independent NLP system that analyzes the poet’s output against rigid rules. It ruthlessly flags symbolic clichés, excessive melancholy, and structural complexity.
3.  **The Paradox:** The artwork is not the resulting poem, but the visible, agonizing **process of self-correction**. The Critic issues an "Algorithmic Veto," forcing the Poet to evolve and refine its style under continuous pressure.


---

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **Generative Model:** `google/gemma-2b-it` (Running on CPU)
* **Natural Language Processing:** `NLTK (VADER)` for sentiment analysis
* **Linguistic Metrics:** `textstat` for Flesch-Kincaid readability scoring
* **Interface:** `Streamlit` for the real-time installation UI

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/echoes-of-the-evening-star.git](https://github.com/YOUR_USERNAME/echoes-of-the-evening-star.git)
   cd echoes-of-the-evening-star
2. **Install dependencies:**
   ```bash
   pip install streamlit torch transformers nltk textstat
3. **Authentication (Gemma Model Access)**: This project uses the google/gemma-2b-it model. You must have an active Hugging Face account and have accepted the license terms on the Gemma model page. Once access is granted, you must authenticate your machine so the script can securely download the weights:
    ```bash
    huggingface-cli login
4. **Launch**: To start the recursive loop and open the web-based UI where the Poet and Critic interact in real-time
    ```bash
    streamlit run poetic_critique.py
    
