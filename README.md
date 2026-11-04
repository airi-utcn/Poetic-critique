# 🌹 ECHOES OF THE EVENING STAR
### *AI & The Self-Correction of Romanian Romanticism*

**An Interactive Art Installation | 2026**

---

## 🎭 The Concept
**Echoes of the Evening Star** stages a recursive paradox: *Can a machine truly master a specific cultural style, and then learn to critique itself out of cliché?*

The system operates as an endless cycle of computational struggle between two warring AI entities:

1.  **The Poet:** A generative language model (`Gemma-2B`) instructed to write in the melancholic, nature-driven voice of Mihai Eminescu—a central figure of the Romanticism literary movement. It continuously drafts new stanzas, trying to capture the broader essence and atmosphere of his poetry, such as the tone found in *Luceafărul*.
2.  **The Critic:** An algorithmic evaluation engine built with NLTK and readability formulas. It acts as a strict literary judge, scoring the poet's output against rigid mathematical thresholds to flag syntactic complexity, incorrect emotional tone, or a lack of romantic symbols.
3.  **The Paradox:** The artwork exists in the tension between the flowing verse and its visible, agonizing **process of self-correction**. It is equal parts poetry and struggle: as the lines generate, the Critic issues real-time algorithmic judgments, forcing the Poet to evolve and refine its style under continuous pressure.

---

## 🛠️ Technical Stack
- **Architecture:** Decoupled Microservice (Java Backend + Python AI Engine)
- **Backend Framework:** Java 17+ with Spring Boot
- **AI Engine API:** Python 3.12 with Flask & Flask-CORS
- **Generative Model:** `google/gemma-2b-it` (via Hugging Face `transformers` & `torch`)
- **Natural Language Processing:** `nltk` (VADER sentiment analysis) & `textstat` (Readability)
- **Frontend Interface:** Custom HTML/CSS/Vanilla JS (Served via Spring Boot)

---

## 🚀 How to Run

This application requires running the AI Engine and the Web Server as two separate processes. 

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/echoes-of-the-evening-star.git
cd echoes-of-the-evening-star
```

### 2. Configure the Python AI Engine
Navigate to the backend directory, create an isolated virtual environment, and install the required machine learning libraries.
```bash
cd java-backend
python -m venv venv

# Activate the virtual environment:
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies:
pip install flask flask-cors torch transformers nltk textstat huggingface-hub
```

### 3. Authenticate the LLM
This project uses the `google/gemma-2b-it` model. You must have an active Hugging Face account and have accepted the license terms on the Gemma model page. Authenticate your terminal so the script can securely download the model weights:
```bash
huggingface-cli login
```

### 4. Launch the Application
**Terminal 1: Start the AI Engine**
Keep your virtual environment activated and start the Flask server. *Leave this terminal window open in the background.*
```bash
python engine.py
# The terminal should output: "Running on http://127.0.0.1:5000"
```

**Terminal 2 (or IDE): Start the Spring Boot Server**
Open the project in your preferred Java IDE (for example, IntelliJ IDEA) and run the `JavaBackendApplication` class, or use Maven from a new terminal window:
```bash
# From the root of the project:
mvn spring-boot:run
```

**Access the Installation:**
Once both servers are running, open a web browser and navigate to:
**`http://localhost:8080`**