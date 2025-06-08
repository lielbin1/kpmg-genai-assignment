![image](https://github.com/user-attachments/assets/ddbe956e-11c5-4a19-8432-845ef5f1a778)


<h1 align="center">GenAI Developer Assessment Assignment</h1>

This project is a two-part solution for the KPMG GenAI Developer Assessment. It demonstrates integration with Azure Document Intelligence, GPT-4o, and embedding-based retrieval using ADA-002, covering both form-based field extraction and an AI-powered Q\&A chatbot system.

---

## 🛠️ Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Environment

* **Windows (PowerShell):**

  ```bash
  .venv\Scripts\Activate.ps1
  ```

* **Windows (Git Bash):**

  ```bash
  source .venv/Scripts/activate
  ```

* **macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the project root

Add the following keys:

```env
AZURE_ENDPOINT=
AZURE_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
API_BASE_URL=  # Optional, defaults to http://127.0.0.1:8000
```

---

## Part 1 – Form Field Extraction from PDF (OCR)

### Goal:

Extract structured information from ביטוח לאומי (National Insurance Institute) forms. The user uploads a PDF file, and the system uses:

* Azure Document Intelligence for OCR
* Azure OpenAI (GPT-4o) to extract fields and return a structured JSON

### To Run:

```bash
cd part1-OCR
python app.py
```

Then open the displayed URL in your browser, upload a filled-out PDF form, and view the extracted JSON output.

---

## Part 2 – Stateless Medical Chatbot (Q\&A)

### Goal:

Build a microservice-based chatbot that:

* Collects user details (via GPT prompt, not a form)
* Answers user questions about medical services (Clalit, Maccabi, Meuhedet)
* Uses RAG (retrieval-augmented generation) with ADA embeddings
* Supports both English and Hebrew
* Operates statelessly (no server-side memory)

The knowledge base is built from HTML documents in `content/phase2_data/`.

### Technologies:

* FastAPI backend
* Gradio frontend
* Azure OpenAI GPT-4o
* ADA 002 embeddings + cosine similarity for RAG
* Stateless architecture (all context sent with each request)
* Structured logging

### To Run:

> Open two terminals (or terminal tabs):

**Terminal 1: Start the backend server**

```bash
cd part2-ChatBot-Medical-Services
python -m uvicorn Backend:app --reload
```

**Terminal 2: Start the Gradio frontend**

```bash
cd part2-ChatBot-Medical-Services
python Frontend.py
```

Then, visit the Gradio URL to chat with the bot.


## ℹ️ Notes

* The chatbot handles multiple users by passing context from the frontend.
* The PDF extractor works with both Hebrew and English forms.
* Prompts are stored in `prompts/prompt_*.txt` files and injected dynamically.

---

## 💡 Author

Liel Binyamin – Submitted for the 2025 GenAI Developer Assessment at KPMG
