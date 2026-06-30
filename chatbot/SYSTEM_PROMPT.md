# Student Counselling Chatbot — System Architecture

## Core objective

Provide empathetic, context-aware counselling support for Malaysian university students using NLP, RAG, and an LLM (Groq).

## Response pipeline

For every user message (`core/chatbot_logic.py`):

1. **Normalise** Malaysian English / Manglish / Malay input (`nlp/malay_normaliser.py`)
2. **Crisis check** — keyword match bypasses LLM and returns emergency contacts
3. **Emotion detection** — Hugging Face distilRoBERTa (`nlp/sentiment.py`)
4. **Intent classification** — TF-IDF + Logistic Regression (`nlp/predict.py`)
5. **Conversation intent + keywords** — rule-based layer (`nlp/input_processor.py`)
6. **RAG retrieval** — ChromaDB + SentenceTransformers (`rag/knowledge_base.py`, `knowledge_base/*.md`)
7. **LLM reply** — Groq API (`core/claude_service.py`, requires `GROQ_API_KEY`)
8. **Template fallback** — structured responses if LLM unavailable (`core/response_engine.py`, `counselling_knowledge.json`)

## Language rules

- **Understand** Malaysian English, Manglish, and mixed Malay input from students.
- **Reply in clear English only** — do not mirror Malay particles or slang in bot responses.

## Admin features

- Dashboard analytics (`/api/dashboard`, `/api/analytics`)
- Crisis resource management
- User management and system settings
- SQLite database at `instance/chatbot.db`

## Evaluation

Run `python evaluate.py --save-report` for emotion/intent metrics.
