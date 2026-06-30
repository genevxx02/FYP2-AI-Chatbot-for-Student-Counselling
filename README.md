# AI Student Counselling Chatbot — Final Year Project

**Track:** Software Engineering / Information Systems  
**Institution:** Multimedia University (MMU)

## Repository contents

| Path | Description |
|------|-------------|
| `chatbot/` | Full-stack application (Flask, NLP, RAG, Groq LLM, admin dashboard) |
| `Chapter_6_Testing_and_Evaluation.docx` | FYP report — Testing & Evaluation chapter |
| `chatbot/evaluation/` | NLP test sets, usability study data, evaluation results |
| `chatbot/docs/` | Demo script and functional test checklist |

## Quick start

All setup, deployment, and API documentation is in **[chatbot/README.md](chatbot/README.md)**.

```powershell
cd chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Open http://127.0.0.1:5000

## Project highlights

- Emotion detection (Hugging Face DistilRoBERTa) and intent classification (TF-IDF + LR)
- Retrieval-Augmented Generation over a counselling knowledge base (ChromaDB)
- Groq LLM counselling replies with template fallback
- Crisis keyword detection with Malaysian hotline referrals
- Admin dashboard with analytics, resource management, and CSV export
- Malaysian English / Manglish input support

## License

Academic project — Final Year Project submission.
