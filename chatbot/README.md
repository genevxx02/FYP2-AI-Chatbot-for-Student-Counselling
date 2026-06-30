# AI Student Counselling Chatbot

AI-powered counselling chatbot for Malaysian university students. Combines emotion and intent NLP, RAG knowledge retrieval, Groq LLM replies, crisis detection, and an admin dashboard.

## Features

- Anonymous student chat (no login required)
- Malaysian English / Manglish input with English counselling replies
- Emotion detection (Hugging Face DistilRoBERTa)
- Intent classification (TF-IDF + Logistic Regression)
- RAG over a local counselling knowledge base (ChromaDB)
- Groq LLM responses with template fallback
- Crisis keyword detection with Malaysian hotline referrals
- Admin dashboard: analytics, resources, users, system settings

## Quick start (local)

```powershell
git clone <your-repo-url>
cd chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | Random string for Flask sessions |
| `GROQ_API_KEY` | Recommended | From [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | No | Default: `llama-3.3-70b-versatile` |
| `FLASK_DEBUG` | No | Set `true` locally, `false` in production |

Train the intent model (only needed if `nlp/model.pkl` is missing):

```powershell
python nlp/train.py
```

Run the server:

```powershell
python app.py
```

Open **http://127.0.0.1:5000**

**First admin account:** go to `/admin` → Register, or `POST /api/register` with email and password.

## Production deployment

The app ships with `wsgi.py`, `Procfile`, and `gunicorn` for cloud hosts (Render, Railway, etc.).

### Environment variables (production)

Set these on your hosting platform:

```
SECRET_KEY=<long-random-string>
GROQ_API_KEY=<your-key>
GROQ_MODEL=llama-3.3-70b-versatile
FLASK_ENV=production
FLASK_DEBUG=false
RAG_ENABLED=true
```

### Render / Railway

1. Connect your GitHub repository (root = `chatbot/` folder).
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`  
   (Or use the included `Procfile`.)
4. Add environment variables above.
5. Deploy. On first boot the app creates SQLite data in `instance/` and builds the ChromaDB index from `knowledge_base/`.

**Notes for cloud deployment:**

- Use **1 worker** — the emotion model and ChromaDB index load per process (~500 MB+ RAM).
- First chat message may be slow while Hugging Face downloads the emotion model.
- SQLite and ChromaDB persist on hosts with a **persistent disk**; on ephemeral free tiers, data resets on redeploy.
- Without `GROQ_API_KEY`, the chatbot still works using template fallback responses.

### Manual production run

```powershell
pip install -r requirements.txt
set FLASK_DEBUG=false
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 1 --timeout 120
```

## Testing

```powershell
pip install -r requirements-dev.txt
python scripts/run_functional_tests.py
python -m pytest tests/test_system.py -v
python evaluate.py --save-report
python scripts/calculate_sus.py evaluation/usability_responses.json
```

See `docs/FUNCTIONAL_TEST_CHECKLIST.md` for the full test plan.

## API endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /` | No | Student chat UI |
| `POST /get` | No | Send chat message |
| `POST /reset` | No | Clear chat session |
| `GET /api/resources` | No | Crisis helplines |
| `GET /api/llm-status` | No | Groq configuration status |
| `GET /api/rag-status` | No | RAG index status |
| `POST /api/login` | No | Admin login |
| `GET /api/dashboard` | Admin | Dashboard statistics |
| `GET /api/analytics` | Admin | Time-filtered analytics |
| `GET /api/analytics/export` | Admin | CSV export |

## Project structure

```
chatbot/
├── app.py              # Flask app, models, routes
├── wsgi.py             # Production entry point
├── core/               # Chatbot logic, LLM, template engine
├── nlp/                # Sentiment, intent, Malay normaliser, model.pkl
├── rag/                # ChromaDB RAG pipeline
├── knowledge_base/     # RAG source markdown (8 documents)
├── templates/        # Web UI (index.html)
├── tests/              # Automated tests
├── scripts/            # Utilities and test runners
├── evaluation/         # NLP test sets and usability data
└── docs/               # Demo script and test checklist
```

## Security checklist before going public

- [ ] Never commit `.env` (already in `.gitignore`)
- [ ] Set a strong `SECRET_KEY` in production
- [ ] Set `FLASK_DEBUG=false` in production
- [ ] Register admin accounts with strong passwords
- [ ] Keep `GROQ_API_KEY` only on the server

## License

Final Year Project — academic use.
