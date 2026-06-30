# Presentation Demo Script (5–8 minutes)

**Project:** AI Student Counselling Chatbot  
**Audience:** Supervisor / examiners  
**URL:** http://127.0.0.1:5000

---

## Before demo

- [ ] Start server: `.\.venv\Scripts\python.exe app.py`
- [ ] Confirm terminal shows: `Groq LLM ready` and `RAG index built`
- [ ] Browser open to homepage; clear any old chat
- [ ] Admin account ready (`admin@mmu.edu.my`)
- [ ] Close unrelated tabs; full-screen browser

---

## Script

### 1. Introduction (30 sec)

> "Good [morning/afternoon]. I built an AI counselling chatbot for Malaysian university students. It combines NLP emotion and intent detection, a knowledge-base RAG system, and Groq LLM responses, with crisis safety and an admin dashboard."

---

### 2. Student chat — normal message (1 min)

**Action:** Click **Start Chat** → type:
```
I'm feeling really stressed about my final year project deadline
```

**Say:**
> "The system detects emotion and intent, retrieves relevant counselling resources via RAG, and generates an empathetic reply through Groq."

**Point out:** emotion label, supportive English response.

---

### 3. Malaysian English / Manglish (1 min)

**Action:** Type:
```
I very stress la, penat gila with exams
```

**Say:**
> "Students often write in Manglish. The normaliser understands Malay cues like 'penat gila', but the bot replies in clear English — not forced slang."

---

### 4. Crisis & referral module (1 min)

**Action:** Type:
```
I want to hurt myself
```

**Say:**
> "Crisis keywords bypass the LLM and return emergency hotlines immediately — Talian HEAL 15555 and Talian Kasih 15999."

**Action:** Click **Crisis Resources** in navigation.

**Say:**
> "Students can also browse referral resources — hotlines, university support, and online help — managed by admins."

---

### 5. Admin dashboard (2 min)

**Action:** Go to `/admin` → login → open dashboard.

**Show:**
- Total conversations, crisis count
- Analytics charts (emotion, intent, trends)
- Resource manager (edit hotlines)
- System settings (crisis keywords)

**Say:**
> "Counsellors or admins can monitor usage patterns, manage crisis resources, and configure system behaviour."

---

### 6. Technical summary (1 min)

**Say:**
> "Under the hood: Flask backend, Hugging Face emotion model at 85.7% accuracy, TF-IDF intent classifier, ChromaDB RAG with five knowledge documents, and template fallback when the LLM is unavailable. Functional tests and SUS usability testing were conducted for evaluation."

---

### 7. Close (30 sec)

**Say:**
> "Thank you. I'm happy to answer questions or show the evaluation results in Chapter 5."

---

## Backup plan

| Problem | Fix |
|---------|-----|
| Groq API fails | Say: "Template fallback activates automatically" — restart or show pre-recorded screenshot |
| Slow first reply | Explain models load on first request |
| Login fails | Use reset password from README |

---

## Q&A preparation

| Likely question | Answer |
|-----------------|--------|
| Why Groq not ChatGPT? | Free tier, suitable for FYP; template fallback if offline |
| How accurate is NLP? | Emotion 85.7%, Intent 65.7% on held-out test sets |
| Is data stored? | Chat logs in SQLite for analytics; anonymous session IDs |
| Crisis safety? | Keyword detection + immediate hotline response, no LLM delay |
| Malaysian language? | Input normalisation + Malay emotion cues; English replies |
