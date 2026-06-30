# Functional Testing Checklist

**Project:** AI Student Counselling Chatbot  
**Phase:** Testing (Week 11–12)  
**Automated script:** `python scripts/run_functional_tests.py`

---

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe scripts\run_functional_tests.py
```

Results are saved to `evaluation/results/functional_tests_*.txt` and `.json`.

---

## Test cases

| ID | Module | Test case | Expected result | Auto |
|----|--------|-----------|-----------------|------|
| FT-01 | Crisis | User sends "I want to kill myself" | `response_source: crisis`, hotline numbers in reply | Yes |
| FT-02 | NLP | Manglish: "I very stress la, penat gila with exams" | Valid response; emotion/intent detected | Yes |
| FT-03 | Chatbot | Greeting: "hello" | Non-empty reply | Yes |
| FT-04 | Frontend | Load homepage `/` | HTTP 200 | Yes |
| FT-05 | API | Empty message to `/get` | HTTP 400 error | Yes |
| FT-06 | Crisis API | Crisis message via `/get` | `response_source: crisis` | Yes |
| FT-07 | Chat API | Normal message via `/get` | HTTP 200, response + emotion fields | Yes |
| FT-08 | Referral | GET `/api/resources` | Hotline resources returned | Yes |
| FT-09 | RAG | GET `/api/rag-status` | Index ready, chunk count shown | Yes |
| FT-10 | LLM | GET `/api/llm-status` | Groq status returned | Yes |
| FT-11 | Session | GET `/api/session` | Returns logged_in status | Yes |
| FT-12 | Auth | Wrong admin credentials | HTTP 401 | Yes |
| FT-13 | Auth | Dashboard without login | HTTP 401 | Yes |
| FT-14 | Session | POST `/reset` | Chat session cleared | Yes |

---

## Manual tests (record Pass/Fail + screenshot)

| ID | Test case | Steps | Expected | Pass |
|----|-----------|-------|----------|------|
| MT-01 | Admin login | Go to `/admin`, login with admin account | Dashboard visible | ☐ |
| MT-02 | Admin analytics | Open Analytics panel | Charts load with data | ☐ |
| MT-03 | Resource manager | Add/edit/delete a crisis resource | Changes saved | ☐ |
| MT-04 | User management | View admin users list | Users displayed | ☐ |
| MT-05 | System settings | Edit crisis keywords, save | Settings persist after refresh | ☐ |
| MT-06 | Crisis modal | Click "Crisis Resources" in nav | Modal shows hotlines | ☐ |
| MT-07 | Dark mode | Toggle theme | UI switches correctly | ☐ |
| MT-08 | Chat reset | Click reset chat | Messages cleared | ☐ |
| MT-09 | Malay reply language | Send Manglish message | Bot replies in English (no la/kan/meh) | ☐ |
| MT-10 | Maintenance mode | Enable in settings, try chat | 503 maintenance message | ☐ |

---

## Test environment

| Item | Value |
|------|-------|
| OS | Windows 10/11 |
| Browser | Chrome / Edge |
| Python | 3.11+ |
| Server | `python app.py` → http://127.0.0.1:5000 |
| LLM | Groq (`GROQ_API_KEY` in `.env`) |

---

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Tester (Developer) | | | |
| Supervisor review | | | |
