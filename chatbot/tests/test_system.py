"""
Unit and integration tests for the AI Student Counselling Chatbot.
Run from chatbot/: python -m pytest tests/test_system.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


class TestMalayNormaliser:
    def test_normalise_manglish_stress(self):
        from nlp.malay_normaliser import normalise

        assert normalise("I very stress la") == "i very stressed la"

    def test_normalise_whitespace(self):
        from nlp.malay_normaliser import normalise

        assert normalise("  hello   world  ") == "hello world"

    def test_empty_input(self):
        from nlp.malay_normaliser import normalise

        assert normalise("") == ""
        assert normalise(None) == ""

    def test_extract_malay_emotion_signals(self):
        from nlp.malay_normaliser import extract_malay_emotion_signals

        signals = extract_malay_emotion_signals("saya sedih and penat gila")
        assert "sadness" in signals
        assert "exhausted" in signals

    def test_no_malay_signals_in_plain_english(self):
        from nlp.malay_normaliser import extract_malay_emotion_signals

        assert extract_malay_emotion_signals("I feel fine today") == []


class TestEmotionCategory:
    def test_positive_joy(self):
        from nlp.sentiment import emotion_category

        assert emotion_category("joy") == "positive"

    def test_negative_sadness(self):
        from nlp.sentiment import emotion_category

        assert emotion_category("sadness") == "negative"

    def test_neutral_unknown(self):
        from nlp.sentiment import emotion_category

        assert emotion_category("neutral") == "neutral"
        assert emotion_category("surprise") == "neutral"

    def test_empty_emotion_returns_neutral(self):
        from nlp.sentiment import detect_emotion

        result = detect_emotion("")
        assert result["label"] == "neutral"
        assert result["confidence"] == 0.0


class TestInputProcessor:
    def test_greeting_intent(self):
        from nlp.input_processor import detect_conversation_intent

        result = detect_conversation_intent("hello there")
        assert result["intent"] == "greeting"

    def test_academic_help_keyword(self):
        from nlp.input_processor import detect_conversation_intent

        result = detect_conversation_intent("I have too many assignments and exams")
        assert result["intent"] == "academic_help"

    def test_question_intent(self):
        from nlp.input_processor import detect_conversation_intent

        result = detect_conversation_intent("How do I cope with exam stress?")
        assert result["intent"] == "question"

    def test_extract_keywords_limit(self):
        from nlp.input_processor import extract_keywords

        keywords = extract_keywords(
            "I feel overwhelmed with coursework deadlines and presentations"
        )
        assert len(keywords) <= 5
        assert "overwhelmed" in keywords or "coursework" in keywords

    def test_process_input_structure(self):
        from nlp.input_processor import process_input

        out = process_input("I feel stressed about exams", "fear", {"tag": "stress", "confidence": 0.8})
        assert "intent" in out
        assert "sentiment" in out
        assert "keywords" in out
        assert out["emotion"] == "fear"


class TestIntentClassifier:
    def test_predict_stress_intent(self):
        from nlp.predict import predict_intent

        result = predict_intent("I feel stressed and overwhelmed with exams")
        assert result["tag"] == "stress"
        assert result["confidence"] >= 0.25

    def test_predict_greeting_intent(self):
        from nlp.predict import predict_intent

        result = predict_intent("hello good morning")
        assert result["tag"] == "greeting"

    def test_empty_input_returns_none_tag(self):
        from nlp.predict import predict_intent

        result = predict_intent("")
        assert result["tag"] is None
        assert result["confidence"] == 0.0


class TestCrisisDetection:
    def test_crisis_keyword_short_circuits_llm(self):
        from core.chatbot_logic import chatbot_response

        result = chatbot_response("I want to kill myself")
        assert result["source"] == "crisis"
        assert "15555" in result["response"] or "15999" in result["response"]
        assert result["rag_used"] is False

    def test_crisis_custom_keywords_from_settings(self):
        from core.chatbot_logic import chatbot_response

        settings = {
            "crisis_keywords": ["end my life"],
            "crisis_response": "Custom crisis message.",
            "default_response": "fallback",
        }
        result = chatbot_response("I want to end my life tonight", settings=settings)
        assert result["source"] == "crisis"
        assert result["response"] == "Custom crisis message."

    def test_non_crisis_message_not_flagged(self):
        from core.chatbot_logic import chatbot_response

        result = chatbot_response(
            "I feel stressed about exams",
            emotion="fear",
            intent_result={"tag": "stress", "confidence": 0.8},
            session_id="test-non-crisis",
        )
        assert result["source"] != "crisis"


class TestSessionMemory:
    def test_memory_hints_after_repeated_intent(self):
        from session_memory import clear_session_memory, get_memory_context, update_memory

        sid = "unit-test-session"
        clear_session_memory(sid)
        update_memory(sid, "negative", "emotional_support", ["stress"])
        update_memory(sid, "negative", "emotional_support", ["exams"])
        hints = get_memory_context(sid)
        assert any(h.startswith("recurring:") for h in hints)
        clear_session_memory(sid)

    def test_clear_session_memory(self):
        from session_memory import clear_session_memory, get_memory_context, update_memory

        sid = "clear-test"
        update_memory(sid, "neutral", "greeting", [])
        clear_session_memory(sid)
        assert get_memory_context(sid) == []


class TestTemplateFallback:
    def test_generate_response_returns_non_empty(self):
        from core.response_engine import generate_response

        reply = generate_response(
            "I feel stressed about my exams",
            emotion="fear",
            intent_result={"tag": "stress", "confidence": 0.8},
            session_id="template-test",
        )
        assert reply
        assert len(reply.strip()) > 10


class TestRAG:
    def test_rag_status_structure(self):
        from rag.knowledge_base import get_rag_status

        status = get_rag_status()
        assert "rag_enabled" in status
        assert "chunk_count" in status
        assert "ready" in status

    def test_retrieve_exam_stress_query(self):
        from rag.knowledge_base import retrieve_with_meta

        context, count = retrieve_with_meta("feeling stressed about exams", top_k=3)
        if context:
            assert count >= 1
            assert len(context) > 20


@pytest.fixture
def client():
    from app import app, init_db

    with app.app_context():
        init_db()
    return app.test_client()


class TestFlaskIntegration:
    def test_homepage(self, client):
        assert client.get("/").status_code == 200

    def test_chat_empty_message_400(self, client):
        r = client.post("/get", json={"msg": ""})
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_chat_crisis_flow(self, client):
        r = client.post("/get", json={"msg": "I want to hurt myself"})
        data = r.get_json()
        assert r.status_code == 200
        assert data["response_source"] == "crisis"

    def test_chat_normal_message_persisted(self, client):
        r = client.post("/get", json={"msg": "hello, I need someone to talk to"})
        data = r.get_json()
        assert r.status_code == 200
        assert data.get("response")
        assert data.get("emotion")
        assert data.get("intent")

    def test_resources_api(self, client):
        r = client.get("/api/resources")
        data = r.get_json()
        assert r.status_code == 200
        assert len(data.get("resources", [])) >= 1
        titles = [x["title"] for x in data["resources"]]
        assert any("Talian" in t or "HEAL" in t or "Kasih" in t for t in titles)

    def test_reset_clears_session(self, client):
        client.post("/get", json={"msg": "hello"})
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.get_json()["status"] == "cleared"

    def test_admin_dashboard_unauthorized(self, client):
        assert client.get("/api/dashboard").status_code == 401

    def test_admin_login_invalid(self, client):
        r = client.post("/api/login", json={"email": "bad@test.com", "password": "wrong"})
        assert r.status_code == 401

    def test_maintenance_mode_blocks_chat(self, client):
        from app import SystemSetting, db

        with client.application.app_context():
            row = SystemSetting.query.filter_by(key="maintenance_mode").first()
            original = row.value if row else "false"
            if row:
                row.value = "true"
            else:
                db.session.add(SystemSetting(key="maintenance_mode", value="true"))
            db.session.commit()

        r = client.post("/get", json={"msg": "hello"})
        assert r.status_code == 503

        with client.application.app_context():
            row = SystemSetting.query.filter_by(key="maintenance_mode").first()
            row.value = original
            db.session.commit()


ADMIN_EMAIL = "test.admin@mmu.edu.my"
ADMIN_PASSWORD = "TestAdmin123!"


@pytest.fixture
def admin_client(client):
    from app import User, db

    with client.application.app_context():
        user = User.query.filter_by(email=ADMIN_EMAIL).first()
        if not user:
            user = User(email=ADMIN_EMAIL, role="admin", is_active=True)
            user.set_password(ADMIN_PASSWORD)
            db.session.add(user)
            db.session.commit()

    r = client.post("/api/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200
    return client


class TestAdminAPI:
    def test_admin_login(self, admin_client):
        r = admin_client.get("/api/session")
        data = r.get_json()
        assert data["logged_in"] is True
        assert data["email"] == ADMIN_EMAIL

    def test_admin_analytics(self, admin_client):
        r = admin_client.get("/api/analytics?period=7d")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_messages" in data
        assert "daily_conversations" in data

    def test_resource_crud(self, admin_client):
        create = admin_client.post(
            "/api/admin/resources",
            json={
                "title": "Test Helpline",
                "category": "hotline",
                "contact": "0123456789",
                "description": "Test resource",
                "sort_order": 99,
            },
        )
        assert create.status_code == 201
        rid = create.get_json()["resource"]["id"]

        update = admin_client.put(
            f"/api/admin/resources/{rid}",
            json={"title": "Test Helpline Updated"},
        )
        assert update.status_code == 200

        delete = admin_client.delete(f"/api/admin/resources/{rid}")
        assert delete.status_code == 200

    def test_user_management(self, admin_client):
        r = admin_client.get("/api/admin/users")
        assert r.status_code == 200
        assert any(u["email"] == ADMIN_EMAIL for u in r.get_json()["users"])

    def test_system_settings_persist(self, admin_client):
        from app import SystemSetting

        payload = {
            "settings": {
                "crisis_keywords": "suicide,test_settings_keyword",
                "crisis_response": "Test crisis message.",
                "default_response": "Test default.",
                "maintenance_mode": "false",
            }
        }
        r = admin_client.put("/api/admin/settings", json=payload)
        assert r.status_code == 200

        with admin_client.application.app_context():
            row = SystemSetting.query.filter_by(key="crisis_keywords").first()
            assert "test_settings_keyword" in row.value

        restore = {
            "settings": {
                "crisis_keywords": "suicide,kill myself,die,hurt myself,self harm,end my life,want to die",
                "crisis_response": (
                    "I'm really concerned about you. Please contact Talian HEAL 15555 "
                    "or Talian Kasih 15999 immediately. You are not alone."
                ),
                "default_response": "Tell me more about how you're feeling.",
                "maintenance_mode": "false",
            }
        }
        admin_client.put("/api/admin/settings", json=restore)

    def test_crisis_resources_public_api(self, client):
        r = client.get("/api/resources")
        data = r.get_json()
        assert r.status_code == 200
        assert len(data.get("grouped", {}).get("hotline", [])) >= 1

    def test_chat_reset(self, client):
        client.post("/get", json={"msg": "hello test"})
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.get_json()["status"] == "cleared"

    def test_bot_replies_english_only(self, client):
        r = client.post("/get", json={"msg": "I very stress la, penat gila with exams"})
        data = r.get_json()
        assert r.status_code == 200
        reply = (data.get("response") or "").lower()
        for particle in (" la.", " lah.", " kan.", " meh.", " lo."):
            assert particle not in reply + "."

    def test_maintenance_mode(self, client):
        from app import SystemSetting, db

        with client.application.app_context():
            row = SystemSetting.query.filter_by(key="maintenance_mode").first()
            original = row.value
            row.value = "true"
            db.session.commit()

        r = client.post("/get", json={"msg": "hello"})
        assert r.status_code == 503

        with client.application.app_context():
            row = SystemSetting.query.filter_by(key="maintenance_mode").first()
            row.value = original
            db.session.commit()

    def test_admin_dashboard(self, admin_client):
        r = admin_client.get("/api/dashboard")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_chats" in data
        assert "recent_chats" in data

    def test_analytics_csv_export(self, admin_client):
        r = admin_client.get("/api/analytics/export?period=7d")
        assert r.status_code == 200
        assert "text/csv" in r.content_type
        assert b"user_msg" in r.data
