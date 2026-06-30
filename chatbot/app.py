from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect, session, Response
from flask_sqlalchemy import SQLAlchemy
from chatbot_logic import chatbot_response
from claude_service import get_status as claude_status, is_configured as claude_is_configured
from rag.knowledge_base import get_rag_status
from session_memory import clear_session_memory
from sentiment import detect_emotion, emotion_category
from nlp.predict import predict_intent
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
from sqlalchemy import case, func, inspect, or_, text
import csv
import io
import os
import time
import uuid
from collections import defaultdict

app = Flask(__name__, instance_relative_config=True)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

os.makedirs(app.instance_path, exist_ok=True)
_db_file = os.path.join(app.instance_path, "chatbot.db").replace("\\", "/")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{_db_file}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

_rate_limit_store = defaultdict(list)


def rate_limit(max_per_minute=20):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            sid = session.get("session_id", "anon")
            now = time.time()
            window = [t for t in _rate_limit_store[sid] if now - t < 60]
            if len(window) >= max_per_minute:
                return jsonify({"error": "Too many requests. Please slow down."}), 429
            window.append(now)
            _rate_limit_store[sid] = window
            return f(*args, **kwargs)
        return wrapper
    return decorator

CRISIS_KEYWORDS_DEFAULT = "suicide,kill myself,die,hurt myself,self harm,end my life,want to die"
CRISIS_RESPONSE_DEFAULT = (
    "I'm really concerned about you. Please contact Talian HEAL 15555 "
    "or Talian Kasih 15999 immediately. You are not alone."
)
DEFAULT_RESPONSE_DEFAULT = "Tell me more about how you're feeling."


# =========================
# MODELS
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="admin")
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        stored = self.password or ""
        if stored.startswith("pbkdf2:") or stored.startswith("scrypt:"):
            return check_password_hash(stored, raw_password)
        return stored == raw_password

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role or "admin",
            "is_active": self.is_active if self.is_active is not None else True,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
            "last_login": self.last_login.strftime("%Y-%m-%d %H:%M") if self.last_login else "",
        }


class Chat(db.Model):
    __table_args__ = (
        db.Index("ix_chat_session_id", "session_id"),
        db.Index("ix_chat_timestamp", "timestamp"),
    )

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100))
    user_msg = db.Column(db.String(500))
    bot_reply = db.Column(db.String(1000))
    # Stores detected emotion label (e.g. joy, sadness); kept as "sentiment" for DB compatibility
    sentiment = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    intent = db.Column(db.String(50))
    source = db.Column(db.String(20))
    rag_used = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime)


class SessionSummary(db.Model):
    __tablename__ = "session_summary"

    session_id = db.Column(db.String(100), primary_key=True)
    summary = db.Column(db.Text)
    turn_count = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)


class CrisisResource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    contact = db.Column(db.String(120), nullable=False)
    description = db.Column(db.String(255), default="")
    is_active = db.Column(db.Boolean, default=True)
    sort_order = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "contact": self.contact,
            "description": self.description or "",
            "is_active": self.is_active,
            "sort_order": self.sort_order or 0,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M") if self.updated_at else "",
        }


class SystemSetting(db.Model):
    key = db.Column(db.String(80), primary_key=True)
    value = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)


# =========================
# HELPERS
# =========================
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def get_setting(key, default=""):
    row = db.session.get(SystemSetting, key)
    return row.value if row else default


def get_crisis_keywords():
    raw = get_setting("crisis_keywords", CRISIS_KEYWORDS_DEFAULT)
    return [kw.strip().lower() for kw in raw.split(",") if kw.strip()]


def is_crisis_message(message):
    lower = (message or "").lower()
    return any(kw in lower for kw in get_crisis_keywords())


def chat_settings():
    return {
        "crisis_keywords": get_crisis_keywords(),
        "crisis_response": get_setting("crisis_response", CRISIS_RESPONSE_DEFAULT),
        "default_response": get_setting("default_response", DEFAULT_RESPONSE_DEFAULT),
        "maintenance_mode": get_setting("maintenance_mode", "false").lower() == "true",
    }


def is_user_active(user):
    return user is not None and user.is_active is not False


def current_admin_user():
    email = session.get("admin")
    if not email:
        return None
    user = User.query.filter_by(email=email).first()
    if not is_user_active(user):
        return None
    return user


def _intent_bucket(chat):
    """Return stored intent label, or unclassified for legacy rows."""
    return chat.intent or "unclassified"


def _emotion_label(chat):
    """Return stored Hugging Face emotion label."""
    return chat.sentiment or "unknown"


def _most_common(counts: dict) -> dict:
    if not counts:
        return {"label": "none", "count": 0}
    label = max(counts, key=counts.get)
    return {"label": label, "count": counts[label]}


def _counts_to_breakdown(counts: dict, key_name: str) -> list:
    return [
        {key_name: label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def _sentiment_bucket(chat):
    """Map stored emotion label to positive/negative/neutral for dashboard counts."""
    return emotion_category(chat.sentiment)


POSITIVE_EMOTIONS = ("joy",)
NEGATIVE_EMOTIONS = ("sadness", "anger", "fear", "disgust")


def _sentiment_bucket_expr():
    emotion = func.lower(func.coalesce(Chat.sentiment, ""))
    return case(
        (emotion.in_(POSITIVE_EMOTIONS), "positive"),
        (emotion.in_(NEGATIVE_EMOTIONS), "negative"),
        else_="neutral",
    )


def _emotion_label_expr():
    return func.coalesce(Chat.sentiment, "unknown")


def _intent_label_expr():
    return func.coalesce(Chat.intent, "unclassified")


def _crisis_conditions():
    conditions = [Chat.source == "crisis"]
    for kw in get_crisis_keywords():
        conditions.append(func.lower(func.coalesce(Chat.user_msg, "")).contains(kw.lower()))
    return or_(*conditions)


def _apply_cutoff(query, cutoff=None):
    if cutoff is not None:
        query = query.filter(Chat.timestamp >= cutoff)
    return query


def _sql_total(cutoff=None):
    return _apply_cutoff(db.session.query(func.count(Chat.id)), cutoff).scalar() or 0


def _sql_emotion_distribution(cutoff=None):
    label = _emotion_label_expr()
    rows = (
        _apply_cutoff(db.session.query(label, func.count(Chat.id)), cutoff)
        .group_by(label)
        .all()
    )
    return dict(rows)


def _sql_intent_distribution(cutoff=None):
    label = _intent_label_expr()
    rows = (
        _apply_cutoff(db.session.query(label, func.count(Chat.id)), cutoff)
        .group_by(label)
        .all()
    )
    return dict(rows)


def _sql_sentiment_buckets(cutoff=None):
    bucket = _sentiment_bucket_expr()
    rows = (
        _apply_cutoff(db.session.query(bucket, func.count(Chat.id)), cutoff)
        .group_by(bucket)
        .all()
    )
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for label, count in rows:
        counts[label] = count
    return counts


def _sql_crisis_count(cutoff=None):
    return (
        _apply_cutoff(db.session.query(func.count(Chat.id)), cutoff)
        .filter(_crisis_conditions())
        .scalar()
        or 0
    )


def _sql_daily_counts(cutoff=None):
    day = func.date(Chat.timestamp)
    rows = (
        _apply_cutoff(
            db.session.query(day, func.count(Chat.id)).filter(Chat.timestamp.isnot(None)),
            cutoff,
        )
        .group_by(day)
        .order_by(day)
        .all()
    )
    return {str(row[0]): row[1] for row in rows}


def _sql_crisis_by_day(cutoff=None):
    day = func.date(Chat.timestamp)
    rows = (
        _apply_cutoff(
            db.session.query(day, func.count(Chat.id))
            .filter(Chat.timestamp.isnot(None))
            .filter(_crisis_conditions()),
            cutoff,
        )
        .group_by(day)
        .order_by(day)
        .all()
    )
    return {str(row[0]): row[1] for row in rows}


def _sql_peak_hour(cutoff=None):
    hour_expr = func.cast(func.strftime("%H", Chat.timestamp), db.Integer)
    row = (
        _apply_cutoff(
            db.session.query(hour_expr, func.count(Chat.id)).filter(Chat.timestamp.isnot(None)),
            cutoff,
        )
        .group_by(hour_expr)
        .order_by(func.count(Chat.id).desc())
        .first()
    )
    return {"hour": row[0], "count": row[1]} if row else None


def _sql_unique_sessions(cutoff=None):
    return (
        _apply_cutoff(db.session.query(func.count(func.distinct(Chat.session_id))), cutoff)
        .scalar()
        or 0
    )


def _recent_chats(limit=10):
    rows = Chat.query.order_by(Chat.timestamp.desc()).limit(limit).all()
    return [
        {
            "user_msg": c.user_msg[:80] + ("..." if len(c.user_msg or "") > 80 else ""),
            "bot_reply": c.bot_reply[:80] + ("..." if len(c.bot_reply or "") > 80 else ""),
            "sentiment": c.sentiment,
            "emotion": c.sentiment,
            "confidence": c.confidence,
            "intent": _intent_bucket(c),
            "timestamp": c.timestamp.strftime("%Y-%m-%d %H:%M") if c.timestamp else "",
        }
        for c in rows
    ]


def _ensure_chat_columns():
    inspector = inspect(db.engine)
    if "chat" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("chat")}
    alters = []
    if "confidence" not in columns:
        alters.append("ALTER TABLE chat ADD COLUMN confidence FLOAT")
    if "intent" not in columns:
        alters.append("ALTER TABLE chat ADD COLUMN intent VARCHAR(50)")
    if "source" not in columns:
        alters.append("ALTER TABLE chat ADD COLUMN source VARCHAR(20)")
    if "rag_used" not in columns:
        alters.append("ALTER TABLE chat ADD COLUMN rag_used BOOLEAN DEFAULT 0")
    with db.engine.begin() as conn:
        for stmt in alters:
            conn.execute(text(stmt))


def _ensure_chat_indexes():
    inspector = inspect(db.engine)
    if "chat" not in inspector.get_table_names():
        return
    existing = {idx["name"] for idx in inspector.get_indexes("chat")}
    statements = []
    if "ix_chat_session_id" not in existing:
        statements.append("CREATE INDEX IF NOT EXISTS ix_chat_session_id ON chat (session_id)")
    if "ix_chat_timestamp" not in existing:
        statements.append("CREATE INDEX IF NOT EXISTS ix_chat_timestamp ON chat (timestamp)")
    with db.engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _ensure_user_columns():
    inspector = inspect(db.engine)
    if "user" not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns("user")}
    alters = []
    if "role" not in columns:
        alters.append("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'admin'")
    if "is_active" not in columns:
        alters.append("ALTER TABLE user ADD COLUMN is_active BOOLEAN DEFAULT 1")
    if "created_at" not in columns:
        alters.append("ALTER TABLE user ADD COLUMN created_at DATETIME")
    if "last_login" not in columns:
        alters.append("ALTER TABLE user ADD COLUMN last_login DATETIME")
    with db.engine.begin() as conn:
        for stmt in alters:
            conn.execute(text(stmt))


def _seed_defaults():
    if not SystemSetting.query.first():
        defaults = {
            "crisis_keywords": CRISIS_KEYWORDS_DEFAULT,
            "crisis_response": CRISIS_RESPONSE_DEFAULT,
            "default_response": DEFAULT_RESPONSE_DEFAULT,
            "maintenance_mode": "false",
        }
        for key, value in defaults.items():
            db.session.add(SystemSetting(key=key, value=value))

    if not CrisisResource.query.first():
        resources = [
            ("Suicide & Crisis Lifeline", "hotline", "999", "Call or text in an emergency", 1),
            ("Talian Kasih", "hotline", "15999", "Crisis text and call line", 2),
            ("Talian HEAL", "hotline", "15555", "Mental health support hotline", 3),
            ("University Counseling Center", "university", "connect@mmu.edu.my", "General inquiries", 4),
            ("Campus Safety", "university", "03-8312 5482", "Campus security", 5),
            ("MIASA Crisis Helpline", "online", "1-800-18-0066", "Mental health support", 6),
            ("Befrienders Kuala Lumpur", "online", "0376272929", "Emotional support", 7),
            ("Narcotics Anonymous Malaysia", "online", "011-15114022", "Addiction support", 8),
        ]
        for title, category, contact, description, sort_order in resources:
            db.session.add(
                CrisisResource(
                    title=title,
                    category=category,
                    contact=contact,
                    description=description,
                    sort_order=sort_order,
                )
            )

    db.session.commit()


def init_db():
    db.create_all()
    _ensure_user_columns()
    _ensure_chat_columns()
    _ensure_chat_indexes()
    _seed_defaults()


def bootstrap_application():
    """Initialise database and RAG index. Safe to call on every process start."""
    from rag.knowledge_base import build_index, get_index_stats

    with app.app_context():
        init_db()

    force_rebuild = os.getenv("RAG_FORCE_REBUILD", "false").lower() in {"1", "true", "yes"}
    build_index(force_rebuild=force_rebuild)
    return get_index_stats()


def _peak_hour(cutoff=None):
    return _sql_peak_hour(cutoff)


# =========================
# SESSION (ANONYMOUS USER)
# =========================
@app.before_request
def create_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "admin" not in session:
        return redirect("/admin")
    return redirect("/admin?panel=dashboard")


# =========================
# CHAT
# =========================
def _get_recent_history(limit=5):
    """Return the last N conversation turns for the current session."""
    rows = (
        Chat.query.filter_by(session_id=session["session_id"])
        .order_by(Chat.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "user_msg": row.user_msg,
            "bot_reply": row.bot_reply,
            "emotion": row.sentiment,
            "intent": row.intent,
        }
        for row in reversed(rows)
    ]


def _record_login(user):
    user.last_login = datetime.utcnow()


def _update_session_summary(session_id, intent, emotion):
    """Track lightweight session metadata for analytics."""
    if not session_id:
        return
    row = db.session.get(SessionSummary, session_id)
    if row is None:
        row = SessionSummary(session_id=session_id, turn_count=0, summary="")
        db.session.add(row)
    row.turn_count = (row.turn_count or 0) + 1
    row.summary = (
        f"Turns: {row.turn_count}. Last intent: {intent or 'unknown'}. "
        f"Last emotion: {emotion or 'neutral'}."
    )
    row.updated_at = datetime.utcnow()


def _save_chat(user_msg, bot_reply, emotion, confidence, intent, source=None, rag_used=False):
    session_id = session.get("session_id")
    chat = Chat(
        session_id=session_id,
        user_msg=user_msg,
        bot_reply=bot_reply,
        sentiment=emotion,
        confidence=confidence,
        intent=intent,
        source=(source or "template")[:20],
        rag_used=bool(rag_used),
        timestamp=datetime.now(),
    )
    db.session.add(chat)
    _update_session_summary(session_id, intent, emotion)
    db.session.commit()
    return chat


@app.route("/get", methods=["POST"])
@rate_limit(max_per_minute=20)
def get_bot_response():
    settings = chat_settings()
    if settings["maintenance_mode"]:
        return jsonify({"error": "Chat is temporarily unavailable for maintenance."}), 503

    user_msg = (request.form.get("msg") or (request.json or {}).get("msg", "")).strip()
    if not user_msg:
        return jsonify({"error": "Message is required"}), 400

    emotion_result = detect_emotion(user_msg)

    try:
        intent_result = predict_intent(user_msg)
    except (FileNotFoundError, OSError, ValueError):
        intent_result = {"tag": None, "confidence": 0.0}

    intent_label = intent_result.get("tag") or "unclassified"

    history = _get_recent_history(5)

    result = chatbot_response(
        user_msg,
        settings=settings,
        emotion=emotion_result["label"],
        intent_result=intent_result,
        history=history,
        session_id=session["session_id"],
    )
    bot_reply = result["response"]
    _save_chat(
        user_msg,
        bot_reply,
        emotion_result["label"],
        emotion_result["confidence"],
        intent_label,
        source=result.get("source", "template"),
        rag_used=result.get("rag_used", False),
    )

    return jsonify({
        "response": bot_reply,
        "emotion": emotion_result["label"],
        "confidence": emotion_result["confidence"],
        "intent": intent_label,
        "response_source": result.get("source", "template"),
        "conversation_intent": result.get("conversation_intent"),
        "rag_used": result.get("rag_used", False),
        "rag_chunks": result.get("rag_chunks", 0),
    })


@app.route("/reset", methods=["POST"])
def reset_chat():
    session_id = session.get("session_id")
    if session_id:
        clear_session_memory(session_id)
        SessionSummary.query.filter_by(session_id=session_id).delete()
        db.session.commit()
    session.pop("session_id", None)
    return jsonify({"status": "cleared"})


# =========================
# PUBLIC API
# =========================
@app.route("/api/resources", methods=["GET"])
def api_public_resources():
    resources = (
        CrisisResource.query.filter_by(is_active=True)
        .order_by(CrisisResource.sort_order.asc(), CrisisResource.id.asc())
        .all()
    )
    grouped = {"hotline": [], "university": [], "online": []}
    for resource in resources:
        grouped.setdefault(resource.category, []).append(resource.to_dict())
    return jsonify({"resources": [r.to_dict() for r in resources], "grouped": grouped})


@app.route("/api/claude-status", methods=["GET"])
@app.route("/api/llm-status", methods=["GET"])
def api_llm_status():
    """Check whether the Groq LLM integration is active. Use ?test=1 for a live API ping."""
    test = request.args.get("test", "").lower() in {"1", "true", "yes"}
    return jsonify(claude_status(test_api=test))


@app.route("/api/rag-status", methods=["GET"])
def api_rag_status():
    """Check whether the RAG knowledge index is built and ready."""
    return jsonify(get_rag_status())


# =========================
# AUTH API
# =========================
@app.route("/api/session", methods=["GET"])
def api_session():
    if "admin" in session:
        return jsonify({"logged_in": True, "email": session["admin"]})
    return jsonify({"logged_in": False})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not is_user_active(user) or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    if not (user.password.startswith("pbkdf2:") or user.password.startswith("scrypt:")):
        user.set_password(password)

    _record_login(user)
    session["admin"] = email
    db.session.commit()
    return jsonify({"success": True, "email": email})


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    user = User(email=email, role="admin", is_active=True, created_at=datetime.utcnow())
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    session["admin"] = email
    return jsonify({"success": True, "email": email})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop("admin", None)
    return jsonify({"success": True})


@app.route("/api/dashboard", methods=["GET"])
@admin_required
def api_dashboard():
    total = _sql_total()
    buckets = _sql_sentiment_buckets()
    emotion_counts = _sql_emotion_distribution()
    intent_counts = _sql_intent_distribution()
    most_common_emotion = _most_common(emotion_counts)
    most_common_intent = _most_common(intent_counts)
    crisis = _sql_crisis_count()

    today = datetime.now().date()
    active_users = (
        db.session.query(func.count(func.distinct(Chat.session_id)))
        .filter(func.date(Chat.timestamp) == today)
        .scalar()
    ) or 0

    return jsonify({
        "total": total,
        "total_chats": total,
        "positive": buckets["positive"],
        "negative": buckets["negative"],
        "neutral": buckets["neutral"],
        "crisis": crisis,
        "total_crisis_cases": crisis,
        "active_users": active_users,
        "emotion_distribution": emotion_counts,
        "intent_distribution": intent_counts,
        "emotion_breakdown": _counts_to_breakdown(emotion_counts, "emotion"),
        "intent_breakdown": _counts_to_breakdown(intent_counts, "intent"),
        "most_common_emotion": most_common_emotion,
        "most_common_intent": most_common_intent,
        "intent_counts": intent_counts,
        "recent_chats": _recent_chats(10),
        "admin_email": session["admin"],
    })


# =========================
# ANALYTICS
# =========================
@app.route("/api/analytics", methods=["GET"])
@admin_required
def api_analytics():
    period = request.args.get("period", "7d")
    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(period)
    cutoff = datetime.now() - timedelta(days=days) if days else None

    total = _sql_total(cutoff)
    unique_sessions = _sql_unique_sessions(cutoff)
    sentiment = _sql_sentiment_buckets(cutoff)
    intents = _sql_intent_distribution(cutoff)
    daily = _sql_daily_counts(cutoff)
    crisis_by_day = _sql_crisis_by_day(cutoff)

    return jsonify({
        "period": period,
        "total_messages": total,
        "unique_sessions": unique_sessions,
        "avg_messages_per_session": round(total / max(unique_sessions, 1), 1),
        "sentiment": sentiment,
        "intents": intents,
        "intent_breakdown": [
            {"intent": label, "count": count}
            for label, count in sorted(intents.items(), key=lambda item: item[1], reverse=True)
        ],
        "daily_conversations": [{"date": k, "count": v} for k, v in sorted(daily.items())],
        "crisis_trend": [{"date": k, "count": v} for k, v in sorted(crisis_by_day.items())],
        "peak_hour": _sql_peak_hour(cutoff),
    })


@app.route("/api/analytics/export", methods=["GET"])
@admin_required
def export_analytics():
    period = request.args.get("period", "30d")
    days_map = {"7d": 7, "30d": 30, "90d": 90, "all": None}
    days = days_map.get(period, 30)

    query = Chat.query
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        query = query.filter(Chat.timestamp >= cutoff)

    chats = query.order_by(Chat.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "session_id", "user_msg", "bot_reply", "emotion", "confidence",
        "intent", "source", "rag_used", "timestamp",
    ])
    for chat in chats:
        writer.writerow([
            chat.id,
            chat.session_id,
            chat.user_msg,
            chat.bot_reply,
            chat.sentiment,
            chat.confidence if chat.confidence is not None else "",
            _intent_bucket(chat),
            chat.source or "",
            chat.rag_used if chat.rag_used is not None else "",
            chat.timestamp.strftime("%Y-%m-%d %H:%M:%S") if chat.timestamp else "",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=analytics_{period}.csv"},
    )


# =========================
# RESOURCES (ADMIN)
# =========================
@app.route("/api/admin/resources", methods=["GET"])
@admin_required
def list_resources():
    resources = CrisisResource.query.order_by(
        CrisisResource.sort_order.asc(), CrisisResource.id.asc()
    ).all()
    return jsonify({"resources": [r.to_dict() for r in resources]})


@app.route("/api/admin/resources", methods=["POST"])
@admin_required
def create_resource():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    category = (data.get("category") or "").strip()
    contact = (data.get("contact") or "").strip()

    if not title or not category or not contact:
        return jsonify({"error": "Title, category, and contact are required"}), 400

    resource = CrisisResource(
        title=title,
        category=category,
        contact=contact,
        description=(data.get("description") or "").strip(),
        sort_order=int(data.get("sort_order") or 0),
        is_active=data.get("is_active", True),
        updated_at=datetime.utcnow(),
    )
    db.session.add(resource)
    db.session.commit()
    return jsonify({"success": True, "resource": resource.to_dict()}), 201


@app.route("/api/admin/resources/<int:resource_id>", methods=["PUT", "PATCH"])
@admin_required
def update_resource(resource_id):
    resource = CrisisResource.query.get_or_404(resource_id)
    data = request.get_json(silent=True) or {}

    for field in ("title", "category", "contact", "description"):
        if field in data:
            setattr(resource, field, (data[field] or "").strip())
    if "sort_order" in data:
        resource.sort_order = int(data["sort_order"] or 0)
    if "is_active" in data:
        resource.is_active = bool(data["is_active"])

    resource.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "resource": resource.to_dict()})


@app.route("/api/admin/resources/<int:resource_id>", methods=["DELETE"])
@admin_required
def delete_resource(resource_id):
    resource = CrisisResource.query.get_or_404(resource_id)
    resource.is_active = False
    resource.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True})


# =========================
# USER MANAGEMENT
# =========================
@app.route("/api/admin/users", methods=["GET"])
@admin_required
def list_users():
    users = User.query.order_by(User.id.asc()).all()
    return jsonify({"users": [u.to_dict() for u in users]})


@app.route("/api/admin/users", methods=["POST"])
@admin_required
def create_user():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()
    role = (data.get("role") or "admin").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    user = User(email=email, role=role, is_active=True, created_at=datetime.utcnow())
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"success": True, "user": user.to_dict()}), 201


@app.route("/api/admin/users/<int:user_id>", methods=["PUT", "PATCH"])
@admin_required
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json(silent=True) or {}
    current = current_admin_user()

    if "email" in data:
        email = (data["email"] or "").strip()
        if email and email != user.email and User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already in use"}), 409
        if email:
            user.email = email

    if "role" in data:
        user.role = (data["role"] or "admin").strip()

    if "is_active" in data:
        if current and current.id == user.id and not data["is_active"]:
            return jsonify({"error": "You cannot deactivate your own account"}), 400
        user.is_active = bool(data["is_active"])

    if data.get("password"):
        user.set_password(data["password"])

    db.session.commit()
    if current and user.email == session.get("admin"):
        session["admin"] = user.email

    return jsonify({"success": True, "user": user.to_dict()})


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    current = current_admin_user()

    if current and current.id == user.id:
        return jsonify({"error": "You cannot delete your own account"}), 400

    active_admins = User.query.filter_by(is_active=True, role="admin").count()
    if user.is_active and user.role == "admin" and active_admins <= 1:
        return jsonify({"error": "At least one active admin is required"}), 400

    user.is_active = False
    db.session.commit()
    return jsonify({"success": True})


# =========================
# SYSTEM SETTINGS
# =========================
@app.route("/api/admin/change-password", methods=["POST"])
@admin_required
def change_password():
    data = request.get_json(silent=True) or {}
    current_pw = (data.get("current_password") or "").strip()
    new_pw = (data.get("new_password") or "").strip()

    if not current_pw or not new_pw:
        return jsonify({"error": "Current and new password are required"}), 400
    if len(new_pw) < 8:
        return jsonify({"error": "New password must be at least 8 characters"}), 400

    user = current_admin_user()
    if not user or not user.check_password(current_pw):
        return jsonify({"error": "Current password is incorrect"}), 401

    user.set_password(new_pw)
    db.session.commit()
    return jsonify({"success": True, "message": "Password updated successfully"})


@app.route("/api/admin/settings", methods=["GET"])
@admin_required
def get_settings():
    rows = SystemSetting.query.all()
    settings = {row.key: row.value for row in rows}
    return jsonify({"settings": settings})


@app.route("/api/admin/settings", methods=["PUT"])
@admin_required
def update_settings():
    data = request.get_json(silent=True) or {}
    incoming = data.get("settings") if isinstance(data.get("settings"), dict) else data
    allowed = {"crisis_keywords", "crisis_response", "default_response", "maintenance_mode"}

    for key, value in incoming.items():
        if key not in allowed:
            continue
        row = db.session.get(SystemSetting, key)
        if row:
            row.value = str(value)
            row.updated_at = datetime.utcnow()
        else:
            db.session.add(SystemSetting(key=key, value=str(value)))

    db.session.commit()
    rows = SystemSetting.query.all()
    return jsonify({"success": True, "settings": {row.key: row.value for row in rows}})


# =========================
# LEGACY AUTH
# =========================
@app.route("/register", methods=["POST"])
def register():
    email = request.form["email"]
    password = request.form["password"]

    if User.query.filter_by(email=email).first():
        return "User exists"

    user = User(email=email, role="admin", is_active=True, created_at=datetime.utcnow())
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return redirect("/admin")


@app.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]

    user = User.query.filter_by(email=email).first()
    if user and is_user_active(user) and user.check_password(password):
        if not (user.password.startswith("pbkdf2:") or user.password.startswith("scrypt:")):
            user.set_password(password)
        _record_login(user)
        db.session.commit()
        session["admin"] = email
        return redirect("/admin?panel=dashboard")
    return "Login failed"


@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect("/admin")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    rag_stats = bootstrap_application()
    print(
        f"RAG index built: {rag_stats['chunks']} chunks from {rag_stats['files']} files"
    )

    if claude_is_configured():
        print(f"Groq LLM ready (model={os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')})")
    else:
        status = claude_status()
        print(f"LLM inactive — {status['message']}")

    debug = os.getenv("FLASK_DEBUG", "true").lower() in {"1", "true", "yes"}
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug)
