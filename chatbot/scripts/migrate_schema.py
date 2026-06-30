"""Apply schema migrations to the SQLite database (standalone, no Flask imports)."""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "instance" / "chatbot.db"


def _columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _indexes(conn, table):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    )
    return {row[0] for row in cur.fetchall()}


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    if "user" in tables:
        cols = _columns(conn, "user")
        if "last_login" not in cols:
            conn.execute("ALTER TABLE user ADD COLUMN last_login DATETIME")
            print("Added user.last_login")

    if "chat" in tables:
        cols = _columns(conn, "chat")
        if "source" not in cols:
            conn.execute("ALTER TABLE chat ADD COLUMN source VARCHAR(20)")
            print("Added chat.source")
        if "rag_used" not in cols:
            conn.execute("ALTER TABLE chat ADD COLUMN rag_used BOOLEAN DEFAULT 0")
            print("Added chat.rag_used")

        indexes = _indexes(conn, "chat")
        if "ix_chat_session_id" not in indexes:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_chat_session_id ON chat (session_id)"
            )
            print("Created index ix_chat_session_id")
        if "ix_chat_timestamp" not in indexes:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_chat_timestamp ON chat (timestamp)"
            )
            print("Created index ix_chat_timestamp")

    if "session_summary" not in tables:
        conn.execute(
            """
            CREATE TABLE session_summary (
                session_id VARCHAR(100) PRIMARY KEY,
                summary TEXT,
                turn_count INTEGER DEFAULT 0,
                updated_at DATETIME
            )
            """
        )
        print("Created session_summary table")

    conn.commit()
    conn.close()
    print(f"Migration complete: {db_path}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    migrate(path)
