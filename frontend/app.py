import sys

import streamlit as st
import sqlite3
import bcrypt
import joblib
import chromadb
import pandas as pd
import os
import smtplib
import mimetypes
import uuid
from email.message import EmailMessage

from pathlib import Path
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# Portable paths for Windows and cloud deployment
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "helpdesk.db"
MODEL_PATH = BASE_DIR / "models" / "it_helpdesk_classifier_v4.pkl"
VECTOR_DIR = BASE_DIR / "knowledge_base" / "vector_store"

# Ensure required local directories exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
ATTACHMENT_DIR = BASE_DIR / "uploads" / "ticket_attachments"
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB
CONFIDENCE_THRESHOLD = 0.70  # Predictions below 70% require human review.
ALLOWED_ATTACHMENT_EXTENSIONS = {"png", "jpg", "jpeg", "pdf", "doc", "docx", "txt", "log", "csv"}

# SLA targets are calendar hours (not business hours). Adjust to your organisation's policy.
SLA_TARGETS = {
    "High": {"response_hours": 1, "resolution_hours": 8},
    "Medium": {"response_hours": 4, "resolution_hours": 24},
    "Low": {"response_hours": 8, "resolution_hours": 72},
}

st.set_page_config(
    page_title="AI IT Helpdesk",
    page_icon="",
    layout="wide"
)

# ----------------------------------
# AUTHENTICATION
# ----------------------------------


# ------------------------------------
# BACKEND MODULE PATH CONFIGURATION
# ------------------------------------
BACKEND_DIR = BASE_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

def authenticate_user(username, password):
    """
    Authenticate a user using the configured database adapter.
    Compatible with SQLite and PostgreSQL.
    """

    if not username or not password:
        return None

    import database

    try:
        with database.database_connection() as conn:
            user = database.fetch_one(
                conn,
                """
                SELECT user_id, username, password_hash, role
                FROM users
                WHERE LOWER(username) = LOWER(?)
                """,
                (username.strip(),)
            )

        if not user:
            return None

        # Both SQLite Row and PostgreSQL RealDictRow
        # support access using column names.
        stored_hash = user["password_hash"]

        if not stored_hash:
            return None

        valid = bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8")
        )

        if not valid:
            return None

        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"]
        }

    except Exception as exc:
        print(f"Authentication database error: {exc}")
        return None



def login_required():
    return "user" in st.session_state


# ----------------------------------
# LOGIN PAGE
# ----------------------------------

if not login_required():

    st.title(" AI-Based IT Helpdesk")
    st.subheader("Secure Login")

    with st.form("login_form"):

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        login_button = st.form_submit_button(
            "Login",
            type="primary"
        )

        if login_button:

            if not username.strip() or not password:
                st.error("Enter username and password.")

            else:

                user = authenticate_user(
                    username,
                    password
                )

                if user:

                    st.session_state["user"] = user
                    st.session_state["analysis"] = None

                    st.success("Login successful!")
                    st.rerun()

                else:

                    st.error("Invalid username or password.")

    st.stop()


# ----------------------------------
# CURRENT USER
# ----------------------------------

current_user = st.session_state["user"]


# ----------------------------------
# TICKET COMMENTS DATABASE
# ----------------------------------


def send_email_notification(
    recipient,
    subject,
    body,
    ticket_id=None,
    notification_type="TICKET_COMMENT"
):
    """Send Gmail notification and record delivery status in PostgreSQL."""

    import os
    import ssl
    import smtplib
    from pathlib import Path
    from email.message import EmailMessage
    from dotenv import load_dotenv
    import database

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)

    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or username
    port = int(os.getenv("SMTP_PORT", "587"))

    def write_email_log(status, error_type=None, error_message=None):
        """Save email delivery metadata without storing email body or secrets."""

        try:
            with database.database_connection() as conn:
                database.execute_query(
                    conn,
                    """
                    INSERT INTO email_logs (
                        ticket_id,
                        recipient,
                        subject,
                        notification_type,
                        delivery_status,
                        error_type,
                        error_message,
                        sent_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?,
                        CASE WHEN ? = 'SENT'
                             THEN CURRENT_TIMESTAMP
                             ELSE NULL
                        END
                    )
                    """,
                    (
                        int(ticket_id) if ticket_id is not None else None,
                        str(recipient),
                        str(subject or "(No subject)"),
                        str(notification_type),
                        status,
                        error_type,
                        error_message,
                        status
                    )
                )
                conn.commit()

            return True

        except Exception as log_error:
            print(
                "Email log database error:",
                type(log_error).__name__
            )
            return False

    if not recipient:
        return False, "Email recipient is missing."

    if not all([host, username, password, sender]):
        write_email_log(
            "FAILED",
            "ConfigurationError",
            "SMTP configuration is incomplete."
        )
        return False, "Email configuration is incomplete."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    try:
        context = ssl.create_default_context()

        with smtplib.SMTP(host, port, timeout=20) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(username, password)
            server.send_message(message)

        logged = write_email_log("SENT")

        if not logged:
            print("Warning: Email sent but logging failed.")

        return True, "Email notification sent successfully."

    except Exception as error:
        error_type = type(error).__name__

        write_email_log(
            "FAILED",
            error_type,
            "SMTP delivery failed."
        )

        print("Email notification failed:", error_type)

        return False, f"Email notification failed: {error_type}"



def notify_ticket_comment(ticket_id, sender_user_id, comment):
    """Notify ticket owner/admins and record notification results."""

    import database

    recipients = set()
    notification_results = []

    with database.database_connection() as conn:

        ticket = database.fetch_one(
            conn,
            """
            SELECT t.title, t.user_id, u.email
            FROM tickets t
            LEFT JOIN users u ON u.user_id = t.user_id
            WHERE t.ticket_id = ?
            """,
            (int(ticket_id),)
        )

        if ticket is None:
            return []

        ticket = dict(ticket)

        sender = database.fetch_one(
            conn,
            "SELECT username, role FROM users WHERE user_id = ?",
            (int(sender_user_id),)
        )

        if sender is None:
            return []

        sender = dict(sender)

        if sender["role"] in ("admin", "support"):

            owner_id = ticket["user_id"]
            owner_email = ticket["email"]

            if owner_email and int(owner_id or 0) != int(sender_user_id):
                recipients.add(owner_email)

            notification_type = "ADMIN_REPLY"

        else:

            admin_rows = database.fetch_all(
                conn,
                """
                SELECT email
                FROM users
                WHERE role = 'admin'
                  AND email IS NOT NULL
                """
            )

            recipients.update(
                dict(row)["email"]
                for row in admin_rows
                if dict(row)["email"]
            )

            notification_type = "USER_COMMENT"

    for recipient in recipients:

        success, result_message = send_email_notification(
            recipient,
            f"New reply on Helpdesk Ticket #{ticket_id}",
            (
                f"Ticket: {ticket['title']}\n"
                f"From: {sender['username']} ({sender['role']})\n\n"
                f"Message: {comment}\n\n"
                "Please sign in to the AI IT Helpdesk "
                "to view the conversation."
            ),
            ticket_id=ticket_id,
            notification_type=notification_type
        )

        notification_results.append({
            "recipient": recipient,
            "status": "SENT" if success else "FAILED",
            "message": result_message
        })

    return notification_results



def ensure_sla_columns():
    """
    Validate required SLA columns.
    Schema changes are managed separately in Supabase.
    """
    import database

    required_columns = {
        "response_due_at",
        "resolution_due_at",
        "first_response_at",
        "resolved_at"
    }

    if database.get_database_type() == "postgres":
        with database.database_connection() as conn:
            rows = database.fetch_all(
                conn,
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'tickets'
                """
            )

        existing = {row["column_name"] for row in rows}

    else:
        with database.database_connection() as conn:
            rows = database.fetch_all(
                conn,
                "PRAGMA table_info(tickets)"
            )

        existing = {row["name"] for row in rows}

    missing = required_columns - existing

    if missing:
        raise RuntimeError(
            f"Missing SLA columns: {sorted(missing)}"
        )

    print("SLA columns verified successfully.")

def calculate_sla_deadlines(priority, created_at=None):
    created = created_at or datetime.utcnow()
    target = SLA_TARGETS.get(priority, SLA_TARGETS["Medium"])
    return (
        (created + timedelta(hours=target["response_hours"])).strftime("%Y-%m-%d %H:%M:%S"),
        (created + timedelta(hours=target["resolution_hours"])).strftime("%Y-%m-%d %H:%M:%S")
    )

def sla_state(due_at, completed_at=None, now=None):
    """Return an SLA state safely when SQLite/pandas values are strings or NULL/NaN."""
    due = pd.to_datetime(due_at, errors="coerce")

    if pd.isna(due):
        return "Not Set"

    completed = pd.to_datetime(completed_at, errors="coerce")
    if not pd.isna(completed):
        return "Met" if completed <= due else "Breached"

    current = pd.to_datetime(now, errors="coerce") if now is not None else pd.Timestamp.now()
    if pd.isna(current):
        current = pd.Timestamp.now()

    return "Overdue" if current > due else "Within SLA"


def ensure_ai_prediction_table():
    """
    Validate the existing AI prediction table.
    Does not create, drop or modify tables.
    """
    import database

    required_columns = {
        "prediction_id",
        "user_id",
        "user_query",
        "predicted_category",
        "confidence",
        "confidence_decision",
        "review_status",
        "reviewed_category",
        "reviewed_by",
        "reviewed_at",
        "created_at"
    }

    if database.get_database_type() == "postgres":
        with database.database_connection() as conn:
            rows = database.fetch_all(
                conn,
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'ai_predictions'
                """
            )

        existing = {row["column_name"] for row in rows}

    else:
        with database.database_connection() as conn:
            rows = database.fetch_all(
                conn,
                "PRAGMA table_info(ai_predictions)"
            )

        existing = {row["name"] for row in rows}

    missing = required_columns - existing

    if missing:
        raise RuntimeError(
            f"Missing ai_predictions columns: {sorted(missing)}"
        )

    print("AI prediction table verified successfully.")


def save_ai_prediction(user_id, user_query, predicted_category, confidence, decision):
    import database

    sql = """
        INSERT INTO ai_predictions
            (user_id, user_query, predicted_category, confidence, confidence_decision)
        VALUES (?, ?, ?, ?, ?)
    """

    params = (
        int(user_id),
        user_query,
        predicted_category,
        float(confidence),
        decision
    )

    with database.database_connection() as conn:
        if database.get_database_type() == "postgres":
            cursor = database.execute_query(
                conn,
                sql + " RETURNING prediction_id",
                params
            )
            row = cursor.fetchone()
            return row["prediction_id"]

        cursor = database.execute_query(conn, sql, params)
        return cursor.lastrowid

def load_human_review_queue():
    """Load AI predictions requiring human review."""
    import database

    with database.database_connection() as conn:
        rows = database.fetch_all(
            conn,
            """
            SELECT
                p.prediction_id,
                p.user_query,
                p.predicted_category,
                ROUND((p.confidence * 100)::numeric, 2)
                AS confidence_percent,
                p.review_status,
                p.created_at,
                u.username
            FROM ai_predictions p
            LEFT JOIN users u
                ON u.user_id = p.user_id
            WHERE p.confidence_decision = ?
            ORDER BY p.prediction_id DESC
            """,
            ("HUMAN_REVIEW_REQUIRED",)
        )

    return pd.DataFrame([dict(row) for row in rows])

def export_human_review_feedback():
    """Export reviewed predictions for model training."""
    import database

    with database.database_connection() as conn:
        rows = database.fetch_all(
            conn,
            """
            SELECT
                prediction_id,
                user_query,
                reviewed_category AS category
            FROM ai_predictions
            WHERE review_status = ?
              AND reviewed_category IS NOT NULL
            ORDER BY prediction_id
            """,
            ("Reviewed",)
        )

    feedback = pd.DataFrame([dict(row) for row in rows])

    if feedback.empty:
        return pd.DataFrame(
            columns=["prediction_id", "user_query", "category"]
        )

    feedback = feedback.dropna(
        subset=["user_query", "category"]
    ).copy()

    feedback["user_query"] = (
        feedback["user_query"].astype(str).str.strip()
    )
    feedback["category"] = (
        feedback["category"].astype(str).str.strip()
    )

    feedback = feedback[
        (feedback["user_query"] != "") &
        (feedback["category"] != "")
    ]

    feedback = feedback.drop_duplicates(
        subset=["user_query"],
        keep="last"
    )

    return feedback[
        ["prediction_id", "user_query", "category"]
    ]

    feedback = feedback.dropna(subset=["user_query", "category"]).copy()
    feedback["user_query"] = feedback["user_query"].astype(str).str.strip()
    feedback["category"] = feedback["category"].astype(str).str.strip()
    feedback = feedback[
        (feedback["user_query"] != "") &
        (feedback["category"] != "")
    ]
    feedback = feedback.drop_duplicates(subset=["user_query"], keep="last")
    return feedback[["prediction_id", "user_query", "category"]]


ensure_ai_prediction_table()
ensure_sla_columns()


def ensure_ticket_comments_table():
    """Create ticket comments table for SQLite or PostgreSQL."""
    import database

    if database.get_database_type() == "postgres":
        sql = """
            CREATE TABLE IF NOT EXISTS ticket_comments (
                comment_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                ticket_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                comment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """
    else:
        sql = """
            CREATE TABLE IF NOT EXISTS ticket_comments (
                comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                comment TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """

    with database.database_connection() as conn:
        database.execute_query(conn, sql)



ensure_ticket_comments_table()


def ensure_ticket_attachments_table():
    """Create ticket attachment metadata table."""
    import database

    if database.get_database_type() == "postgres":
        sql = """
            CREATE TABLE IF NOT EXISTS ticket_attachments (
                attachment_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                ticket_id INTEGER NOT NULL,
                uploaded_by INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL,
                mime_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
                FOREIGN KEY (uploaded_by) REFERENCES users(user_id)
            )
        """
    else:
        sql = """
            CREATE TABLE IF NOT EXISTS ticket_attachments (
                attachment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER NOT NULL,
                uploaded_by INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL,
                mime_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
                FOREIGN KEY (uploaded_by) REFERENCES users(user_id)
            )
        """

    with database.database_connection() as conn:
        database.execute_query(conn, sql)



def save_ticket_attachment(ticket_id, uploaded_file, user_id):
    """Validate, store and register an attachment for an accessible ticket."""
    import database

    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower().lstrip(".")
    content = uploaded_file.getvalue()

    if extension not in ALLOWED_ATTACHMENT_EXTENSIONS:
        raise ValueError(
            "File type not allowed. Upload PNG, JPG, PDF, DOCX, DOC, TXT, LOG or CSV."
        )

    if not content:
        raise ValueError("The selected file is empty.")

    if len(content) > MAX_ATTACHMENT_SIZE:
        raise ValueError("File is larger than the 10 MB limit.")

    stored_name = f"{uuid.uuid4().hex}.{extension}"

    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ATTACHMENT_DIR / stored_name

    try:
        with database.database_connection() as conn:

            ticket = database.fetch_one(
                conn,
                """
                SELECT user_id, title, assigned_to
                FROM tickets
                WHERE ticket_id = ?
                """,
                (int(ticket_id),)
            )

            if ticket is None:
                raise ValueError("Ticket not found.")

            is_owner = (
                int(ticket["user_id"] or 0) == int(user_id)
            )

            is_assigned_support = (
                current_user["role"] == "support"
                and int(ticket["assigned_to"] or 0) == int(user_id)
            )

            if (
                current_user["role"] != "admin"
                and not is_owner
                and not is_assigned_support
            ):
                raise PermissionError(
                    "You are not allowed to attach files to this ticket."
                )

            destination.write_bytes(content)

            mime_type = (
                uploaded_file.type
                or mimetypes.guess_type(original_name)[0]
                or "application/octet-stream"
            )

            database.execute_query(
                conn,
                """
                INSERT INTO ticket_attachments
                    (ticket_id, uploaded_by, original_name,
                     stored_name, file_size, mime_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(ticket_id),
                    int(user_id),
                    original_name,
                    stored_name,
                    len(content),
                    mime_type
                )
            )

            database.execute_query(
                conn,
                """
                INSERT INTO audit_logs
                    (user_id, ticket_id, action, details)
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    int(ticket_id),
                    "TICKET_ATTACHMENT_UPLOADED",
                    f"Attachment uploaded: {original_name}"
                )
            )

        return stored_name

    except Exception:
        if destination.exists():
            destination.unlink()
        raise



def load_ticket_attachments(ticket_id):
    """Load attachment metadata for a ticket."""
    import database

    with database.database_connection() as conn:
        rows = database.fetch_all(
            conn,
            """
            SELECT
                a.attachment_id,
                a.original_name,
                a.stored_name,
                a.file_size,
                a.mime_type,
                a.created_at,
                u.username
            FROM ticket_attachments a
            LEFT JOIN users u
                ON u.user_id = a.uploaded_by
            WHERE a.ticket_id = ?
            ORDER BY a.attachment_id DESC
            """,
            (int(ticket_id),)
        )

    columns = [
        "attachment_id",
        "original_name",
        "stored_name",
        "file_size",
        "mime_type",
        "created_at",
        "username"
    ]

    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=columns
    )



ensure_ticket_attachments_table()


def save_ticket_comment(ticket_id, user_id, comment):
    """Save a ticket message and audit entry transactionally."""

    import database

    clean_comment = comment.strip()

    if not clean_comment:
        raise ValueError("Comment cannot be empty.")

    with database.database_connection() as conn:

        ticket = database.fetch_one(
            conn,
            """
            SELECT user_id, title, assigned_to
            FROM tickets
            WHERE ticket_id = ?
            """,
            (int(ticket_id),)
        )

        if ticket is None:
            raise ValueError("Ticket not found.")

        ticket = dict(ticket)

        is_owner = int(ticket["user_id"] or 0) == int(user_id)

        is_assigned_support = (
            current_user["role"] == "support"
            and int(ticket["assigned_to"] or 0) == int(user_id)
        )

        if (
            current_user["role"] != "admin"
            and not is_owner
            and not is_assigned_support
        ):
            raise PermissionError(
                "You are not allowed to comment on this ticket."
            )

        database.execute_query(
            conn,
            """
            INSERT INTO ticket_comments
                (ticket_id, user_id, comment)
            VALUES (?, ?, ?)
            """,
            (int(ticket_id), int(user_id), clean_comment)
        )

        database.execute_query(
            conn,
            """
            INSERT INTO audit_logs
                (user_id, ticket_id, action, details)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(user_id),
                int(ticket_id),
                "TICKET_COMMENTED",
                f"Comment added to ticket: {ticket['title']}"
            )
        )

        role_row = database.fetch_one(
            conn,
            "SELECT role FROM users WHERE user_id = ?",
            (int(user_id),)
        )

        if role_row and dict(role_row)["role"] in ("admin", "support"):

            database.execute_query(
                conn,
                """
                UPDATE tickets
                SET first_response_at =
                    COALESCE(first_response_at, CURRENT_TIMESTAMP)
                WHERE ticket_id = ?
                """,
                (int(ticket_id),)
            )



def load_ticket_comments(ticket_id):
    """Load ticket conversation using the configured database."""

    import database

    with database.database_connection() as conn:

        rows = database.fetch_all(
            conn,
            """
            SELECT
                c.comment_id,
                c.ticket_id,
                c.user_id,
                u.username,
                u.role,
                c.comment,
                c.created_at
            FROM ticket_comments c
            LEFT JOIN users u ON c.user_id = u.user_id
            WHERE c.ticket_id = ?
            ORDER BY c.comment_id ASC
            """,
            (int(ticket_id),)
        )

        return pd.DataFrame([dict(row) for row in rows])



st.sidebar.title("Helpdesk Menu")

st.sidebar.write(
    f"Welcome, {current_user['username']}"
)

st.sidebar.caption(
    f"Role: {current_user['role']}"
)

if st.sidebar.button("Logout"):

    for key in ["user", "analysis"]:
        st.session_state.pop(key, None)

    st.rerun()


# ----------------------------------
# ROLE-BASED NAVIGATION
# ----------------------------------

role = current_user["role"]

if role == "admin":
    menu_options = [
        "AI Helpdesk",
        "My Tickets",
        "Admin Dashboard"
    ]

elif role == "support":
    menu_options = [
        "AI Helpdesk",
        "My Tickets",
        "Support Dashboard"
    ]

else:
    menu_options = [
        "AI Helpdesk",
        "My Tickets"
    ]

menu = st.sidebar.radio(
    "Navigation",
    menu_options
)

st.title("AI-Based IT Helpdesk")


# ----------------------------------
# LOAD AI COMPONENTS
# ----------------------------------
@st.cache_resource
def load_ai_components(model_path, vector_dir, model_version):

    import pandas as pd

    # 1. Load existing V4 classifier
    classifier = joblib.load(model_path)

    # 2. Initialize embedding model
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    # 3. Initialize ChromaDB
    client = chromadb.PersistentClient(
        path=str(vector_dir)
    )

    # 4. Create or retrieve collection
    collection = client.get_or_create_collection(
        name="it_helpdesk_knowledge"
    )

    # 5. Load knowledge base CSV
    kb_path = BASE_DIR / "knowledge_base" / "it_knowledge_base.csv"

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base CSV not found: {kb_path}"
        )

    kb_df = pd.read_csv(kb_path).fillna("")

    required_columns = [
        "id", "category", "title", "content", "source"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in kb_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing knowledge-base columns: {missing_columns}"
        )

    # 6. Import records only if collection is empty
    if collection.count() == 0:

        documents = (
            kb_df["title"].astype(str)
            + "\n"
            + kb_df["content"].astype(str)
        ).tolist()

        embeddings = embedding_model.encode(
            documents,
            convert_to_numpy=True
        ).tolist()

        metadatas = [
            {
                "category": str(row["category"]),
                "title": str(row["title"]),
                "source": str(row["source"])
            }
            for _, row in kb_df.iterrows()
        ]

        collection.add(
            ids=kb_df["id"].astype(str).tolist(),
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(
        "ChromaDB records loaded:",
        collection.count()
    )

    return classifier, collection, embedding_model



# Include model file modification time in the cache key.
# This forces a reload when the model file changes.
classifier, collection, embedding_model = load_ai_components(
    str(MODEL_PATH),
    str(VECTOR_DIR),
    MODEL_PATH.stat().st_mtime_ns
)

st.sidebar.caption(
    f"Loaded model: {MODEL_PATH.name}"
)

# ----------------------------------
# AI FUNCTIONS
# ----------------------------------

def classify_issue(query):

    category = classifier.predict([query])[0]

    confidence = None

    if hasattr(classifier, "predict_proba"):

        probabilities = classifier.predict_proba([query])[0]

        confidence = float(max(probabilities))

    return category, confidence


def retrieve_knowledge(query, category, top_k=3):

    embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        where={"category": category}
    )

    return (
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0]
    )


# ----------------------------------
# CREATE TICKET
# ----------------------------------

def create_support_ticket(
    title,
    description,
    category_name,
    priority,
    user_id
):
    import database

    conn = database.get_connection()

    try:
        category = database.fetch_one(
            conn,
            """
            SELECT category_id
            FROM categories
            WHERE LOWER(category_name) = LOWER(?)
            """,
            (category_name,)
        )

        if category is None:
            raise ValueError("Invalid ticket category.")

        deadlines = calculate_sla_deadlines(priority)

        ticket_row = database.fetch_one(
            conn,
            """
            INSERT INTO tickets
            (
                user_id,
                title,
                description,
                category_id,
                priority,
                status,
                response_due_at,
                resolution_due_at
            )
            VALUES (?, ?, ?, ?, ?, 'Open', ?, ?)
            RETURNING ticket_id
            """,
            (
                user_id,
                title,
                description,
                category["category_id"],
                priority,
                deadlines[0],
                deadlines[1]
            )
        )

        if ticket_row is None:
            raise RuntimeError("Ticket creation failed.")

        ticket_id = ticket_row["ticket_id"]

        database.execute_query(
            conn,
            """
            INSERT INTO audit_logs
            (user_id, ticket_id, action, details)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                ticket_id,
                "TICKET_CREATED",
                f"Created by {current_user['username']}: {title}"
            )
        )

        conn.commit()
        return ticket_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

# ----------------------------------
# AI HELPDESK
# ----------------------------------

if menu == "AI Helpdesk":

    st.subheader("How can we help you?")

    query = st.text_area(
        "Describe your IT issue:",
        height=120
    )

    if st.button("Analyse Issue", type="primary"):

        if not query.strip():

            st.warning("Enter your issue.")

        else:

            with st.spinner("AI analysing issue..."):

                category, confidence = classify_issue(query)

                decision = (
                    "AI_CLASSIFIED"
                    if confidence is not None and confidence >= CONFIDENCE_THRESHOLD
                    else "HUMAN_REVIEW_REQUIRED"
                )

                documents, metadatas = retrieve_knowledge(
                    query,
                    category
                )

                prediction_id = save_ai_prediction(
                    current_user["user_id"],
                    query.strip(),
                    category,
                    confidence,
                    decision
                )

            st.session_state["analysis"] = {
                "query": query,
                "category": category,
                "confidence": confidence,
                "decision": decision,
                "prediction_id": prediction_id,
                "documents": documents,
                "metadatas": metadatas
            }

    analysis = st.session_state.get("analysis")

    if analysis:

        st.success("Issue analysis completed!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Category",
            analysis["category"]
        )

        if analysis["confidence"] is not None:

            col2.metric(
                "Classification Confidence",
                f"{analysis['confidence']:.2%}"
            )

        st.caption(
            f"Prediction ID: {analysis.get('prediction_id', 'N/A')} | "
            f"Review threshold: {CONFIDENCE_THRESHOLD:.0%}"
        )
        if analysis.get("decision") == "HUMAN_REVIEW_REQUIRED":
            st.warning(
                "Human review required: confidence is below 70%. "
                "The predicted category is a suggestion and should be checked by IT support."
            )
        else:
            st.success("AI classification accepted by the configured confidence threshold.")

        st.subheader("Troubleshooting Guidance")

        for i, document in enumerate(analysis["documents"]):

            title = analysis["metadatas"][i].get(
                "title",
                f"Article {i+1}"
            )

            with st.expander(title):
                st.write(document)

        st.subheader("Create Support Ticket")

        with st.form("ticket_form"):

            title = st.text_input(
                "Ticket Title",
                value=analysis["query"][:100]
            )

            description = st.text_area(
                "Description",
                value=analysis["query"]
            )

            categories = [
                "Account & Password",
                "Network & VPN",
                "Printer",
                "Software",
                "System Health"
            ]

            predicted = analysis["category"]

            category_index = (
                categories.index(predicted)
                if predicted in categories else 0
            )

            category = st.selectbox(
                "Category",
                categories,
                index=category_index
            )

            priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High"],
                index=1
            )

            submit = st.form_submit_button(
                "Create Support Ticket"
            )

            if submit:

                if not title.strip() or not description.strip():

                    st.error("Complete all required fields.")

                else:

                    ticket_id = create_support_ticket(
                        title.strip(),
                        description.strip(),
                        category,
                        priority,
                        current_user["user_id"]
                    )

                    st.success(
                        f"Ticket created successfully! ID: {ticket_id}"
                    )


# ----------------------------------
# MY TICKETS
# ----------------------------------

elif menu == "My Tickets":

    st.subheader("My Support Tickets")

 # Use the configured database adapter (SQLite / PostgreSQL)
    import database

    if current_user["role"] == "admin":

        query = """
            SELECT
                t.ticket_id,
                t.title,
                c.category_name,
                t.priority,
                t.status,
                t.created_at,
                t.response_due_at,
                t.resolution_due_at,
                t.first_response_at,
                t.resolved_at
            FROM tickets t
            LEFT JOIN categories c
                ON t.category_id = c.category_id
            ORDER BY t.ticket_id DESC
        """

        params = ()

    else:

        query = """
            SELECT
                t.ticket_id,
                t.title,
                c.category_name,
                t.priority,
                t.status,
                t.created_at,
                t.response_due_at,
                t.resolution_due_at,
                t.first_response_at,
                t.resolved_at
            FROM tickets t
            LEFT JOIN categories c
                ON t.category_id = c.category_id
            WHERE t.user_id = ?
            ORDER BY t.ticket_id DESC
        """

        params = (int(current_user["user_id"]),)

    # Execute using the configured database connection
    with database.database_connection() as conn:

        ticket_rows = database.fetch_all(
            conn,
            query,
            params
        )

        tickets = pd.DataFrame(
            [dict(row) for row in ticket_rows]
        )

    print("My Tickets database:", database.get_database_type())
    print("Logged-in user:", current_user["username"])
    print("User ID:", current_user["user_id"])
    print("Tickets retrieved:", len(tickets))

    st.dataframe(
        tickets,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader(" Ticket Conversation")

    if tickets.empty:
        st.info("You do not have any support tickets yet.")
    else:
        ticket_choices = tickets["ticket_id"].astype(int).tolist()

        selected_conversation_ticket = st.selectbox(
            "Select a ticket to view its conversation",
            ticket_choices,
            format_func=lambda ticket_id: (
                f"Ticket #{ticket_id} - "
                + str(tickets.loc[
                    tickets["ticket_id"] == ticket_id, "title"
                ].iloc[0])
            ),
            key="conversation_ticket_selector"
        )

        selected_ticket_row = tickets[
            tickets["ticket_id"] == selected_conversation_ticket
        ].iloc[0]

        st.caption(
            f"Status: {selected_ticket_row['status']} | "
            f"Priority: {selected_ticket_row['priority']}"
        )

        conversation = load_ticket_comments(
            selected_conversation_ticket
        )

        if conversation.empty:
            st.info("No messages yet. Start the conversation below.")
        else:
            for _, message in conversation.iterrows():
                sender = message["username"] or "Unknown user"
                sender_role = message["role"] or "user"

                with st.chat_message(
                    "assistant" if sender_role == "admin" else "user"
                ):
                    st.markdown(
                        f"**{sender}**  {sender_role}  "
                        f"{message['created_at']}"
                    )
                    st.write(message["comment"])

        with st.form(
            f"ticket_reply_form_{selected_conversation_ticket}",
            clear_on_submit=True
        ):
            reply_text = st.text_area(
                "Write a message",
                placeholder="Describe your update or reply...",
                max_chars=3000
            )
            send_reply = st.form_submit_button(
                "Send Message",
                type="primary"
            )

            if send_reply:
                if not reply_text.strip():
                    st.warning("Please enter a message before sending.")
                else:
                    try:
                        save_ticket_comment(
                            selected_conversation_ticket,
                            current_user["user_id"],
                            reply_text
                        )
                        notify_ticket_comment(
                            selected_conversation_ticket,
                            current_user["user_id"],
                            reply_text.strip()
                        )
                        st.success("Message sent successfully.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Unable to send message: {error}")

        st.markdown("####  Ticket Attachments")
        st.caption("Allowed: PNG, JPG, JPEG, PDF, DOC, DOCX, TXT, LOG and CSV. Maximum size: 10 MB per file.")
        with st.form(f"attachment_form_{selected_conversation_ticket}"):
            attachment = st.file_uploader(
                "Choose a file to attach",
                type=sorted(ALLOWED_ATTACHMENT_EXTENSIONS),
                key=f"attachment_uploader_{selected_conversation_ticket}"
            )
            upload_attachment = st.form_submit_button("Upload Attachment")

            if upload_attachment:
                if attachment is None:
                    st.warning("Please choose a file first.")
                else:
                    try:
                        save_ticket_attachment(
                            selected_conversation_ticket,
                            attachment,
                            current_user["user_id"]
                        )
                        st.success("Attachment uploaded successfully.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Unable to upload attachment: {error}")

        attachments = load_ticket_attachments(selected_conversation_ticket)
        if attachments.empty:
            st.info("No attachments have been uploaded for this ticket.")
        else:
            for _, attachment_row in attachments.iterrows():
                stored_name = Path(str(attachment_row["stored_name"])).name
                file_path = (ATTACHMENT_DIR / stored_name).resolve()
                safe_root = ATTACHMENT_DIR.resolve()
                if safe_root not in file_path.parents or not file_path.is_file():
                    st.warning(f"File is missing or unavailable: {attachment_row['original_name']}")
                    continue

                size_kb = int(attachment_row["file_size"]) / 1024
                label = (
                    f"{attachment_row['original_name']}  "
                    f"{size_kb:.1f} KB  uploaded by {attachment_row['username'] or 'Unknown'}  "
                    f"{attachment_row['created_at']}"
                )
                with file_path.open("rb") as file_handle:
                    st.download_button(
                        label=f" {label}",
                        data=file_handle.read(),
                        file_name=attachment_row["original_name"],
                        mime=attachment_row["mime_type"] or "application/octet-stream",
                        key=f"download_attachment_{attachment_row['attachment_id']}"
                    )


# ----------------------------------
#  ADMIN DASHBOARD
# ----------------------------------




elif menu == "Admin Dashboard":

    if current_user["role"] != "admin":
        st.error("Access denied.")
        st.stop()

    st.subheader("  IT Support Dashboard")

    st.divider()
    st.header("AI Human Review Queue")
    review_queue = load_human_review_queue()
    
    pending_reviews = review_queue[review_queue["review_status"] == "Pending"] if not review_queue.empty else review_queue
    st.metric("Pending AI Reviews", len(pending_reviews))

    if review_queue.empty:
        st.info("No AI predictions have been routed for human review yet.")
    else:
        st.dataframe(review_queue, use_container_width=True, hide_index=True)
        if not pending_reviews.empty:
            review_ids = pending_reviews["prediction_id"].astype(int).tolist()
            selected_review_id = st.selectbox(
                "Select prediction to review",
                review_ids,
                key="ai_review_prediction_selector"
            )
            selected_review = pending_reviews[
                pending_reviews["prediction_id"] == selected_review_id
            ].iloc[0]
            st.write("**Original query:**", selected_review["user_query"])
            st.write("**AI suggested category:**", selected_review["predicted_category"])
            review_categories = [
                "Account & Password", "Network & VPN", "Printer",
                "Software", "System Health"
            ]
            default_category = (
                selected_review["predicted_category"]
                if selected_review["predicted_category"] in review_categories else review_categories[0]
            )
            with st.form(f"ai_human_review_form_{selected_review_id}"):
                corrected_category = st.selectbox(
                    "Confirm or correct category",
                    review_categories,
                    index=review_categories.index(default_category)
                )
                submit_review = st.form_submit_button("Save Human Review", type="primary")
            if submit_review:
                import database

                with database.database_connection() as review_conn:
                    cursor = database.execute_query(
                        review_conn,
                        """
                        UPDATE ai_predictions
                        SET review_status = 'Reviewed',
                            reviewed_category = ?,
                            reviewed_by = ?,
                            reviewed_at = CURRENT_TIMESTAMP
                        WHERE prediction_id = ?
                          AND confidence_decision = 'HUMAN_REVIEW_REQUIRED'
                          AND review_status = 'Pending'
                        """,
                        (
                            corrected_category,
                            int(current_user["user_id"]),
                            int(selected_review_id)
                        )
                    )

                    updated_count = cursor.rowcount

                if updated_count == 1:
                    st.success("Human review saved successfully.")
                    st.rerun()
                else:
                    st.warning('No pending prediction was updated. Please refresh the review queue.')


    st.divider()
    st.subheader("Human Review Feedback Export")
    feedback_export = export_human_review_feedback()
    st.metric("Reviewed Feedback Records", len(feedback_export))

    if not feedback_export.empty:
        csv_data = feedback_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Human Review Feedback CSV",
            data=csv_data,
            file_name="human_review_feedback.csv",
            mime="text/csv",
            key="download_human_review_feedback"
        )
        st.dataframe(
            feedback_export,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No reviewed feedback records available.")


    # PostgreSQL / SQLite compatible database access
    import database

    with database.database_connection() as conn:

        # KPI SUMMARY
        total_row = database.fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM tickets"
        )

        open_row = database.fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM tickets WHERE status = 'Open'"
        )

        progress_row = database.fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM tickets WHERE status = 'In Progress'"
        )

        resolved_row = database.fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM tickets WHERE status = 'Resolved'"
        )

        high_row = database.fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM tickets WHERE priority = 'High'"
        )

        total = int(total_row["total"])
        open_count = int(open_row["total"])
        progress_count = int(progress_row["total"])
        resolved_count = int(resolved_row["total"])
        high_count = int(high_row["total"])

        # SLA MONITORING
        sla_rows = database.fetch_all(
            conn,
            """
            SELECT
                ticket_id,
                title,
                priority,
                status,
                response_due_at,
                resolution_due_at,
                first_response_at,
                resolved_at
            FROM tickets
            WHERE status != 'Resolved' OR resolved_at IS NOT NULL
            ORDER BY ticket_id DESC
            """
        )

        sla_columns = [
            "ticket_id", "title", "priority", "status",
            "response_due_at", "resolution_due_at",
            "first_response_at", "resolved_at"
        ]

        sla_df = pd.DataFrame(
            [dict(row) for row in sla_rows],
            columns=sla_columns
        )

        # TICKET MANAGEMENT
        ticket_rows = database.fetch_all(
            conn,
            """
            SELECT
                t.ticket_id,
                t.title,
                t.description,
                u.username AS created_by,
                a.username AS assigned_to,
                c.category_name,
                t.priority,
                t.status,
                t.created_at,
                t.updated_at,
                t.response_due_at,
                t.resolution_due_at,
                t.first_response_at,
                t.resolved_at
            FROM tickets t
            LEFT JOIN users u
                ON t.user_id = u.user_id
            LEFT JOIN users a
                ON t.assigned_to = a.user_id
            LEFT JOIN categories c
                ON t.category_id = c.category_id
            ORDER BY t.ticket_id DESC
            """
        )

        ticket_columns = [
            "ticket_id", "title", "description",
            "created_by", "assigned_to", "category_name",
            "priority", "status", "created_at", "updated_at",
            "response_due_at", "resolution_due_at",
            "first_response_at", "resolved_at"
        ]

        tickets = pd.DataFrame(
            [dict(row) for row in ticket_rows],
            columns=ticket_columns
        )

    # DISPLAY KPI SUMMARY
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tickets", total)
    c2.metric("Open Tickets", open_count)
    c3.metric("In Progress", progress_count)

    c4, c5 = st.columns(2)
    c4.metric("Resolved Tickets", resolved_count)
    c5.metric("High Priority", high_count)

    st.divider()
    st.subheader(" SLA Monitoring")

    now_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    if not sla_df.empty:
        sla_df["Response SLA"] = sla_df.apply(
            lambda r: sla_state(
                r["response_due_at"],
                r["first_response_at"],
                now_text
            ),
            axis=1
        )

        sla_df["Resolution SLA"] = sla_df.apply(
            lambda r: sla_state(
                r["resolution_due_at"],
                r["resolved_at"],
                now_text
            ),
            axis=1
        )

        sla_cols = [
            "ticket_id", "title", "priority", "status",
            "response_due_at", "Response SLA",
            "resolution_due_at", "Resolution SLA"
        ]

        st.dataframe(
            sla_df[sla_cols],
            use_container_width=True,
            hide_index=True
        )

        response_breaches = int(
            sla_df["Response SLA"].isin(["Overdue", "Breached"]).sum()
        )

        resolution_breaches = int(
            sla_df["Resolution SLA"].isin(["Overdue", "Breached"]).sum()
        )

        m1, m2 = st.columns(2)
        m1.metric("Response SLA breaches", response_breaches)
        m2.metric("Resolution SLA breaches", resolution_breaches)

    else:
        st.info("No tickets available for SLA monitoring.")

    st.divider()

    st.subheader(" Ticket Management")

    # SEARCH
    search_text = st.text_input(
        "Search by Ticket ID, Title or Username"
    )

    # FILTERS
    f1, f2, f3 = st.columns(3)

    status_filter = f1.selectbox(
        "Filter by Status",
        ["All", "Open", "In Progress", "Resolved"]
    )

    priority_filter = f2.selectbox(
        "Filter by Priority",
        ["All", "Low", "Medium", "High"]
    )

    category_options = ["All"] + sorted(
        tickets["category_name"].dropna().unique().tolist()
    )

    category_filter = f3.selectbox(
        "Filter by Category",
        category_options
    )

    # DATE FILTER
    from datetime import date, timedelta

    d1, d2 = st.columns(2)

    start_date = d1.date_input(
        "From Date",
        value=date.today() - timedelta(days=30)
    )

    end_date = d2.date_input(
        "To Date",
        value=date.today()
    )

    filtered = tickets.copy()

    if search_text.strip():

        search = search_text.strip().lower()

        filtered = filtered[
            filtered.apply(
                lambda row: (
                    search in str(row["ticket_id"]).lower()
                    or search in str(row["title"]).lower()
                    or search in str(row["created_by"]).lower()
                ),
                axis=1
            )
        ]

    if status_filter != "All":
        filtered = filtered[
            filtered["status"] == status_filter
        ]

    if priority_filter != "All":
        filtered = filtered[
            filtered["priority"] == priority_filter
        ]

    if category_filter != "All":
        filtered = filtered[
            filtered["category_name"] == category_filter
        ]

    ticket_dates = pd.to_datetime(
        filtered["created_at"],
        errors="coerce"
    ).dt.date

    filtered = filtered[
        (ticket_dates >= start_date)
        & (ticket_dates <= end_date)
    ]

    st.caption(
        f"Showing {len(filtered)} of {len(tickets)} tickets"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )


# ----------------------------------
    # PERFORMANCE ANALYTICS
    # ----------------------------------

    st.divider()
    st.header("IT Support Performance Analytics")

    analytics_conn = sqlite3.connect(DB_PATH)

    analytics_df = pd.read_sql_query("""
        SELECT
            t.ticket_id,
            t.priority,
            t.status,
            t.created_at,
            t.first_response_at,
            t.resolved_at,
            t.response_due_at,
            t.resolution_due_at,
            u.username AS support_staff
        FROM tickets t
        LEFT JOIN users u
            ON t.assigned_to = u.user_id
    """, analytics_conn)

    analytics_conn.close()

    for col in [
        "created_at",
        "first_response_at",
        "resolved_at"
    ]:
        analytics_df[col] = pd.to_datetime(
            analytics_df[col], errors="coerce"
        )

    analytics_df["FRT_Hours"] = (
        analytics_df["first_response_at"]
        - analytics_df["created_at"]
    ).dt.total_seconds() / 3600

    analytics_df["Resolution_Hours"] = (
        analytics_df["resolved_at"]
        - analytics_df["created_at"]
    ).dt.total_seconds() / 3600

    total_tickets = len(analytics_df)
    open_tickets = (analytics_df["status"] == "Open").sum()
    progress_tickets = (analytics_df["status"] == "In Progress").sum()
    resolved_tickets = (analytics_df["status"] == "Resolved").sum()

    avg_frt = analytics_df["FRT_Hours"].mean()
    avg_resolution = analytics_df["Resolution_Hours"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Tickets", total_tickets)
    c2.metric("Open Tickets", int(open_tickets))
    c3.metric("In Progress", int(progress_tickets))
    c4.metric("Resolved", int(resolved_tickets))

    c5, c6 = st.columns(2)

    c5.metric(
        "Average First Response",
        f"{avg_frt:.2f} Hours" if pd.notna(avg_frt) else "N/A"
    )

    c6.metric(
        "Average Resolution Time",
        f"{avg_resolution:.2f} Hours"
        if pd.notna(avg_resolution) else "N/A"
    )

    st.subheader("Ticket Status Distribution")

    status_data = (
        analytics_df["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Tickets")
    )

    if not status_data.empty:
        st.bar_chart(
            status_data.set_index("Status")
        )

    st.subheader("Support Staff Workload")

    staff_data = (
        analytics_df.dropna(subset=["support_staff"])
        .groupby("support_staff")
        .agg(
            Assigned=("ticket_id", "count"),
            Open=("status", lambda x: (x == "Open").sum()),
            In_Progress=("status", lambda x: (x == "In Progress").sum()),
            Resolved=("status", lambda x: (x == "Resolved").sum())
        )
        .reset_index()
    )

    st.dataframe(
        staff_data,
        use_container_width=True,
        hide_index=True
    )


    # ----------------------------------
    # ADVANCED PERFORMANCE ANALYTICS
    # ----------------------------------

    st.divider()
    st.subheader("Advanced Performance Analytics")

    # Load due-date fields and support staff roster
    import database

    with database.database_connection() as adv_conn:
        adv_rows = database.fetch_all(
            adv_conn,
            """
            SELECT
                t.ticket_id,
                t.priority,
                t.status,
                t.created_at,
                t.first_response_at,
                t.resolved_at,
                t.response_due_at,
                t.resolution_due_at,
                u.username AS support_staff
            FROM tickets t
            LEFT JOIN users u
                ON t.assigned_to = u.user_id
            """
        )

        roster_rows = database.fetch_all(
            adv_conn,
            """
            SELECT user_id, username
            FROM users
            WHERE role = ?
            ORDER BY username
            """,
            ("support",)
        )

    adv_df = pd.DataFrame([dict(row) for row in adv_rows])
    support_roster = pd.DataFrame([dict(row) for row in roster_rows])

    adv_date_cols = [
        "created_at", "first_response_at", "resolved_at",
        "response_due_at", "resolution_due_at"
    ]
    for col in adv_date_cols:
        adv_df[col] = pd.to_datetime(adv_df[col], errors="coerce")

    adv_df["FRT_Hours"] = (
        adv_df["first_response_at"] - adv_df["created_at"]
    ).dt.total_seconds() / 3600

    adv_df["Resolution_Hours"] = (
        adv_df["resolved_at"] - adv_df["created_at"]
    ).dt.total_seconds() / 3600

    response_done = (
        adv_df["first_response_at"].notna()
        & adv_df["response_due_at"].notna()
    )
    resolution_done = (
        adv_df["resolved_at"].notna()
        & adv_df["resolution_due_at"].notna()
    )

    adv_df["Response_SLA_Result"] = "Pending"
    adv_df["Resolution_SLA_Result"] = "Pending"

    adv_df.loc[response_done, "Response_SLA_Result"] = (
        adv_df.loc[response_done, "first_response_at"]
        <= adv_df.loc[response_done, "response_due_at"]
    ).map({True: "Met", False: "Breached"})

    adv_df.loc[resolution_done, "Resolution_SLA_Result"] = (
        adv_df.loc[resolution_done, "resolved_at"]
        <= adv_df.loc[resolution_done, "resolution_due_at"]
    ).map({True: "Met", False: "Breached"})

    response_count = int(response_done.sum())
    resolution_count = int(resolution_done.sum())

    response_met = int(
        (adv_df["Response_SLA_Result"] == "Met").sum()
    )
    resolution_met = int(
        (adv_df["Resolution_SLA_Result"] == "Met").sum()
    )

    response_pct = (
        response_met / response_count * 100
        if response_count else None
    )
    resolution_pct = (
        resolution_met / resolution_count * 100
        if resolution_count else None
    )

    st.markdown("**SLA Compliance (evaluated completed events only)**")
    sla1, sla2, sla3, sla4 = st.columns(4)
    sla1.metric("Response SLA Compliance",
                f"{response_pct:.1f}%" if response_pct is not None else "N/A")
    sla2.metric("Evaluated Responses", response_count)
    sla3.metric("Resolution SLA Compliance",
                f"{resolution_pct:.1f}%" if resolution_pct is not None else "N/A")
    sla4.metric("Evaluated Resolutions", resolution_count)

    # Staff performance: include support users with zero assigned tickets
    assigned_staff = adv_df.dropna(subset=["support_staff"]).copy()

    if not assigned_staff.empty:
        staff_detail = (
            assigned_staff.groupby("support_staff")
            .agg(
                Assigned=("ticket_id", "count"),
                Resolved=("status", lambda x: (x == "Resolved").sum()),
                Open=("status", lambda x: (x == "Open").sum()),
                In_Progress=("status", lambda x: (x == "In Progress").sum()),
                Average_FRT_Hours=("FRT_Hours", "mean"),
                Average_Resolution_Hours=("Resolution_Hours", "mean")
            )
            .reset_index()
        )
    else:
        staff_detail = pd.DataFrame(columns=[
            "support_staff", "Assigned", "Resolved", "Open",
            "In_Progress", "Average_FRT_Hours", "Average_Resolution_Hours"
        ])

    staff_detail = support_roster[["username"]].rename(
        columns={"username": "support_staff"}
    ).merge(staff_detail, on="support_staff", how="left")

    for metric_col in ["Assigned", "Resolved", "Open", "In_Progress"]:
        if metric_col in staff_detail.columns:
            staff_detail[metric_col] = staff_detail[metric_col].fillna(0).astype(int)

    st.markdown("**Support Staff Performance Comparison**")
    st.dataframe(
        staff_detail.round(2),
        use_container_width=True,
        hide_index=True
    )

    # Daily and monthly ticket trends
    st.markdown("**Daily Ticket Trend**")
    daily_adv = adv_df.dropna(subset=["created_at"]).copy()
    if not daily_adv.empty:
        daily_adv["Date"] = daily_adv["created_at"].dt.strftime("%Y-%m-%d")
        daily_counts = (
            daily_adv.groupby("Date")
            .agg(
                Total_Tickets=("ticket_id", "count"),
                Resolved=("status", lambda x: (x == "Resolved").sum())
            )
        )
        st.line_chart(daily_counts)
    else:
        st.info("No ticket creation dates are available.")

    st.markdown("**Monthly Ticket Trend**")
    monthly_adv = adv_df.dropna(subset=["created_at"]).copy()
    if not monthly_adv.empty:
        monthly_adv["Month"] = monthly_adv["created_at"].dt.to_period("M").astype(str)
        monthly_counts = (
            monthly_adv.groupby("Month")
            .agg(
                Total_Tickets=("ticket_id", "count"),
                Resolved=("status", lambda x: (x == "Resolved").sum()),
                Open=("status", lambda x: (x == "Open").sum())
            )
        )
        st.bar_chart(monthly_counts)
    else:
        st.info("No monthly ticket data is available.")

    # Priority-wise ticket distribution
    st.markdown("**Priority-wise Ticket Distribution**")
    priority_counts = (
        adv_df.groupby("priority", dropna=False)
        .agg(
            Total_Tickets=("ticket_id", "count"),
            Resolved=("status", lambda x: (x == "Resolved").sum()),
            Open=("status", lambda x: (x == "Open").sum()),
            In_Progress=("status", lambda x: (x == "In Progress").sum())
        )
    )
    if not priority_counts.empty:
        st.bar_chart(priority_counts)
    else:
        st.info("No priority data is available.")



##-----------------------------------------

















    st.divider()

    # SUPPORT STAFF OPTIONS (keep inside Admin Dashboard)
    import database

    with database.database_connection() as support_users_conn:
        support_user_rows = database.fetch_all(
            support_users_conn,
            """
            SELECT user_id, username
            FROM users
            WHERE role = ?
            ORDER BY username
            """,
            ("support",)
        )

    support_users = pd.DataFrame(
        [dict(row) for row in support_user_rows]
    )

    # TICKET ASSIGNMENT AND UPDATE
    st.subheader(" Update and Assign Ticket")
    

    if not tickets.empty:

        selected_id = st.selectbox(
            "Select Ticket ID",
            tickets["ticket_id"].tolist()
        )

        selected_row = tickets[
            tickets["ticket_id"] == selected_id
        ].iloc[0]

        st.write("Ticket:", selected_row["title"])

        import database

        with database.database_connection() as assignment_conn:
            assignment_row = database.fetch_one(
                assignment_conn,
                "SELECT assigned_to FROM tickets WHERE ticket_id = ?",
                (int(selected_id),)
            )

        current_assignment = (
            assignment_row["assigned_to"]
            if assignment_row else None
        )

        assignment_options = [0] + support_users["user_id"].astype(int).tolist()

        # Preserve an existing assignment even if that account is no longer a support user.
        if current_assignment is not None and int(current_assignment) not in assignment_options:
            assignment_options.append(int(current_assignment))

        current_assignment_value = int(current_assignment) if current_assignment is not None else 0

        def assignment_label(value):
            if value == 0:
                return "Unassigned"
            match = support_users[support_users["user_id"] == value]
            if not match.empty:
                return str(match.iloc[0]["username"])
            return f"Existing assignment (User ID {value})"

        with st.form("advanced_ticket_update"):

            new_status = st.selectbox(
                "Update Status",
                ["Open", "In Progress", "Resolved"],
                index=[
                    "Open",
                    "In Progress",
                    "Resolved"
                ].index(selected_row["status"])
            )

            new_priority = st.selectbox(
                "Update Priority",
                ["Low", "Medium", "High"],
                index=[
                    "Low",
                    "Medium",
                    "High"
                ].index(selected_row["priority"])
            )

            new_assignment = st.selectbox(
                "Assign To",
                assignment_options,
                index=assignment_options.index(
                    current_assignment_value
                ),
                format_func=assignment_label
            )

            save_changes = st.form_submit_button(
                "Save Changes",
                type="primary"
            )

            if save_changes:

                assigned_user_id = (
                    None if new_assignment == 0
                    else int(new_assignment)
                )

                old_status = selected_row["status"]
                old_priority = selected_row["priority"]

                response_due, resolution_due = calculate_sla_deadlines(new_priority)

                details = (
                    f"Status: {old_status} -> {new_status}; "
                    f"Priority: {old_priority} -> {new_priority}; "
                    f"Assigned to: {assignment_label(new_assignment)}"
                )

                try:
                    with database.database_connection() as update_conn:

                        database.execute_query(
                            update_conn,
                            """
                            UPDATE tickets
                            SET status = ?,
                                priority = ?,
                                assigned_to = ?,
                                response_due_at = CASE
                                    WHEN first_response_at IS NULL
                                    THEN ? ELSE response_due_at
                                END,
                                resolution_due_at = CASE
                                    WHEN status != 'Resolved'
                                         AND ? = 'Resolved'
                                    THEN resolution_due_at
                                    ELSE ?
                                END,
                                resolved_at = CASE
                                    WHEN ? = 'Resolved'
                                    THEN COALESCE(resolved_at, CURRENT_TIMESTAMP)
                                    ELSE NULL
                                END,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE ticket_id = ?
                            """,
                            (
                                new_status,
                                new_priority,
                                assigned_user_id,
                                response_due,
                                new_status,
                                resolution_due,
                                new_status,
                                int(selected_id)
                            )
                        )

                        database.execute_query(
                            update_conn,
                            """
                            INSERT INTO audit_logs
                                (user_id, ticket_id, action, details)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                int(current_user["user_id"]),
                                int(selected_id),
                                "TICKET_UPDATED",
                                details
                            )
                        )

                    st.success("Ticket updated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Ticket update failed: {e}")

    st.divider()

    # TICKET STATUS CHART
    st.subheader(" Ticket Status Analytics")

    # Fetch chart data using a fresh managed connection.
    with database.database_connection() as chart_conn:
        status_rows = database.fetch_all(
            chart_conn,
            """
            SELECT status, COUNT(*) AS total
            FROM tickets
            GROUP BY status
            """
        )

    status_chart = pd.DataFrame(
        [dict(row) for row in status_rows],
        columns=["status", "total"]
    )

    if not status_chart.empty:

        st.bar_chart(
            status_chart.set_index("status")
        )

    st.divider()

    # AUDIT HISTORY
    st.subheader(" Recent Audit History")

    # Audit history also uses its own fresh managed connection.
    with database.database_connection() as audit_conn:
        audit_rows = database.fetch_all(
            audit_conn,
            """
            SELECT
                l.log_id,
                u.username,
                l.ticket_id,
                l.action,
                l.details,
                l.created_at
            FROM audit_logs l
            LEFT JOIN users u
                ON l.user_id = u.user_id
            ORDER BY l.log_id DESC
            LIMIT 50
            """
        )

    audit_history = pd.DataFrame(
        [dict(row) for row in audit_rows],
        columns=["log_id", "username", "ticket_id", "action", "details", "created_at"]
    )

    st.dataframe(
        audit_history,
        use_container_width=True,
        hide_index=True
    )


elif menu == "Support Dashboard":

    if current_user["role"] != "support":
        st.error("Access denied.")
        st.stop()

    st.subheader("IT Support Staff Dashboard")
    st.caption(f"Assigned tickets for {current_user['username']}")

    import database

    with database.database_connection() as support_conn:
        support_rows = database.fetch_all(
            support_conn,
            """
            SELECT
                t.ticket_id,
                t.title,
                t.description,
                u.username AS created_by,
                c.category_name,
                t.priority,
                t.status,
                t.created_at,
                t.updated_at,
                t.response_due_at,
                t.resolution_due_at
            FROM tickets t
            LEFT JOIN users u ON t.user_id = u.user_id
            LEFT JOIN categories c ON t.category_id = c.category_id
            WHERE t.assigned_to = ?
            ORDER BY t.ticket_id DESC
            """,
            (int(current_user["user_id"]),)
        )

    assigned_tickets = pd.DataFrame(
        [dict(row) for row in support_rows],
        columns=[
            "ticket_id", "title", "description", "created_by",
            "category_name", "priority", "status", "created_at",
            "updated_at", "response_due_at", "resolution_due_at"
        ]
    )

    total_assigned = len(assigned_tickets)
    open_assigned = int((assigned_tickets["status"] == "Open").sum()) if total_assigned else 0
    progress_assigned = int((assigned_tickets["status"] == "In Progress").sum()) if total_assigned else 0
    resolved_assigned = int((assigned_tickets["status"] == "Resolved").sum()) if total_assigned else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assigned Tickets", total_assigned)
    k2.metric("Open", open_assigned)
    k3.metric("In Progress", progress_assigned)
    k4.metric("Resolved", resolved_assigned)

    st.divider()
    st.subheader("My Assigned Tickets")

    if assigned_tickets.empty:
        st.info("No tickets have been assigned to you yet.")
    else:
        st.dataframe(
            assigned_tickets,
            use_container_width=True,
            hide_index=True
        )

        ticket_ids = assigned_tickets["ticket_id"].astype(int).tolist()
        selected_support_ticket = st.selectbox(
            "Select an assigned ticket",
            ticket_ids,
            format_func=lambda ticket_id: (
                f"Ticket #{ticket_id} - "
                + str(assigned_tickets.loc[
                    assigned_tickets["ticket_id"] == ticket_id, "title"
                ].iloc[0])
            ),
            key="support_ticket_selector"
        )

        selected_support_row = assigned_tickets[
            assigned_tickets["ticket_id"] == selected_support_ticket
        ].iloc[0]

        st.write("**Description:**", selected_support_row["description"])
        st.write(
            f"**Priority:** {selected_support_row['priority']} | "
            f"**Status:** {selected_support_row['status']}"
        )

        with st.form(f"support_update_{selected_support_ticket}"):
            support_new_status = st.selectbox(
                "Update ticket status",
                ["Open", "In Progress", "Resolved"],
                index=["Open", "In Progress", "Resolved"].index(
                    selected_support_row["status"]
                )
            )
            support_update_button = st.form_submit_button(
                "Save Ticket Status",
                type="primary"
            )

        if support_update_button:
            import database

            with database.database_connection() as support_conn:
                database.execute_query(
                    support_conn,
                    """
                    UPDATE tickets
                    SET status = ?,
                        resolved_at = CASE
                            WHEN ? = 'Resolved'
                            THEN COALESCE(resolved_at, CURRENT_TIMESTAMP)
                            ELSE NULL
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE ticket_id = ? AND assigned_to = ?
                    """,
                    (
                        support_new_status,
                        support_new_status,
                        int(selected_support_ticket),
                        int(current_user["user_id"])
                    )
                )

                database.execute_query(
                    support_conn,
                    """
                    INSERT INTO audit_logs
                    (user_id, ticket_id, action, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        int(current_user["user_id"]),
                        int(selected_support_ticket),
                        "SUPPORT_TICKET_STATUS_UPDATED",
                        f"Status changed to {support_new_status}"
                    )
                )

            st.success("Ticket status updated.")
            st.rerun()

        st.divider()
        st.subheader("Ticket Conversation")

        support_conversation = load_ticket_comments(selected_support_ticket)
        if support_conversation.empty:
            st.info("No messages have been added to this ticket.")
        else:
            for _, message in support_conversation.iterrows():
                st.markdown(
                    f"**{message['username'] or 'Unknown'}** "
                    f"({message['role'] or 'user'}) — {message['created_at']}"
                )
                st.write(message["comment"])
                st.divider()

        with st.form(f"support_reply_{selected_support_ticket}", clear_on_submit=True):
            support_reply = st.text_area(
                "Reply to ticket owner",
                max_chars=3000
            )
            support_send = st.form_submit_button("Send Reply")

        if support_send:
            if not support_reply.strip():
                st.warning("Please enter a message.")
            else:
                try:
                    save_ticket_comment(
                        selected_support_ticket,
                        current_user["user_id"],
                        support_reply
                    )
                    notify_ticket_comment(
                        selected_support_ticket,
                        current_user["user_id"],
                        support_reply.strip()
                    )
                    st.success("Reply sent.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Unable to send reply: {error}")

        st.subheader("Ticket Attachments")
        with st.form(f"support_attachment_form_{selected_support_ticket}"):
            support_attachment = st.file_uploader(
                "Upload an attachment",
                type=sorted(ALLOWED_ATTACHMENT_EXTENSIONS),
                key=f"support_attachment_{selected_support_ticket}"
            )
            support_upload = st.form_submit_button("Upload Attachment")

        if support_upload:
            if support_attachment is None:
                st.warning("Please select a file.")
            else:
                try:
                    save_ticket_attachment(
                        selected_support_ticket,
                        support_attachment,
                        current_user["user_id"]
                    )
                    st.success("Attachment uploaded.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Unable to upload attachment: {error}")

        support_attachments = load_ticket_attachments(selected_support_ticket)
        if not support_attachments.empty:
            for _, attachment_row in support_attachments.iterrows():
                stored_name = Path(str(attachment_row["stored_name"])).name
                file_path = (ATTACHMENT_DIR / stored_name).resolve()
                safe_root = ATTACHMENT_DIR.resolve()
                if safe_root not in file_path.parents or not file_path.is_file():
                    st.warning(f"Attachment unavailable: {attachment_row['original_name']}")
                    continue
                with file_path.open("rb") as file_handle:
                    st.download_button(
                        label=f"Download {attachment_row['original_name']}",
                        data=file_handle.read(),
                        file_name=attachment_row["original_name"],
                        mime=attachment_row["mime_type"] or "application/octet-stream",
                        key=f"support_download_{attachment_row['attachment_id']}"
                    )


st.divider()

st.caption(
    "MSc Computer Science | AI-Based IT Helpdesk | Nimesh Wijebandara"
)
