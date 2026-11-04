
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

BASE_DIR = Path(r"D:\AI_IT_HELPDESK")
DB_PATH = BASE_DIR / "database" / "helpdesk.db"
MODEL_PATH = BASE_DIR / "models" / "it_helpdesk_classifier.pkl"
VECTOR_DIR = BASE_DIR / "knowledge_base" / "vector_store"
ATTACHMENT_DIR = BASE_DIR / "uploads" / "ticket_attachments"
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_ATTACHMENT_EXTENSIONS = {"png", "jpg", "jpeg", "pdf", "doc", "docx", "txt", "log", "csv"}

# SLA targets are calendar hours (not business hours). Adjust to your organisation's policy.
SLA_TARGETS = {
    "High": {"response_hours": 1, "resolution_hours": 8},
    "Medium": {"response_hours": 4, "resolution_hours": 24},
    "Low": {"response_hours": 8, "resolution_hours": 72},
}

st.set_page_config(
    page_title="AI IT Helpdesk",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------
# AUTHENTICATION
# ----------------------------------

def authenticate_user(username, password):

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, username, password_hash, role
            FROM users
            WHERE LOWER(username) = LOWER(?)
            """,
            (username.strip(),)
        )

        user = cursor.fetchone()

        if not user or not user[2]:
            return None

        valid = bcrypt.checkpw(
            password.encode("utf-8"),
            user[2].encode("utf-8")
        )

        if valid:
            return {
                "user_id": user[0],
                "username": user[1],
                "role": user[3]
            }

        return None

    finally:
        conn.close()


def login_required():
    return "user" in st.session_state


# ----------------------------------
# LOGIN PAGE
# ----------------------------------

if not login_required():

    st.title("🤖 AI-Based IT Helpdesk")
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

def send_email_notification(recipient, subject, body):
    """Send an email when SMTP environment variables are configured.

    Required environment variables: SMTP_HOST, SMTP_PORT, SMTP_USERNAME,
    SMTP_PASSWORD and SMTP_FROM. Email is skipped safely when not configured.
    """
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or username
    port = int(os.getenv("SMTP_PORT", "587"))

    if not all([host, username, password, sender, recipient]):
        return False, "Email notifications are not configured."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(message)
        return True, "Email notification sent."
    except Exception as error:
        # Do not block ticket conversations if the mail server is unavailable.
        return False, str(error)


def notify_ticket_comment(ticket_id, sender_user_id, comment):
    """Notify the ticket owner/admins, excluding the person who replied."""
    conn = sqlite3.connect(DB_PATH)
    try:
        ticket = conn.execute(
            """SELECT t.title, t.user_id, u.email
               FROM tickets t
               LEFT JOIN users u ON u.user_id = t.user_id
               WHERE t.ticket_id = ?""",
            (int(ticket_id),)
        ).fetchone()

        if not ticket:
            return

        title, owner_id, owner_email = ticket
        sender = conn.execute(
            "SELECT username, role FROM users WHERE user_id = ?",
            (int(sender_user_id),)
        ).fetchone()
        if not sender:
            return

        sender_name, sender_role = sender
        recipients = set()

        if sender_role == "admin":
            if owner_email and int(owner_id or 0) != int(sender_user_id):
                recipients.add(owner_email)
        else:
            admin_emails = conn.execute(
                "SELECT email FROM users WHERE role = 'admin' AND email IS NOT NULL"
            ).fetchall()
            recipients.update(row[0] for row in admin_emails if row[0])

        for recipient in recipients:
            send_email_notification(
                recipient,
                f"New reply on Helpdesk Ticket #{ticket_id}",
                f"Ticket: {title}\nFrom: {sender_name} ({sender_role})\n\n"
                f"Message: {comment}\n\nPlease sign in to the AI IT Helpdesk to view the conversation."
            )
    finally:
        conn.close()


def ensure_sla_columns():
    """Add SLA timestamps to existing ticket databases without deleting data."""
    conn = sqlite3.connect(DB_PATH)
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(tickets)").fetchall()}
        additions = {
            "response_due_at": "DATETIME",
            "resolution_due_at": "DATETIME",
            "first_response_at": "DATETIME",
            "resolved_at": "DATETIME",
        }
        for column, sql_type in additions.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {column} {sql_type}")
        # Backfill deadlines for existing tickets that do not yet have them.
        rows = conn.execute(
            "SELECT ticket_id, priority, created_at FROM tickets WHERE response_due_at IS NULL OR resolution_due_at IS NULL"
        ).fetchall()
        for ticket_id, priority, created_at in rows:
            try:
                created = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError):
                created = datetime.utcnow()
            target = SLA_TARGETS.get(priority, SLA_TARGETS["Medium"])
            conn.execute(
                "UPDATE tickets SET response_due_at = COALESCE(response_due_at, ?), resolution_due_at = COALESCE(resolution_due_at, ?) WHERE ticket_id = ?",
                ((created + timedelta(hours=target["response_hours"])).strftime("%Y-%m-%d %H:%M:%S"),
                 (created + timedelta(hours=target["resolution_hours"])).strftime("%Y-%m-%d %H:%M:%S"), int(ticket_id))
            )
        conn.commit()
    finally:
        conn.close()

def calculate_sla_deadlines(priority, created_at=None):
    created = created_at or datetime.utcnow()
    target = SLA_TARGETS.get(priority, SLA_TARGETS["Medium"])
    return (
        (created + timedelta(hours=target["response_hours"])).strftime("%Y-%m-%d %H:%M:%S"),
        (created + timedelta(hours=target["resolution_hours"])).strftime("%Y-%m-%d %H:%M:%S")
    )

def sla_state(due_at, completed_at=None, now=None):
    if completed_at:
        return "Met" if completed_at <= due_at else "Breached"
    now = now or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return "Overdue" if due_at and now > due_at else "Within SLA"


ensure_sla_columns()


def ensure_ticket_comments_table():
    """Create the ticket conversation table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
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
        )
        conn.commit()
    finally:
        conn.close()


ensure_ticket_comments_table()


def ensure_ticket_attachments_table():
    """Create metadata storage for ticket attachments."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
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
        )
        conn.commit()
    finally:
        conn.close()


def save_ticket_attachment(ticket_id, uploaded_file, user_id):
    """Validate, store and register an attachment for an accessible ticket."""
    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower().lstrip(".")
    content = uploaded_file.getvalue()

    if extension not in ALLOWED_ATTACHMENT_EXTENSIONS:
        raise ValueError("File type not allowed. Upload PNG, JPG, PDF, DOCX, DOC, TXT, LOG or CSV.")
    if not content:
        raise ValueError("The selected file is empty.")
    if len(content) > MAX_ATTACHMENT_SIZE:
        raise ValueError("File is larger than the 10 MB limit.")

    conn = sqlite3.connect(DB_PATH)
    stored_name = f"{uuid.uuid4().hex}.{extension}"
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ATTACHMENT_DIR / stored_name

    try:
        ticket = conn.execute(
            "SELECT user_id, title FROM tickets WHERE ticket_id = ?",
            (int(ticket_id),)
        ).fetchone()
        if ticket is None:
            raise ValueError("Ticket not found.")
        if current_user["role"] != "admin" and int(ticket[0] or 0) != int(user_id):
            raise PermissionError("You are not allowed to attach files to this ticket.")

        # Save the file before inserting its metadata; remove it if the DB write fails.
        destination.write_bytes(content)
        mime_type = uploaded_file.type or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        conn.execute(
            """INSERT INTO ticket_attachments
               (ticket_id, uploaded_by, original_name, stored_name, file_size, mime_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (int(ticket_id), int(user_id), original_name, stored_name, len(content), mime_type)
        )
        conn.execute(
            """INSERT INTO audit_logs (user_id, ticket_id, action, details)
               VALUES (?, ?, ?, ?)""",
            (int(user_id), int(ticket_id), "TICKET_ATTACHMENT_UPLOADED", f"Attachment uploaded: {original_name}")
        )
        conn.commit()
    except Exception:
        conn.rollback()
        if destination.exists():
            destination.unlink()
        raise
    finally:
        conn.close()


def load_ticket_attachments(ticket_id):
    """Load attachment metadata for a ticket the caller is allowed to view."""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            """SELECT a.attachment_id, a.original_name, a.stored_name,
                      a.file_size, a.mime_type, a.created_at, u.username
               FROM ticket_attachments a
               LEFT JOIN users u ON u.user_id = a.uploaded_by
               WHERE a.ticket_id = ?
               ORDER BY a.attachment_id DESC""",
            conn, params=(int(ticket_id),)
        )
    finally:
        conn.close()


ensure_ticket_attachments_table()


def save_ticket_comment(ticket_id, user_id, comment):
    """Save a message and its corresponding audit entry."""
    clean_comment = comment.strip()
    if not clean_comment:
        raise ValueError("Comment cannot be empty.")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        # Verify that the ticket exists and, for normal users, belongs to them.
        cursor.execute(
            "SELECT user_id, title FROM tickets WHERE ticket_id = ?",
            (int(ticket_id),)
        )
        ticket = cursor.fetchone()

        if ticket is None:
            raise ValueError("Ticket not found.")

        if current_user["role"] != "admin" and ticket[0] != user_id:
            raise PermissionError("You are not allowed to comment on this ticket.")

        cursor.execute(
            """
            INSERT INTO ticket_comments
                (ticket_id, user_id, comment)
            VALUES (?, ?, ?)
            """,
            (int(ticket_id), int(user_id), clean_comment)
        )

        cursor.execute(
            """
            INSERT INTO audit_logs (user_id, ticket_id, action, details)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(user_id),
                int(ticket_id),
                "TICKET_COMMENTED",
                f"Comment added to ticket: {ticket[1]}"
            )
        )

        # Record the first response time when an administrator replies.
        role_row = conn.execute("SELECT role FROM users WHERE user_id = ?", (int(user_id),)).fetchone()
        if role_row and role_row[0] == "admin":
            conn.execute(
                "UPDATE tickets SET first_response_at = COALESCE(first_response_at, CURRENT_TIMESTAMP) WHERE ticket_id = ?",
                (int(ticket_id),)
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_ticket_comments(ticket_id):
    """Return conversation messages for one ticket."""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
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
            conn,
            params=(int(ticket_id),)
        )
    finally:
        conn.close()


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

if current_user["role"] == "admin":

    menu_options = [
        "AI Helpdesk",
        "My Tickets",
        "Admin Dashboard"
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

st.title("🤖 AI-Based IT Helpdesk")


# ----------------------------------
# LOAD AI COMPONENTS
# ----------------------------------

@st.cache_resource
def load_ai_components():

    classifier = joblib.load(MODEL_PATH)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR)
    )

    collection = client.get_collection(
        name="it_helpdesk_knowledge"
    )

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    return classifier, collection, embedding_model


classifier, collection, embedding_model = load_ai_components()


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

    conn = sqlite3.connect(DB_PATH)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT category_id
            FROM categories
            WHERE LOWER(category_name) = LOWER(?)
            """,
            (category_name,)
        )

        category = cursor.fetchone()

        if category is None:
            raise ValueError("Invalid ticket category.")

        cursor.execute(
            """
            INSERT INTO tickets
            (user_id, title, description, category_id, priority, status, response_due_at, resolution_due_at)
            VALUES (?, ?, ?, ?, ?, 'Open', ?, ?)
            """,
            (
                user_id, title, description, category[0], priority,
                *calculate_sla_deadlines(priority)
            )
        )

        ticket_id = cursor.lastrowid

        cursor.execute(
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

                documents, metadatas = retrieve_knowledge(
                    query,
                    category
                )

            st.session_state["analysis"] = {
                "query": query,
                "category": category,
                "confidence": confidence,
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

    st.subheader("🎫 My Support Tickets")

    conn = sqlite3.connect(DB_PATH)

    if current_user["role"] == "admin":

        query = """
            SELECT t.ticket_id, t.title, c.category_name,
                   t.priority, t.status, t.created_at,
                   t.response_due_at, t.resolution_due_at, t.first_response_at, t.resolved_at
            FROM tickets t
            LEFT JOIN categories c
            ON t.category_id = c.category_id
            ORDER BY t.ticket_id DESC
        """

        params = ()

    else:

        query = """
            SELECT t.ticket_id, t.title, c.category_name,
                   t.priority, t.status, t.created_at,
                   t.response_due_at, t.resolution_due_at, t.first_response_at, t.resolved_at
            FROM tickets t
            LEFT JOIN categories c
            ON t.category_id = c.category_id
            WHERE t.user_id = ?
            ORDER BY t.ticket_id DESC
        """

        params = (current_user["user_id"],)

    tickets = pd.read_sql_query(
        query,
        conn,
        params=params
    )

    conn.close()

    st.dataframe(
        tickets,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("💬 Ticket Conversation")

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
                        f"**{sender}** · {sender_role} · "
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

        st.markdown("#### 📎 Ticket Attachments")
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
                    f"{attachment_row['original_name']} · "
                    f"{size_kb:.1f} KB · uploaded by {attachment_row['username'] or 'Unknown'} · "
                    f"{attachment_row['created_at']}"
                )
                with file_path.open("rb") as file_handle:
                    st.download_button(
                        label=f"⬇️ {label}",
                        data=file_handle.read(),
                        file_name=attachment_row["original_name"],
                        mime=attachment_row["mime_type"] or "application/octet-stream",
                        key=f"download_attachment_{attachment_row['attachment_id']}"
                    )


# ----------------------------------
# ADVANCED ADMIN DASHBOARD
# ----------------------------------

elif menu == "Admin Dashboard":

    if current_user["role"] != "admin":
        st.error("Access denied.")
        st.stop()

    st.subheader("📊 Advanced IT Support Dashboard")

    conn = sqlite3.connect(DB_PATH)

    # KPI SUMMARY
    total = conn.execute(
        "SELECT COUNT(*) FROM tickets"
    ).fetchone()[0]

    open_count = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE status='Open'"
    ).fetchone()[0]

    progress_count = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE status='In Progress'"
    ).fetchone()[0]

    resolved_count = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE status='Resolved'"
    ).fetchone()[0]

    high_count = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE priority='High'"
    ).fetchone()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tickets", total)
    c2.metric("Open Tickets", open_count)
    c3.metric("In Progress", progress_count)

    c4, c5 = st.columns(2)
    c4.metric("Resolved Tickets", resolved_count)
    c5.metric("High Priority", high_count)

    st.divider()
    st.subheader("⏱️ SLA Monitoring")
    sla_conn = sqlite3.connect(DB_PATH)
    sla_df = pd.read_sql_query(
        """SELECT ticket_id, title, priority, status, response_due_at, resolution_due_at, first_response_at, resolved_at
           FROM tickets WHERE status != 'Resolved' OR resolved_at IS NOT NULL ORDER BY ticket_id DESC""",
        sla_conn
    )
    sla_conn.close()
    now_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if not sla_df.empty:
        sla_df["Response SLA"] = sla_df.apply(lambda r: sla_state(r["response_due_at"], r["first_response_at"], now_text), axis=1)
        sla_df["Resolution SLA"] = sla_df.apply(lambda r: sla_state(r["resolution_due_at"], r["resolved_at"], now_text), axis=1)
        sla_cols = ["ticket_id", "title", "priority", "status", "response_due_at", "Response SLA", "resolution_due_at", "Resolution SLA"]
        st.dataframe(sla_df[sla_cols], use_container_width=True, hide_index=True)
        response_breaches = int(sla_df["Response SLA"].isin(["Overdue", "Breached"]).sum())
        resolution_breaches = int(sla_df["Resolution SLA"].isin(["Overdue", "Breached"]).sum())
        m1, m2 = st.columns(2)
        m1.metric("Response SLA breaches", response_breaches)
        m2.metric("Resolution SLA breaches", resolution_breaches)
    else:
        st.info("No tickets available for SLA monitoring.")

    st.divider()

    # LOAD TICKETS
    tickets = pd.read_sql_query(
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
        """,
        conn
    )

    st.subheader("🎫 Ticket Management")

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

    st.divider()

    # TICKET ASSIGNMENT AND UPDATE
    st.subheader("📝 Update and Assign Ticket")

    if not tickets.empty:

        selected_id = st.selectbox(
            "Select Ticket ID",
            tickets["ticket_id"].tolist()
        )

        selected_row = tickets[
            tickets["ticket_id"] == selected_id
        ].iloc[0]

        st.write("Ticket:", selected_row["title"])

        # Only admin accounts are assignable at present
        admin_users = pd.read_sql_query(
            """
            SELECT user_id, username
            FROM users
            WHERE role = 'admin'
            ORDER BY username
            """,
            conn
        )

        current_assignment = conn.execute(
            """
            SELECT assigned_to
            FROM tickets
            WHERE ticket_id = ?
            """,
            (int(selected_id),)
        ).fetchone()[0]

        assignment_options = [0] + admin_users["user_id"].tolist()

        if current_assignment is not None:
            if current_assignment not in assignment_options:
                assignment_options.append(current_assignment)

        current_assignment_value = (
            current_assignment
            if current_assignment is not None else 0
        )

        if current_assignment_value not in assignment_options:
            current_assignment_value = 0

        def assignment_label(value):

            if value == 0:
                return "Unassigned"

            match = admin_users[
                admin_users["user_id"] == value
            ]

            if not match.empty:
                return match.iloc[0]["username"]

            return f"User ID {value}"

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
                conn.execute(
                        """
                        UPDATE tickets
                        SET status = ?, priority = ?, assigned_to = ?,
                            response_due_at = CASE WHEN first_response_at IS NULL THEN ? ELSE response_due_at END,
                            resolution_due_at = CASE WHEN status != 'Resolved' AND ? = 'Resolved' THEN resolution_due_at ELSE ? END,
                            resolved_at = CASE WHEN ? = 'Resolved' THEN COALESCE(resolved_at, CURRENT_TIMESTAMP) ELSE NULL END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE ticket_id = ?
                        """,
                    (new_status, new_priority, assigned_user_id, response_due, new_status, resolution_due, new_status, int(selected_id))
                )

                details = (
                    f"Status: {old_status} -> {new_status}; "
                    f"Priority: {old_priority} -> {new_priority}; "
                    f"Assigned to: {assignment_label(new_assignment)}"
                )

                conn.execute(
                    """
                    INSERT INTO audit_logs
                    (user_id, ticket_id, action, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        current_user["user_id"],
                        int(selected_id),
                        "TICKET_UPDATED",
                        details
                    )
                )

                conn.commit()

                st.success("Ticket updated successfully!")
                st.rerun()

    st.divider()

    # TICKET STATUS CHART
    st.subheader("📈 Ticket Status Analytics")

    status_chart = pd.read_sql_query(
        """
        SELECT status, COUNT(*) AS total
        FROM tickets
        GROUP BY status
        """,
        conn
    )

    if not status_chart.empty:

        st.bar_chart(
            status_chart.set_index("status")
        )

    st.divider()

    # AUDIT HISTORY
    st.subheader("📋 Recent Audit History")

    audit_history = pd.read_sql_query(
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
        """,
        conn
    )

    st.dataframe(
        audit_history,
        use_container_width=True,
        hide_index=True
    )

    conn.close()

st.divider()

st.caption(
    "MSc Computer Science | AI-Based IT Helpdesk"
)
