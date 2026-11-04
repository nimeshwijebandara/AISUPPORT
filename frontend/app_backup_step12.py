
import streamlit as st
import sqlite3
import joblib
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ----------------------------------
# PROJECT CONFIGURATION
# ----------------------------------

BASE_DIR = Path(r"D:\AI_IT_HELPDESK")

MODEL_PATH = BASE_DIR / "models" / "it_helpdesk_classifier.pkl"
VECTOR_DIR = BASE_DIR / "knowledge_base" / "vector_store"
DB_PATH = BASE_DIR / "database" / "helpdesk.db"

st.set_page_config(
    page_title="AI IT Helpdesk",
    page_icon="🤖",
    layout="wide"
)

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
# AI CLASSIFICATION
# ----------------------------------

def classify_issue(query):

    category = classifier.predict([query])[0]

    confidence = None

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba([query])[0]
        confidence = float(max(probabilities))

    return category, confidence


# ----------------------------------
# KNOWLEDGE RETRIEVAL
# ----------------------------------

def retrieve_knowledge(query, category, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"category": category}
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas


# ----------------------------------
# CREATE SUPPORT TICKET
# ----------------------------------

def create_support_ticket(
    title,
    description,
    category_name,
    priority="Medium"
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
            raise ValueError(
                f"Unknown category: {category_name}"
            )

        cursor.execute(
            """
            INSERT INTO tickets
            (title, description, category_id, priority, status)
            VALUES (?, ?, ?, ?, 'Open')
            """,
            (
                title,
                description,
                category[0],
                priority
            )
        )

        ticket_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO audit_logs
            (ticket_id, action, details)
            VALUES (?, ?, ?)
            """,
            (
                ticket_id,
                "TICKET_CREATED",
                f"Created from AI Helpdesk: {title}"
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
# USER INTERFACE
# ----------------------------------

st.title("🤖 AI-Based IT Helpdesk")

st.caption(
    "AI Issue Classification and Knowledge-Based Troubleshooting"
)

st.sidebar.title("Helpdesk Menu")

menu = st.sidebar.radio(
    "Navigation",
    ["AI Helpdesk", "My Tickets"]
)

if menu == "AI Helpdesk":

    st.subheader("How can we help you?")

    user_query = st.text_area(
        "Describe your IT issue:",
        placeholder="Example: My printer is offline...",
        height=120
    )

    if st.button("Analyse Issue", type="primary"):

        if not user_query.strip():

            st.warning("Please enter your IT issue.")

        else:

            with st.spinner("AI is analysing your issue..."):

                category, confidence = classify_issue(user_query)

                documents, metadatas = retrieve_knowledge(
                    user_query,
                    category
                )

            st.session_state["analysis"] = {
                "query": user_query,
                "category": category,
                "confidence": confidence,
                "documents": documents,
                "metadatas": metadatas
            }

    # Display saved analysis after Streamlit reruns
    analysis = st.session_state.get("analysis")

    if analysis:

        st.success("Issue analysis completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Category",
                str(analysis["category"])
            )

        with col2:
            confidence = analysis["confidence"]

            if confidence is not None:
                st.metric(
                    "Classification Confidence",
                    f"{confidence:.2%}"
                )

        st.subheader("Troubleshooting Guidance")

        documents = analysis["documents"]
        metadatas = analysis["metadatas"]

        if documents:

            for i, document in enumerate(documents):

                title = metadatas[i].get(
                    "title",
                    f"Knowledge Article {i+1}"
                )

                with st.expander(title, expanded=(i == 0)):
                    st.write(document)

        else:

            st.warning(
                "No matching knowledge article found."
            )

        confidence = analysis["confidence"]

        if confidence is not None and confidence < 0.50:

            st.warning(
                "Low classification confidence. "
                "Please review the predicted category "
                "before creating a ticket."
            )

        st.divider()

        # ----------------------------------
        # TICKET CREATION FORM
        # ----------------------------------

        st.subheader("🎫 Create Support Ticket")

        st.write(
            "If your issue is not resolved, "
            "you can submit a support ticket."
        )

        with st.form("ticket_creation_form"):

            ticket_title = st.text_input(
                "Ticket Title",
                value=analysis["query"][:100]
            )

            ticket_description = st.text_area(
                "Issue Description",
                value=analysis["query"]
            )

            ticket_category = st.selectbox(
                "Issue Category",
                [
                    "Account & Password",
                    "Network & VPN",
                    "Printer",
                    "Software",
                    "System Health"
                ],
                index=[
                    "Account & Password",
                    "Network & VPN",
                    "Printer",
                    "Software",
                    "System Health"
                ].index(analysis["category"])
                if analysis["category"] in [
                    "Account & Password",
                    "Network & VPN",
                    "Printer",
                    "Software",
                    "System Health"
                ] else 0
            )

            ticket_priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High"]
            )

            submitted = st.form_submit_button(
                "Create Support Ticket",
                type="primary"
            )

            if submitted:

                if not ticket_title.strip():
                    st.error("Please enter a ticket title.")

                elif not ticket_description.strip():
                    st.error("Please enter an issue description.")

                else:

                    try:

                        new_ticket_id = create_support_ticket(
                            title=ticket_title.strip(),
                            description=ticket_description.strip(),
                            category_name=ticket_category,
                            priority=ticket_priority
                        )

                        st.session_state["last_ticket_id"] = (
                            new_ticket_id
                        )

                        st.success(
                            f"Ticket created successfully! "
                            f"Ticket ID: {new_ticket_id}"
                        )

                    except Exception as error:

                        st.error(
                            f"Ticket creation failed: {error}"
                        )

    st.divider()

    st.caption(
        "MSc Computer Science Project | "
        "AI-Based IT Helpdesk and Automated Troubleshooting System"
    )


# ----------------------------------
# MY TICKETS
# ----------------------------------

elif menu == "My Tickets":

    st.subheader("🎫 Support Ticket History")

    conn = sqlite3.connect(DB_PATH)

    tickets = conn.execute(
        """
        SELECT
            t.ticket_id,
            t.title,
            c.category_name,
            t.priority,
            t.status,
            t.created_at
        FROM tickets t
        LEFT JOIN categories c
            ON t.category_id = c.category_id
        ORDER BY t.ticket_id DESC
        """
    ).fetchall()

    conn.close()

    if tickets:

        import pandas as pd

        ticket_df = pd.DataFrame(
            tickets,
            columns=[
                "Ticket ID",
                "Title",
                "Category",
                "Priority",
                "Status",
                "Created At"
            ]
        )

        st.dataframe(
            ticket_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No support tickets found.")
