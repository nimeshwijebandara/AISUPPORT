
import streamlit as st
from pathlib import Path
import joblib
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(r"D:\AI_IT_HELPDESK")

MODEL_PATH = BASE_DIR / "models" / "it_helpdesk_classifier.pkl"
VECTOR_DIR = BASE_DIR / "knowledge_base" / "vector_store"

st.set_page_config(
    page_title="AI IT Helpdesk",
    page_icon="🤖",
    layout="wide"
)

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

def classify_issue(query):
    category = classifier.predict([query])[0]

    confidence = None
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba([query])[0]
        confidence = float(max(probabilities))

    return category, confidence

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

st.title("🤖 AI-Based IT Helpdesk")
st.caption("AI Issue Classification and Knowledge-Based Troubleshooting")

st.sidebar.title("Helpdesk Menu")
st.sidebar.info("AI-powered IT support assistant")

st.subheader("How can we help you?")

user_query = st.text_area(
    "Describe your IT issue:",
    placeholder="Example: My printer is offline and I cannot print documents...",
    height=120
)

if st.button("Submit Issue", type="primary"):

    if not user_query.strip():
        st.warning("Please enter your IT issue.")

    else:
        with st.spinner("AI is analysing your issue..."):

            category, confidence = classify_issue(user_query)

            documents, metadatas = retrieve_knowledge(
                user_query,
                category
            )

        st.success("Issue analysis completed!")

        st.subheader("AI Classification")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Category", str(category))

        with col2:
            if confidence is not None:
                st.metric(
                    "Classification Confidence",
                    f"{confidence:.2%}"
                )

        st.subheader("Troubleshooting Guidance")

        if documents:
            for i, document in enumerate(documents):
                title = metadatas[i].get(
                    "title", f"Knowledge Article {i+1}"
                )

                with st.expander(title, expanded=(i == 0)):
                    st.write(document)

        else:
            st.warning(
                "No matching knowledge article was found. "
                "Please contact the IT support team."
            )

        st.subheader("Support Status")

        if confidence is not None and confidence < 0.50:
            st.warning(
                "Low classification confidence. "
                "Manual IT support review is recommended."
            )
        elif not documents:
            st.warning(
                "Manual IT support review is recommended."
            )
        else:
            st.info(
                "Suggested troubleshooting guidance is available. "
                "Follow the instructions and contact IT support "
                "if the issue remains unresolved."
            )

st.divider()

st.caption(
    "MSc Computer Science Project | AI-Based IT Helpdesk "
    "and Automated Troubleshooting System"
)
