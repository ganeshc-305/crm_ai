import os
import streamlit as st
import logging

from src.agents.supervisor import Supervisor
from src.agents.document_agent import DocumentAgent
from src.agents.retriever_agent import RetrieverAgent
from src.rag.loaders import load_documents_from_dir, split_documents_to_texts
from src import db as db_mod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CRM AI Assistant")
st.title("CRM AI Assistant")

# Sidebar configuration
st.sidebar.header("Configuration")
persist_dir = st.sidebar.text_input("Vector persist dir", value=os.environ.get("RAG_PERSIST_DIR", "data/vector_store"))
api_key = st.sidebar.text_input("LLM API Key (optional)", type="password")

# Initialize DB if configured
database_url = os.environ.get('DATABASE_URL') or os.environ.get('JDBC_DATABASE_URL') or os.environ.get('JDBC_URL')
engine = None
if database_url:
    try:
        engine = db_mod.init_db(database_url)
        st.sidebar.success("DB initialized")
    except Exception as e:
        st.sidebar.error(f"DB init failed: {e}")

# Initialize supervisor and agents in session state to persist across reruns
if 'sup' not in st.session_state:
    sup = Supervisor()
    doc_agent = DocumentAgent(persist_dir=persist_dir)
    retr_agent = RetrieverAgent(persist_dir=persist_dir)
    sup.register('document', doc_agent)
    sup.register('retriever', retr_agent)
    st.session_state['sup'] = sup
else:
    sup = st.session_state['sup']

# Load persisted chat history from DB if available
if 'history' not in st.session_state:
    st.session_state['history'] = []
    if engine is not None:
        try:
            recent = db_mod.load_recent_messages(limit=200)
            for r in recent:
                st.session_state['history'].append({'role': r['role'], 'content': r['content']})
        except Exception as e:
            logger.exception('Failed to load messages from DB: %s', e)

# File upload area
st.header("Documents")
uploaded = st.file_uploader("Upload files to ingest", accept_multiple_files=True)
if uploaded:
    saved = []
    os.makedirs(os.environ.get('UPLOAD_DIR', 'data/uploads'), exist_ok=True)
    for up in uploaded:
        dest = os.path.join(os.environ.get('UPLOAD_DIR', 'data/uploads'), up.name)
        with open(dest, 'wb') as f:
            f.write(up.getbuffer())
        saved.append(dest)
    st.success(f"Saved {len(saved)} files to data/uploads")

if st.button("Ingest uploaded files"):
    docs = load_documents_from_dir(os.environ.get('UPLOAD_DIR', 'data/uploads'))
    if not docs:
        st.warning("No documents found in upload directory to ingest.")
    else:
        texts, metadatas = split_documents_to_texts(docs)
        with st.spinner("Ingesting chunks into vector store..."):
            res = sup.dispatch('document', {'type': 'ingest_texts', 'texts': texts, 'metadatas': metadatas})
        if res.get('status') == 'ok':
            st.success(f"Ingested {res.get('ingested')} chunks")
        else:
            st.error(f"Ingestion failed: {res}")

# Chat area
st.header("Chat (RAG)")
with st.form('chat_form', clear_on_submit=True):
    message = st.text_input('Enter your question')
    submit = st.form_submit_button('Send')

if submit and message:
    # append user turn
    st.session_state['history'].append({'role': 'user', 'content': message})
    # persist to DB if available
    if engine is not None:
        try:
            db_mod.save_message('user', message, metadata={})
        except Exception as e:
            logger.exception('Failed to save user message: %s', e)

    # query retriever agent
    try:
        qres = sup.dispatch('retriever', {'type': 'query', 'query': message, 'k': 4})
        if qres.get('status') == 'ok':
            results = qres.get('results', [])
            # build assistant reply from retrieved docs (no LLM call here)
            if results:
                reply_parts = [f"Source: {r.get('metadata',{}).get('source','unknown')}\n{r.get('content')[:500]}" for r in results]
                assistant_text = "\n\n---\n\n".join(reply_parts)
            else:
                assistant_text = "No relevant documents found."
        else:
            assistant_text = f"Retriever error: {qres}"
    except Exception as e:
        assistant_text = f"Retriever exception: {e}"

    st.session_state['history'].append({'role': 'assistant', 'content': assistant_text})
    # persist assistant message
    if engine is not None:
        try:
            db_mod.save_message('assistant', assistant_text, metadata={})
        except Exception as e:
            logger.exception('Failed to save assistant message: %s', e)

# Render chat history
for turn in st.session_state['history']:
    if turn['role'] == 'user':
        st.markdown(f"**User:** {turn['content']}")
    else:
        st.markdown(f"**Assistant (RAG):**\n\n{turn['content']}")

st.info("Use 'Ingest uploaded files' after uploading. This app shows retrieved passages; integrate an LLM call for full generative answers.")
