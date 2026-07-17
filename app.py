"""
STREAMLIT CHAT APP  (what users see on your website)
====================================================

Starts the web UI:

    streamlit run app.py

This file only handles the interface. All RAG logic lives in src/query.py - that
keeps the separation between "interface" and "core" clean. For a later voice
interface you only swap this file; the core stays the same.
"""

import streamlit as st

from src.query import load_query_engine

st.set_page_config(page_title="Survival101 RAG", page_icon="🏕️")
st.title("🏕️ Survival101")
st.caption("A RAG chatbot that answers only from curated survival sources.")


# @st.cache_resource makes sure the index + models are loaded only ONCE - not on
# every message. Without it the app would be unbearably slow.
@st.cache_resource(show_spinner="Loading knowledge base and models ...")
def get_engine():
    return load_query_engine()


engine = get_engine()

# Keep the chat history in session state (survives individual interactions).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the existing history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box at the bottom.
if question := st.chat_input("Ask a survival question ..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the sources and composing an answer ..."):
            answer = engine.query(question)
            st.markdown(str(answer))

            # Transparency: show which sources the answer came from.
            # This is didactic gold - you can see whether retrieval was good.
            with st.expander("Sources used"):
                for node in answer.source_nodes:
                    source = node.metadata.get("file_name", "unknown")
                    st.markdown(f"**{source}** (score: {node.score:.3f})")
                    st.caption(node.text[:300] + " ...")

    st.session_state.messages.append(
        {"role": "assistant", "content": str(answer)}
    )
