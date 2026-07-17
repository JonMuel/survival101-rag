"""
STREAMLIT CHAT-APP  (das, was die Nutzer auf deiner Webseite sehen)
===================================================================

Startet die Web-Oberflaeche:

    streamlit run app.py

Diese Datei kuemmert sich NUR um die Oberflaeche. Die gesamte RAG-Logik liegt
in src/query.py - so bleibt die Trennung zwischen "Interface" und "Kern" sauber.
Fuer ein spaeteres Voice-Interface tauschst du nur diese Datei aus, der Kern
bleibt gleich.
"""

import streamlit as st

from src.query import load_query_engine

st.set_page_config(page_title="Survival101 RAG", page_icon="🏕️")
st.title("🏕️ Survival101")
st.caption("Ein RAG-Chatbot, der nur auf Basis kuratierter Survival-Quellen antwortet.")


# @st.cache_resource sorgt dafuer, dass Index + Modelle nur EINMAL geladen
# werden - nicht bei jeder Nachricht. Ohne das waere die App unertraeglich langsam.
@st.cache_resource(show_spinner="Lade Wissensbasis und Modelle ...")
def get_engine():
    return load_query_engine()


engine = get_engine()

# Chat-Verlauf im Session-State halten (ueberlebt einzelne Interaktionen).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Bisherigen Verlauf anzeigen.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Eingabefeld unten.
if frage := st.chat_input("Stell eine Survival-Frage ..."):
    st.session_state.messages.append({"role": "user", "content": frage})
    with st.chat_message("user"):
        st.markdown(frage)

    with st.chat_message("assistant"):
        with st.spinner("Suche in den Quellen und formuliere Antwort ..."):
            antwort = engine.query(frage)
            st.markdown(str(antwort))

            # Transparenz: zeige, aus welchen Quellen die Antwort stammt.
            # Das ist didaktisch Gold - du siehst, ob das Retrieval gut war.
            with st.expander("Verwendete Quellen"):
                for node in antwort.source_nodes:
                    quelle = node.metadata.get("file_name", "unbekannt")
                    st.markdown(f"**{quelle}** (Score: {node.score:.3f})")
                    st.caption(node.text[:300] + " ...")

    st.session_state.messages.append(
        {"role": "assistant", "content": str(antwort)}
    )
