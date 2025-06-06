"""
Streamlit UI for the trail-weather chatbot.
-------------------------------------------------
Run with:
    streamlit run app.py

Prerequisites (install via pip/conda as needed):
    streamlit>=1.28  # for st.chat_* components
    langgraph, langchain, pydantic, python-dotenv, geocoder, numpy, pandas, scikit-learn
    plus whatever you already use in trail_weather_graph.py

Environment variables expected by trail_weather_graph.py can be set in a .env file or
through the sidebar inputs at runtime:
    OPENWEATHER_API_KEY=<your-key>
    GOOGLE_API_KEY=<your-key>
"""

import os

import streamlit as st

from trail_weather_graph import build_graph

# ──────────────────────────────  PAGE CONFIG  ─────────────────────────────
st.set_page_config(
    page_title="Trail & Weather Chatbot",
    page_icon="🌲",
    layout="centered",
)

# ──────────────────────────────  INITIALISATION  ─────────────────────────
# Build the LangGraph once per session
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Conversation dictionary mirrors the State type in trail_weather_graph.py
if "bot_state" not in st.session_state:
    st.session_state.bot_state = {"messages": []}

# ──────────────────────────────  SIDEBAR  ────────────────────────────────
with st.sidebar:
    st.header("🔧 Settings")

    # Allow the user to paste API keys without editing .env
    openweather_key = st.text_input(
        "OpenWeather API Key",
        value=os.getenv("OPENWEATHER_API_KEY", ""),
        type="password",
    )
    google_key = st.text_input(
        "Google Maps API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password"
    )
    if st.button("💾 Save keys"):
        os.environ["OPENWEATHER_API_KEY"] = openweather_key
        os.environ["GOOGLE_API_KEY"] = google_key
        st.success("Keys updated for this session.")

    st.markdown("---")
    if st.button("🧹 Clear chat history"):
        st.session_state.bot_state = {"messages": []}
        st.rerun()

# ──────────────────────────────  CHAT HISTORY  ───────────────────────────
for msg in st.session_state.bot_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ──────────────────────────────  INPUT & RESPONSE  ───────────────────────
user_prompt = st.chat_input("Ask me about trails…")
if user_prompt:
    # Store the user message in the graph state first
    st.session_state.bot_state["messages"].append(
        {"role": "user", "content": user_prompt}
    )

    # Echo the user message immediately (appears on the right by default)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Run the LangGraph to get the assistant reply
    with st.spinner("Thinking…"):
        st.session_state.bot_state = st.session_state.graph.invoke(
            st.session_state.bot_state, {"recursion_limit": 100}
        )

    # Assistant reply is always the last element
    assistant_reply = st.session_state.bot_state["messages"][-1]
    with st.chat_message("assistant"):
        st.markdown(assistant_reply.content)

    # No explicit rerun needed – the current script already rendered the
    # new messages above.  Comment out the following line if you prefer the
    # older auto‑refresh behaviour.
    # st.rerun()
