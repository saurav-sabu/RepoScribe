import streamlit as st
from pathlib import Path
import shutil
from main import readme_generation_team  # import your Team

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "repo"

def reset_repo():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.mkdir(parents=True, exist_ok=True)

# Run cleanup ONLY once per session
if "initialized" not in st.session_state:
    reset_repo()
    st.session_state.initialized = True
    st.session_state.messages = []


st.set_page_config(
    page_title="RepoScribe",
    page_icon="📘",
    layout="wide"
)

st.title("📘 RepoScribe")
st.caption("Chat with your README generation agents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something (e.g. Paste GitHub repo URL)")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run team
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = readme_generation_team.run(
                user_input,
                stream=False  # set True if you want streaming
            )

            # Extract text safely
            if hasattr(response, "content"):
                assistant_text = response.content
            else:
                assistant_text = str(response)

            st.markdown(assistant_text)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text}
    )
