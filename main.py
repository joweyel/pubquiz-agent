import streamlit as st
from src.agent import run_llm

st.set_page_config(page_title="LangChain Pub-Quiz Agent", layout="centered")

st.title("Pub Quiz Chatbot")

with st.sidebar:
    st.subheader("Clear Chat")
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ich bin ein Chat-Bot für DAS Pub-Quiz.",
            "sources": [],
        }
    ]

# Iterate iver akk messages and show theem at each rerun in the ui
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools_used"):
            st.caption(f"Tools used: {', '.join(message['tools_used'])}")
        # Show sources if there are any
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message.get("sources"):
                    st.markdown(f"- {source}")

# Obtain prompt from user to send to agent
if prompt := st.chat_input("Was ist deine Frage?"):
    chat_memory: list[dict[str, str]] = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # Get results from agent
                result = run_llm(prompt, chat_memory)
                answer = str(result.get("answer") or "(No answer returned.)").strip()

            st.markdown(answer)
            if tools_used := result.get("tools_used", []):
                st.markdown(f"**Tools used**: {', '.join(tools_used)}")

            if sources := result.get("sources", []):
                sources = list(set(sources))
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "tools_used": tools_used,
                }
            )

        except Exception as ex:
            st.error("Error while generating a response!")
            st.exception(ex)
