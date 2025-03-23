import streamlit as st
from openai import OpenAI
import os

with st.sidebar:
    language = st.sidebar.selectbox(
        label="Select language",
        options=["Cantonese", "English"],
        index=0
    )

api_key_can = os.getenv("OPENAI_API_KEY_CAN")
api_key_en = os.getenv("OPENAI_API_KEY_EN")

if not api_key_can or not api_key_en:
    st.error("Missing OpenAI API key.")
    st.stop()

api_key = api_key_can if language == "Cantonese" else api_key_en

st.title("Translate")

system_message = {
    "role": "system",
    "content": f"You are a professional translator. Given text containing Cantonese-English code-switching, "
               f"translate it into {language} only. Preserve the original meaning and structure as much as possible. "
               f"Use punctuation similar to the original text. Do not use em dashes or semicolons if the original "
               f"only uses commas."
}

if "messages" not in st.session_state:
    st.session_state["messages"] = []

messages_with_system = [system_message] + st.session_state.messages

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    client = OpenAI(api_key=api_key)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_system + [{"role": "user", "content": prompt}]
    )

    msg = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
