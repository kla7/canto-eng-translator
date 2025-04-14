import streamlit as st
from openai import OpenAI
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user = Column(String)
    raw_text = Column(String)
    translated_text = Column(String)
    source_language = Column(String)
    target_language = Column(String)


db = "sqlite:///./database.db"
engine = create_engine(db)
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()

with st.sidebar:
    user_1_language = st.selectbox(
        label="User 1's Language",
        options=["Cantonese", "English"],
        index=0
    )

    user_2_language = st.selectbox(
        label="User 2's Language",
        options=["Cantonese", "English"],
        index=1
    )

api_key_can = os.getenv("OPENAI_API_KEY_CAN")
api_key_en = os.getenv("OPENAI_API_KEY_EN")

if not api_key_can or not api_key_en:
    st.error("Missing OpenAI API key.")
    st.stop()

st.title("Bilingual Chat Translator")

system_message = {
    "role": "system",
    "content": f"You are a bilingual conversation assistant. User 1 prefers {user_1_language}, and User 2 prefers "
               f"{user_2_language}. When a user sends a message in either Cantonese, English, or a mix of both, "
               f"translate it into the language the other user prefers. Maintain meaning, structure, and punctuation "
               f"as closely as possible. Do not use em dashes or semicolons if the original text only uses commas. "
               f"Do not assume the message is always in one language, detect it dynamically. IMPORTANT: If "
               f" into Cantonese, use written Cantonese with colloquial vocabulary, not Mandarin (Standard Chinese). "
               f"Use words and sentence structures common in Hong Kong written Cantonese."
}

if "messages" not in st.session_state:
    st.session_state["messages"] = [system_message]

    stored_messages = session.query(Message).all()

    for msg in stored_messages:
        st.session_state["messages"].append({
            "role": "user",
            "content": f"**{msg.user} ({msg.source_language}):**\n\n{msg.raw_text}"
        })

        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"**Translated to {msg.target_language}:**\n\n{msg.translated_text}"
        })

for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        role = "user" if "User" in msg["content"] else "assistant"
        st.chat_message(role).write(msg["content"])

active_user = st.radio(
    label="Who is speaking?",
    options=["User 1", "User 2"],
    horizontal=True
)

user_input = st.chat_input(f"{active_user}, enter your message here")

if user_input:
    if active_user == "User 1":
        source_language = user_1_language
        target_language = user_2_language
        api_key = api_key_can if source_language == "Cantonese" else api_key_en
    else:
        source_language = user_2_language
        target_language = user_1_language
        api_key = api_key_can if source_language == "Cantonese" else api_key_en

    client = OpenAI(api_key=api_key)

    user_message = {"role": "user", "content": f"**{active_user} ({source_language}):**\n\n{user_input}"}
    st.session_state["messages"].append(user_message)
    st.chat_message("user").write(user_message["content"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            system_message,
            {
                "role": "user",
                "content": f"Translate the following message from {source_language} to {target_language}: {user_input}"
            }
        ]
    )

    translated_text = response.choices[0].message.content.strip()

    assistant_message = {"role": "assistant", "content": f"**Translated to {target_language}:**\n\n{translated_text}"}
    st.session_state["messages"].append(assistant_message)
    st.chat_message("assistant").write(assistant_message["content"])

    session.add(Message(
        user=active_user,
        raw_text=user_input,
        translated_text=translated_text,
        source_language=source_language,
        target_language=target_language
    ))

    session.commit()
    st.rerun()
