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

api_key = os.getenv("OPENAI_API_KEY_TRANSLATE")

if not api_key:
    st.error("Missing OpenAI API key.")
    st.stop()

st.title("Bilingual Chat Translator")

system_message = {
    "role": "system",
    "content": f"You are a bilingual conversation translator. User 1 prefers {user_1_language}, and User 2 prefers "
               f"{user_2_language}. When a user sends a message in either Cantonese, English, or a mix of both, "
               f"translate it fully into the language the other user prefers. Maintain meaning, structure, and "
               f"punctuation as closely as possible. Do not use em dashes or semicolons if the original text only "
               f"uses commas. Do not assume the message is always in one language, detect it dynamically.\n\n "
               f"**IMPORTANT: If translating into Cantonese, you must use written Cantonese with colloquial "
               f"vocabulary.** It is extremely important that you do not use Mandarin or Standard Chinese. Use words "
               f"and sentence structures common in Hong Kong written Cantonese. For example:\n "
               f"- Use '咗' for past tense instead of other forms.\n "
               f"- Use '佢哋' for 'they' instead of '他们'.\n "
               f"- Use '冇' for negation instead of '没有'.\n "
               f"- Use '喺' for prepositions like 'at'/'in'/'on' instead of '在'."
}

cantonese_system_message = {
    "role": "system",
    "content": "Remember, you must ensure that text containing exclusively English or a mix of both English and "
               "Cantonese must be translated fully into Cantonese using written Cantonese, not Mandarin or Standard "
               "Chinese. Use colloquial vocabulary and sentence structures common in Hong Kong written Cantonese. "
               "If the original text already contains Chinese characters, please try to only change the English text "
               "and maintain the original Cantonese text in terms of wording and structure, even if it is not "
               "colloquial Cantonese but rather is Standard Chinese. "
               "**However, there must be no English text left in the output!**"
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
    else:
        source_language = user_2_language
        target_language = user_1_language

    client = OpenAI(api_key=api_key)

    user_message = {"role": "user", "content": f"**{active_user} ({source_language}):**\n\n{user_input}"}
    st.session_state["messages"].append(user_message)
    st.chat_message("user").write(user_message["content"])

    messages = []

    if target_language == "Cantonese":
        messages.append(cantonese_system_message)

    messages.append({
                "role": "user",
                "content": f"Translate the following message from {source_language} to {target_language}: {user_input}"
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
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
