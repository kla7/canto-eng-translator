import streamlit as st
import whisper
import torch
import tempfile
import os
from openai import OpenAI
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

torch.classes.__path__ = []

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_TRANSLATE"))

if 'messages' not in st.session_state:
    rows = session.query(Message).all()
    st.session_state['messages'] = [
        {'user': row.user, 'raw_text': row.raw_text, 'translated_text': row.translated_text} for row in rows
    ]

st.set_page_config(page_title="Cantonese-English Translator", page_icon='🌐')

st.title('🌐 Cantonese-English Translator | 粵語英文翻譯機', anchor='translator')

with st.sidebar:
    user_1_language = st.selectbox(
        label="User 1's Language | 用戶1的語言",
        options=["Cantonese 粵語", "English 英文"],
        index=0
    )

    user_2_language = st.selectbox(
        label="User 2's Language | 用戶2的語言",
        options=["Cantonese 粵語", "English 英文"],
        index=1
    )


def transcribe_and_translate(uploaded_file, source_language, target_language):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    asr_model = whisper.load_model('medium').to(device)

    transcription = asr_model.transcribe(
        audio=tmp_path,
        language='zh',
        task='transcribe',
        initial_prompt='我啱啱食完lunch，好飽啊。你今晚有冇興趣去party？。'
    )

    raw_text = transcription['text']
    print(raw_text)

    os.remove(tmp_path)

    del asr_model
    torch.cuda.empty_cache()

    system_prompt = {
        "role": "system",
        "content": f"You are a bilingual conversation translator. User 1 prefers {user_1_language}, and User 2 prefers "
                   f"{user_2_language}. When a user sends a message in either Cantonese, English, or a mix of both, "
                   f"translate it fully into the language the other user prefers. Maintain meaning, structure, and "
                   f"punctuation as closely as possible. Do not use em dashes or semicolons if the original text only "
                   f"uses commas. Do not assume the message is always in one language, detect it dynamically.\n\n "
                   f"**IMPORTANT: If translating into Cantonese, you must use written Cantonese with colloquial "
                   f"vocabulary.** It is extremely important that you do not use Mandarin or Standard Chinese. Use "
                   f"words and sentence structures common in Hong Kong written Cantonese. For example:\n "
                   f"- Use '咗' for past tense instead of other forms.\n "
                   f"- Use '佢哋' for 'they' instead of '他们'.\n "
                   f"- Use '冇' for negation instead of '没有'.\n "
                   f"- Use '喺' for prepositions like 'at'/'in'/'on' instead of '在'.\n"
                   f"Remember, you must ensure that text containing exclusively English or a mix of both English and "
                   f"Cantonese must be translated fully into Cantonese using written Cantonese, not Mandarin or "
                   f"Standard Chinese. Use colloquial vocabulary and sentence structures common in Hong Kong written "
                   f"Cantonese. If the original text already contains Chinese characters, please try to only change "
                   f"the English text and maintain the original Cantonese text in terms of wording and structure, even "
                   f"if it is not colloquial Cantonese but rather is Standard Chinese. "
                   f"**However, there must be no English text left in the output!**"
    }

    user_prompt = {
        "role": "user",
        "content": f"Translate the following message from {source_language} to {target_language}: {raw_text}"
    }

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, user_prompt]
    )

    translated_text = response.choices[0].message.content.strip()
    print(translated_text)

    return raw_text, translated_text


st.markdown('---')

for msg in (st.session_state['messages']):
    align = 'left' if msg['user'] == "User 1" else 'right'
    bubble_color = '#0492d4' if msg['user'] == 'User 1' else '#09bd0f'

    st.markdown(
        f"""
        <div style='text-align: {align}; margin-bottom: 1rem;'>
            <div style='display: inline-block; background-color: {bubble_color}; padding: 10px 15px;
            border-radius: 15px; max-width: 80%; text-align: left;'>
                {msg['raw_text']}<br><br>
                <em>{msg['translated_text']}</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('---')
st.subheader('New Message 新信息')


left_col, right_col = st.columns(2)

with left_col:
    audio_file_1 = st.file_uploader('User 1 Audio | 用戶1的音頻', type=['.wav'], key='uploader1')
    if st.button(label='Translate 翻譯', key='translate1') and audio_file_1:
        raw, translated = transcribe_and_translate(audio_file_1, user_1_language, user_2_language)
        message = {'user': 'User 1', 'raw_text': raw, 'translated_text': translated}
        st.session_state['messages'].append(message)
        session.add(Message(
            user='User 1',
            raw_text=raw,
            translated_text=translated,
            source_language=user_1_language,
            target_language=user_2_language
        ))
        session.commit()
        st.rerun()

with right_col:
    audio_file_2 = st.file_uploader('User 2 Audio | 用戶2的音頻', type=['.wav'], key='uploader2')
    if st.button(label='Translate 翻譯', key='translate2') and audio_file_2:
        raw, translated = transcribe_and_translate(audio_file_2, user_2_language, user_1_language)
        message = {'user': 'User 2', 'raw_text': raw, 'translated_text': translated}
        st.session_state['messages'].append(message)
        session.add(Message(
            user='User 2',
            raw_text=raw,
            translated_text=translated,
            source_language=user_2_language,
            target_language=user_1_language
        ))
        session.commit()
        st.rerun()

st.markdown('---')

if st.button('Clear 清除'):
    st.session_state['messages'] = []
    session.query(Message).delete()
    session.commit()
    st.rerun()
