import streamlit as st
import whisper
import torch
import os
import wave
import pyaudio
import tempfile
from openai import OpenAI
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

torch.classes.__path__ = []

Base = declarative_base()

# audio recording params
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024


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

script_map = {
    'Simplified 简体字':
        {
            'title': '粤语英文翻译机',
            'lang': '语言',
            'cantonese': '粤语',
            'prompt': '我啱啱食完lunch，好饱啊。你今晚有冇兴趣去party？We can go together.',
            'start': '开始',
            'record': '录音',
            'translate': '翻译',
            'audio': '音频',
            'enter': '输入'
        },
    'Traditional 繁體字':
        {
            'title': '粵語英文翻譯機',
            'lang': '語言',
            'cantonese': '粵語',
            'prompt': '我啱啱食完lunch，好飽啊。你今晚有冇興趣去party？We can go together.',
            'start': '開始',
            'record': '錄音',
            'translate': '翻譯',
            'audio': '音頻',
            'enter': '輸入'
        }
}

if 'messages' not in st.session_state:
    rows = session.query(Message).all()
    st.session_state['messages'] = [
        {'user': row.user, 'raw_text': row.raw_text, 'translated_text': row.translated_text} for row in rows
    ]

st.set_page_config(page_title="Cantonese-English Translator", page_icon='🌐')

with st.sidebar:
    options = ['Simplified 简体字', 'Traditional 繁體字']
    script = st.pills('Script 简繁转换 / 簡繁轉換', options, default=options[0])

    user_1_language = st.selectbox(
        label=f"User 1's Language | 用戶1的{script_map[script]['lang']}",
        options=[f"Cantonese {script_map[script]['cantonese']}", "English 英文"],
        index=0
    )

    user_2_language = st.selectbox(
        label=f"User 2's Language | 用戶2的{script_map[script]['lang']}",
        options=[f"Cantonese {script_map[script]['cantonese']}", "English 英文"],
        index=1
    )

st.title(f'🌐 Cantonese-English Translator | {script_map[script]["title"]}', anchor='translator')


def record_audio(filename: str = 'output.wav', record_seconds: int = 10) -> str:
    """
    Record audio using PyAudio.
    :param filename: A string corresponding to the name of the output audio file.
    :param record_seconds: An integer corresponding to the duration of the recording in seconds.
    :return: The name of the audio file.
    """
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []

    timer_placeholder = st.empty()

    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

        elapsed = i / (RATE / CHUNK)
        timer_placeholder.markdown(f'{elapsed:.1f}s / {record_seconds:.1f}s')

    stream.stop_stream()
    stream.close()
    audio.terminate()

    timer_placeholder.empty()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


def transcribe_and_translate(input_str: str, source_language: str, target_language: str, mode: str) -> tuple[str, str]:
    """
    Transcribe the provided audio file using Whisper and translate the corresponding transcription from the
    source language to the target language using GPT-4o-mini.
    :param input_str: If transcribing, a string corresponding to the name of the input audio file.
    If translating, the input string.
    :param source_language: A string corresponding to the source language from which the text is to be translated.
    :param target_language: A string corresponding to the target language into which the text is to be translated.
    :param mode: A string corresponding to the mode in which the input should be handled: 'transcribe' or 'translate'.
    :return: A tuple containing the raw text output from Whisper and the translated text output from GPT-4o-mini.
    """
    if mode == 'transcribe':
        asr_model = whisper.load_model('medium').to(device)

        transcription = asr_model.transcribe(
            audio=input_str,
            language='zh',
            task='transcribe',
            initial_prompt=script_map[script]['prompt']
        )

        raw_text = transcription['text']

        del asr_model
        torch.cuda.empty_cache()

    else:
        raw_text = input_str

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

    return raw_text, translated_text


st.markdown('---')

with st.container(height=400):
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
st.subheader('New Message 新信息', anchor='new-message')

tab1, tab2, tab3 = st.tabs([
    f'Speech {script_map[script]["record"]}',
    f'File {script_map[script]["audio"]}',
    'Text 打字'
])

with tab1:
    left_col, right_col = st.columns(2, border=True)

    start_rec = script_map[script]['start'] + script_map[script]['record']

    with left_col:
        st.markdown('#### User 1 | 用戶1')
        st.markdown(f'{user_1_language}')
        if st.button(label='🎙️', key='record1', help=f'Start recording | {start_rec}', use_container_width=True):
            with st.spinner(f'Recording... 正在{script_map[script]["record"]}...'):
                wav_path = record_audio()
            with st.spinner(f'Translating... 正在{script_map[script]["translate"]}...'):
                raw, translated = transcribe_and_translate(wav_path, user_1_language, user_2_language, 'transcribe')
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
            os.remove(wav_path)
            st.rerun()

    with right_col:
        st.markdown('#### User 2 | 用戶2')
        st.markdown(f'{user_2_language}')
        if st.button('🎙️', key='record2', help=f'Start recording | {start_rec}', use_container_width=True):
            with st.spinner(f'Recording... 正在{script_map[script]["record"]}...'):
                wav_path = record_audio()
            with st.spinner(f'Translating... 正在{script_map[script]["translate"]}...'):
                raw, translated = transcribe_and_translate(wav_path, user_2_language, user_1_language, 'transcribe')
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
            os.remove(wav_path)
            st.rerun()

with tab2:
    left_col, right_col = st.columns(2, border=True)

    audio = script_map[script]['audio']

    with left_col:
        st.markdown('#### User 1 | 用戶1')
        st.markdown(f'{user_1_language}')
        audio_file_1 = st.file_uploader(f'User 1 Audio | 用戶1的{audio}', type=['.wav'], key='uploader1')
        if st.button(label=f'Translate {script_map[script]["translate"]}', key='translate1-file') and audio_file_1:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file_1.read())
                tmp_path = tmp.name
            with st.spinner(f'Translating... 正在{script_map[script]["translate"]}...'):
                raw, translated = transcribe_and_translate(tmp_path, user_1_language, user_2_language, 'transcribe')
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
            os.remove(tmp_path)
            st.rerun()

    with right_col:
        st.markdown('#### User 2 | 用戶2')
        st.markdown(f'{user_2_language}')
        audio_file_2 = st.file_uploader(f'User 2 Audio | 用戶2的{audio}', type=['.wav'], key='uploader2')
        if st.button(label=f'Translate {script_map[script]["translate"]}', key='translate2-file') and audio_file_2:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file_2.read())
                tmp_path = tmp.name
            with st.spinner(f'Translating... 正在{script_map[script]["translate"]}...'):
                raw, translated = transcribe_and_translate(tmp_path, user_2_language, user_1_language, 'transcribe')
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
            os.remove(tmp_path)
            st.rerun()

with tab3:
    left_col, right_col = st.columns(2, border=True)

    enter_text = script_map[script]['enter'] + '文字'

    with left_col:
        st.markdown('#### User 1 | 用戶1')
        st.markdown(f'{user_1_language}')
        text1 = st.text_input('User 1 text', placeholder=f'Enter text {enter_text}', label_visibility='collapsed')
        if st.button(label=f'Translate {script_map[script]["translate"]}', key='translate1-text') and text1:
            with st.spinner(f'Translating... 正在{script_map[script]["translate"]}...'):
                raw, translated = transcribe_and_translate(text1, user_1_language, user_2_language, 'translate')
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
        st.markdown('#### User 2 | 用戶2')
        st.markdown(f'{user_2_language}')
        text2 = st.text_input('User 2 text', placeholder=f'Enter text {enter_text}', label_visibility='collapsed')
        if st.button(label=f'Translate {script_map[script]["translate"]}', key='translate2-text') and text2:
            with st.spinner(f'Translating... 正在{script_map[script]["translate"]}...'):
                raw, translated = transcribe_and_translate(text2, user_2_language, user_1_language, 'translate')
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
