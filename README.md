# Cantonese-English Live Conversation Translator

**Author:** Kasey La (kaseyla@brandeis.edu)

## Background

This project aims to create a Cantonese-English translation app that can support live conversations between two users
that may have difficulties communicating with one another due to language barriers. Conversations that this app can
support include those between:

1. A Cantonese speaker who does not speak/understand any English with an English speaker who does not
   speak/understand any Cantonese
2. A speaker who can speak/understand both languages but is not fully fluent in their non-dominant language, thus they
   may tend to mix in their dominant language when speaking their non-dominant language (i.e., code-switching), with a
   speaker who only speaks the first speaker's non-dominant language

The second conversation type is most prevalent for the Chinese-American community where most of the younger generation
are heritage speakers of Cantonese and may have issues communicating with older members of their family who do not
speak English. These heritage speakers typically are able to fare well with listening but have trouble on the speaking
front, specifically in cases where they may struggle with a limited Cantonese vocabulary. This project can be used to
support these conversations between the generations.

## App Features

* A toggle to switch between using Simplified Chinese script and Traditional Chinese script.
* An option to select which primary language (Cantonese or English) is being used for each user.
    * This language is the one that the other user's messages will be translated into.
* A log for conversation messages, which restores conversation history when the app is restarted.
    * All messages can be cleared using the "Clear" button.
* 3 input methods: speech (maximum 10 seconds), file upload, text
* Pipeline: ASR - OpenAI Whisper (medium), Machine Translation - OpenAI GPT-4o mini, frontend - Streamlit, backend - SQLAlchemy
    * Cantonese colloquial vocabulary is supported for both ASR transcriptions and translations.

## Instructions

The instructions for running the app are as follows:

### Requirements

* Python 3.11 or higher
* Install dependencies from [requirements.txt](https://github.com/kla7/canto-eng-translator/blob/main/requirements.txt)
* Get an OpenAI API key, setting the environment variable name to `OPENAI_API_KEY_TRANSLATE`

### Streamlit

To run the app:
* In your terminal, run
```
streamlit run translate.py
```

* The default server is http://localhost:8501/

## Contents of this repository

This folder contains 6 files:

1. This **README** file.
2. **translate.py**, a script containing the Streamlit app and the full ASR-MT pipeline.
3. **asr.py**, a script containing standalone ASR for evaluation purposes.
    * Can be run from command line. Run `python asr.py -h` for the help menu.
4. **evaluate_whisper.py**, a script containing an evaluation script for Whisper outputs.
    * Can be run from command line. Run `python evaluate_whisper.py -h` for the help menu.
5. **presentation.pdf**, presentation slides that contain a brief summary of the project.
6. **requirements.txt**, the dependencies for running the project.