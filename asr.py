import whisper
import torch
import os
import re
import argparse


def transcribe(audio: str, output_txt: str, prompt_on: bool) -> None:
    """
    Transcribe the provided audio.
    :param audio: An audio file containing speech data.
    :param output_txt: An output txt in which the transcription will be written.
    :param prompt_on: If true, Whisper will be given an initial prompt.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = whisper.load_model('medium').to(device)

    transcription = model.transcribe(
        audio=audio,
        language='zh',
        task='transcribe',
        initial_prompt='我啱啱食完lunch，好飽啊。你今晚有冇興趣去party？We can go together.' if prompt_on else ''  # sample
    )

    with open(output_txt, 'a', newline='', encoding='utf-8') as f:
        audio_id = os.path.splitext(os.path.basename(audio))[0]
        transcribed_text = transcription['text']
        f.write(f'{transcribed_text}\n')

    print(audio_id, transcribed_text)


def numeric_sort(key: str) -> list:
    """
    Customized sorting so that file IDs can be sorted numerically instead of alphabetically.
    :param key: A file ID.
    :return: A list splitting the characters in the ID into digits and non-digits.
    """
    return [int(char) if char.isdigit() else char for char in re.split(r'(\d+)', key)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='asr.py',
        description='Standalone ASR script that transcribes a batch of audio files with Whisper.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='A string representing the name of the directory containing the audio files to be transcribed.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='A string representing the name of the output directory where the transcription will be written.'
    )
    parser.add_argument(
        '-u', '--unprompted',
        action='store_true',
        help='A boolean determining whether the transcriptions should be created without the initial prompt (baseline).'
    )
    parser.add_argument(
        '-p', '--prompted',
        action='store_true',
        help='A boolean determining whether the transcriptions should be created with the initial prompt.'
    )
    args = parser.parse_args()

    directory = args.input_dir
    audio_files = os.listdir(directory)
    audio_files_sorted = sorted(audio_files, key=numeric_sort)

    filename = os.path.basename(directory)

    if args.unprompted:
        out_file = os.path.join(args.output_dir, f'predicted_baseline_{filename}.txt')
        for audio_file in audio_files_sorted:
            path = os.path.join(directory, audio_file)
            transcribe(path, out_file, prompt_on=False)

    if args.prompted:
        out_file = os.path.join(args.output_dir, f'predicted_prompted_{filename}.txt')
        for audio_file in audio_files_sorted:
            path = os.path.join(directory, audio_file)
            transcribe(path, out_file, prompt_on=True)
