import argparse
import re
from Levenshtein import distance as levenshtein_distance
from jiwer import wer


def split_lang(text: str) -> tuple[str, str]:
    """
    Split Cantonese and English from input text.
    :param text: A string containing Cantonese-English code-switched text.
    :return: A tuple containing the Cantonese text and the English text.
    """
    cantonese = ''.join(char for char in text if '\u4e00' <= char <= '\u9fff')
    english = ' '.join(re.findall(r'[a-zA-Z0-9]+', text))
    return cantonese, english


def compute_cantonese_cer(gold: str, predicted: str) -> float:
    """
    Given Cantonese gold and predicted strings, compute the Levenshtein distance.
    :param gold: The Cantonese gold string.
    :param predicted: The Cantonese predicted string.
    :return: The Levenshtein distance between the Cantonese gold and predicted strings.
    """
    if not gold:
        return 1.0 if predicted else 0.0
    return levenshtein_distance(gold, predicted) / len(gold) if len(gold) > 0 else 0.0


def compute_english_wer(gold: str, predicted: str) -> float:
    """
    Given English gold and predicted strings, compute the word error rate.
    :param gold: The English gold string.
    :param predicted: The English predicted string.
    :return: The word error rate between the English gold and predicted strings.
    """
    gold = gold.lower()
    predicted = predicted.lower()
    return wer(gold, predicted)


def evaluate_sample(gold: str, predicted: str) -> dict:
    """
    Evaluate a predicted string against its gold counterpart.
    :param gold: The Cantonese-English code-switched gold string.
    :param predicted: The Cantonese-English code-switched predicted string.
    :return: A dict containing gold and pred raw text, Cantonese and English split text, Cantonese CER, and English WER.
    """
    gold_can, gold_en = split_lang(gold)
    predicted_can, predicted_en = split_lang(predicted)

    cantonese_cer = compute_cantonese_cer(gold_can, predicted_can)
    english_wer = compute_english_wer(gold_en, predicted_en)

    return {
        "Gold": gold,
        "Predicted": predicted,
        "Gold Cantonese": gold_can,
        "Pred Cantonese": predicted_can,
        "Gold English": gold_en,
        "Pred English": predicted_en,
        "Cantonese CER": cantonese_cer,
        "English WER": english_wer
    }


def get_results(gold_text: list[str], predicted_text: list[str], output_txt: str) -> None:
    """
    Retrieve the evaluation results for a batch of Cantonese-English code-switched text.
    :param gold_text: A list of Cantonese-English code-switched gold text.
    :param predicted_text: A list Cantonese-English code-switched predicted text.
    :param output_txt: A .txt file containing batched evaluation results.
    """
    results = [evaluate_sample(gold, pred) for gold, pred in zip(gold_text, predicted_text)]

    with open(output_txt, 'w', encoding='utf-8') as out:
        for i, result in enumerate(results, 1):
            print(f"Sample {i}", file=out)
            print(f"Gold: {result['Gold']}", file=out)
            print(f"Predicted: {result['Predicted']}", file=out)
            print(f"Cantonese CER: {result['Cantonese CER']:.3f}", file=out)
            print(f"English WER: {result['English WER']:.3f}", file=out)
            print('-' * 60, file=out)

        cantonese_cer = [result['Cantonese CER'] for result in results]
        english_wer = [result['English WER'] for result in results]

        avg_cer = sum(cantonese_cer) / len(cantonese_cer) if len(cantonese_cer) > 0 else 0.0
        avg_wer = sum(english_wer) / len(english_wer) if len(english_wer) > 0 else 0.0

        print(f"Average Cantonese CER: {avg_cer:.3f}", file=out)
        print(f"Average English WER: {avg_wer:.3f}", file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluate_whisper.py',
        description='Evaluates Whisper output.'
    )
    parser.add_argument(
        'gold_txt',
        type=str,
        help='A string representing the name of the .txt file containing gold transcriptions.'
    )
    parser.add_argument(
        'predicted_txt',
        type=str,
        help='A string representing the name of the .txt file containing predicted transcriptions.'
    )
    parser.add_argument(
        'output_txt',
        type=str,
        help='A string representing the name of the .txt file to write the evaluation results.'
    )
    args = parser.parse_args()

    gold_transcriptions = []
    predicted_transcriptions = []

    with open(args.gold_txt, 'r', encoding='utf-8') as f:
        for line in f:
            gold_transcriptions.append(line.strip())

    with open(args.predicted_txt, 'r', encoding='utf-8') as f:
        for line in f:
            predicted_transcriptions.append(line.strip())

    get_results(gold_transcriptions, predicted_transcriptions, args.output_txt)
