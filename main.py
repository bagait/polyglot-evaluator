import argparse
import warnings
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --- Model & Language Mappings ---
TRANSLATION_MODELS = {
    'de': 'Helsinki-NLP/opus-mt-en-de',  # German
    'es': 'Helsinki-NLP/opus-mt-en-es',  # Spanish
    'fr': 'Helsinki-NLP/opus-mt-en-fr',  # French
    'it': 'Helsinki-NLP/opus-mt-en-it',  # Italian
}

EVALUATION_MODEL = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'

# --- Core Functions ---

def load_data(dataset_name, split, num_samples):
    """Load a specified number of samples from a Hugging Face dataset."""
    print(f"\nLoading {num_samples} samples from '{dataset_name}' split '{split}'...")
    try:
        dataset = load_dataset(dataset_name, split=f'{split}[:{num_samples}]')
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and the dataset name is correct.")
        exit(1)

def translate_text(text, translator_pipeline):
    """Translate a single piece of text."""
    if not text or not isinstance(text, str):
        return ""
    # Some translation models have a max length, truncate to be safe
    truncated_text = text[:512]
    result = translator_pipeline(truncated_text)
    return result[0]['translation_text']

def evaluate_sentiment(dataset, sentiment_analyzer, text_column='text'):
    """Evaluate sentiment analysis accuracy on a dataset."""
    correct_predictions = 0
    predictions = []
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    original_label_map = {0: 'negative', 1: 'positive'}

    print(f"Evaluating sentiment on {len(dataset)} samples...")
    for item in tqdm(dataset):
        text = item[text_column]
        true_label_id = item['label']
        true_label_name = original_label_map[true_label_id]

        # Get model prediction
        pred = sentiment_analyzer(text)
        predicted_label_name = pred[0]['label'].lower()

        # Map roberta labels (LABEL_0, LABEL_1, LABEL_2) to readable names
        if predicted_label_name == 'label_0': predicted_label_name = 'negative'
        elif predicted_label_name == 'label_1': predicted_label_name = 'neutral'
        elif predicted_label_name == 'label_2': predicted_label_name = 'positive'

        if predicted_label_name == true_label_name:
            correct_predictions += 1
        predictions.append(predicted_label_name)

    accuracy = correct_predictions / len(dataset)
    return accuracy

def main(args):
    """Main execution function."""
    # 1. Load sentiment analysis pipeline
    print(f"Loading multilingual sentiment analysis model: {EVALUATION_MODEL}")
    sentiment_analyzer = pipeline('sentiment-analysis', model=EVALUATION_MODEL, tokenizer=EVALUATION_MODEL, device=0 if args.use_gpu else -1)

    # 2. Load and evaluate on the original English dataset
    original_dataset = load_data('imdb', 'test', args.num_samples)
    original_accuracy = evaluate_sentiment(original_dataset, sentiment_analyzer)

    results = [{'Language': 'English (Original)', 'Accuracy': original_accuracy}]

    # 3. Translate and evaluate for each target language
    for lang_code in args.languages:
        if lang_code not in TRANSLATION_MODELS:
            print(f"Warning: No translation model defined for '{lang_code}'. Skipping.")
            continue

        model_name = TRANSLATION_MODELS[lang_code]
        print(f"\n--- Processing language: {lang_code.upper()} using {model_name} ---")

        # Load translation model and tokenizer
        print("Loading translation model...")
        translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if args.use_gpu:
            translator_model = translator_model.to('cuda')
        translator_pipeline = pipeline('translation', model=translator_model, tokenizer=translator_tokenizer, device=0 if args.use_gpu else -1)

        # Translate dataset
        print("Translating dataset...")
        translated_texts = [translate_text(text, translator_pipeline) for text in tqdm(original_dataset['text'])]
        
        # Create a new dataset with translated text
        translated_dataset = original_dataset.map(lambda e, i: {'text': translated_texts[i]}, with_indices=True)

        # Evaluate on the translated dataset
        translated_accuracy = evaluate_sentiment(translated_dataset, sentiment_analyzer)
        results.append({'Language': f'{lang_code.upper()}', 'Accuracy': translated_accuracy})

    # 4. Display final report
    print("\n--- Evaluation Report ---")
    report_df = pd.DataFrame(results)
    report_df['Accuracy'] = report_df['Accuracy'].apply(lambda x: f"{x:.2%}")
    report_df['Performance Drop'] = pd.to_numeric(report_df['Accuracy'].str.replace('%', ''))
    original_acc_numeric = report_df.loc[0, 'Performance Drop']
    report_df['Performance Drop'] = report_df['Performance Drop'].apply(
        lambda x: f"-{(original_acc_numeric - x):.2f} pts"
    )
    report_df.loc[0, 'Performance Drop'] = "-"

    print(report_df.to_string(index=False))
    print("\nDone.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate NLP benchmarks and evaluate model performance degradation.")
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=50, 
        help='Number of samples to process from the dataset.'
    )
    parser.add_argument(
        '--languages', 
        nargs='+', 
        default=['de', 'es', 'fr'], 
        help=f'List of language codes to translate to. Supported: {list(TRANSLATION_MODELS.keys())}'
    )
    parser.add_argument(
        '--use_gpu', 
        action='store_true', 
        help='Enable GPU acceleration if a CUDA-enabled GPU is available.'
    )

    args = parser.parse_args()
    main(args)