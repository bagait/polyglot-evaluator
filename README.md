# Polyglot Evaluator

A command-line tool to automatically translate NLP benchmarks into multiple languages and evaluate how a model's performance degrades on the translated text. This helps quantify the impact of translation quality on multilingual model evaluation, a critical step when high-quality, human-translated datasets are not available.

This project is inspired by the challenges in creating reliable multilingual benchmarks and provides a practical way to probe model robustness across languages.



## Features

-   Loads data from any text-based Hugging Face dataset (example uses `imdb`).
-   Translates text from English to multiple target languages using `Helsinki-NLP` models.
-   Evaluates a multilingual sentiment model on the original and translated datasets.
-   Generates a clear report comparing accuracy and showing the performance drop for each language.
-   Supports GPU acceleration for faster processing.

## Installation

1.  Clone the repository:
    bash
    git clone https://github.com/bagait/polyglot-evaluator.git
    cd polyglot-evaluator
    

2.  Install the required Python packages. It is highly recommended to use a virtual environment.
    bash
    pip install -r requirements.txt
    

    *Note: If you have a CUDA-enabled GPU, ensure you have the correct version of PyTorch with CUDA support installed to use the `--use_gpu` flag.* 

## Usage

The main script `main.py` accepts several arguments to customize the evaluation.

**Basic Example**

Run the evaluation on 50 samples, translating to German, Spanish, and French:

bash
python main.py --num_samples 50


**Custom Languages and More Samples**

Run the evaluation on 100 samples, translating to German and Italian:

bash
python main.py --num_samples 100 --languages de it


**Using a GPU**

If you have a compatible GPU, you can significantly speed up the process:

bash
python main.py --num_samples 200 --use_gpu


### Example Output


--- Evaluation Report ---
     Language Accuracy Performance Drop
English (Original)   88.00%                -
              DE   72.00%      -16.00 pts
              ES   80.00%       -8.00 pts
              FR   74.00%      -14.00 pts

Done.


## How It Works

1.  **Load Data**: The script starts by loading a specified number of samples from the `imdb` dataset from the Hugging Face Hub. This dataset contains movie reviews labeled as positive or negative.

2.  **Load Evaluation Model**: A pre-trained multilingual sentiment analysis model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) is loaded. This model can classify sentiment in many languages.

3.  **Baseline Evaluation**: The model's accuracy is first calculated on the original English text. This serves as the baseline performance.

4.  **Translation Loop**: The script iterates through the list of target languages provided by the user.
    -   For each language, it dynamically loads the appropriate `Helsinki-NLP/opus-mt-en-[lang]` translation model.
    -   It then translates the text of each sample from the original dataset.

5.  **Translated Evaluation**: The multilingual sentiment model is run on the newly translated dataset, and a new accuracy score is computed.

6.  **Report Generation**: After all languages are processed, a `pandas` DataFrame is created to summarize the results. It shows the accuracy for each language and calculates the performance drop relative to the original English baseline, highlighting the impact of translation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
