This repository implements a sentiment analysis model on the IMDB movie review dataset using a fine-tuned [RoBERTa-base](https://huggingface.co/roberta-base) model. The classification head leverages the [CLS] token for binary sentiment classification (positive vs. negative).

## Model Overview

- **Base Model**: `roberta-base` (from Hugging Face)
- **Classification Head**: Dropout + Linear layer over the `[CLS]` token
- **Loss Function**: CrossEntropyLoss
- **Tokenizer**: RoBERTa tokenizer with max length 128

##  Results

-  **Test Accuracy**: 87.94%
-  **F1 Score**: 87.41%
-  Confusion Matrix:

| True \ Pred | Negative | Positive |
|-------------|----------|----------|
| **Negative**| 11519    | 981      |
| **Positive**| 2034     | 10000    |

##  Usage

1. Clone the repo and install dependencies:
    ```bash
    git clone https://github.com/your-username/imdb-sentiment-roberta-cls.git
    cd imdb-sentiment-roberta-cls
    pip install -r requirements.txt
    ```

2. Train the model:
    ```bash
    python train.py
    ```

3. Evaluate and visualize:
    ```bash
    python evaluate.py
    ```

##  Dependencies

- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `matplotlib`

(See `requirements.txt` for full list)

##  Dataset

- IMDB reviews dataset (binary classification)
- Loaded from HuggingFace: `stanfordnlp/imdb`

##  Author

- [Huangyinlin Zhang], University of Michigan
