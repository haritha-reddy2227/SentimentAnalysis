# SentimentAnalysis

A simple sentiment analysis project using BERT and Hugging Face Transformers.

This repository includes a synthetic dataset generator, model training, and a Streamlit inference app for predicting negative, neutral, or positive sentiment.

## Repository structure

- `app/`
  - `app.py` - Streamlit application for interactive sentiment prediction.
- `dataset/`
  - `sentiment.csv` - generated sentiment dataset used for training.
- `model/`
  - Saved pretrained tokenizer and classification model files.
- `results/`
  - Training checkpoints and output from model training.
- `generate_dataset.py`
  - Script to generate a synthetic sentiment dataset.
- `train.py`
  - Script to train BERT on the generated dataset and save the model.
- `bert_test.py`
  - Quick test script using the Hugging Face pipeline for sentiment inference.
- `requirements.txt`
  - Python dependencies for running and training the project.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:
  ```bash
  venv\Scripts\activate
  ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generate the dataset

Run the dataset generator to create `dataset/sentiment.csv`:

```bash
python generate_dataset.py
```

### Train the model

Train a BERT sentiment classification model and save it to `./model`:

```bash
python train.py
```

### Run the Streamlit app

Launch the web interface for predictions:

```bash
streamlit run app/app.py
```

Then open the local URL shown by Streamlit in your browser.

### Quick test

Run the simple pipeline test:

```bash
python bert_test.py
```

## Testing Examples

After training the model, test predictions with these example inputs:

### Positive Sentiment Examples

```
"I love this product, it's amazing!"
"Excellent quality and great customer service!"
"Best purchase ever, highly recommended!"
"The product exceeded all my expectations!"
```

### Neutral Sentiment Examples

```
"The product arrived yesterday"
"It works as expected"
"The design is standard"
"Average performance, nothing special"
```

### Negative Sentiment Examples

```
"Worst service ever, very disappointed"
"Poor quality, not worth the money"
"The product stopped working after a week"
"Terrible experience, do not recommend"
```

### Using the Streamlit App

1. Run the app:

   ```bash
   streamlit run app/app.py
   ```

2. Paste any text into the text area and click "Predict" to see the sentiment classification.

### Programmatic Testing

You can also test predictions directly in Python:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("./model")
model = BertForSequenceClassification.from_pretrained("./model")

text = "I love this product!"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1).item()

labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
print(f"Text: {text}")
print(f"Sentiment: {labels[prediction]}")
```

## Notes

- The training dataset is synthetic and intended for demo purposes.
- `train.py` uses `bert-base-uncased` with `num_labels=3` and maps labels as:
  - `0` = NEGATIVE
  - `1` = NEUTRAL
  - `2` = POSITIVE
- To use a custom dataset, replace `dataset/sentiment.csv` and adjust preprocessing or label mapping as needed.

## Dependencies

Key dependencies are:

- `torch`
- `transformers`
- `pandas`
- `scikit-learn`
- `streamlit`
- `accelerate`
