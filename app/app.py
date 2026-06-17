import streamlit as st
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

labels = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}

st.title("Sentiment Analysis Using BERT")

text = st.text_area("Enter Text")

if st.button("Predict"):
    if text.strip():

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        st.success(f"Prediction: {labels[prediction]}")