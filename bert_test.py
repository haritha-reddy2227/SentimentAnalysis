from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("It is okay")

print(result)