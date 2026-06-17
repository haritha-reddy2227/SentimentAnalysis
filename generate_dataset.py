import pandas as pd

positive = [
    "I love this product",
    "Amazing experience",
    "Excellent quality",
    "Highly recommended",
    "Very satisfied",
    "Fantastic service",
    "Best purchase ever",
    "Wonderful product",
    "Great value for money",
    "Outstanding performance"
]

neutral = [
    "It is okay",
    "Average performance",
    "Nothing special",
    "The product arrived yesterday",
    "It works as expected",
    "Neither good nor bad",
    "The package was delivered today",
    "The design is standard",
    "Performance is acceptable",
    "It meets basic requirements"
]

negative = [
    "Worst service ever",
    "Bad product",
    "Very disappointed",
    "Poor quality",
    "Terrible experience",
    "Waste of money",
    "Not recommended",
    "The product stopped working",
    "Customer support is awful",
    "I regret buying this"
]

data = []

for i in range(300):
    data.append([positive[i % len(positive)] + f" #{i}", 2])

for i in range(300):
    data.append([neutral[i % len(neutral)] + f" #{i}", 1])

for i in range(300):
    data.append([negative[i % len(negative)] + f" #{i}", 0])

df = pd.DataFrame(data, columns=["text", "label"])

df.to_csv("dataset/sentiment.csv", index=False)

print("Dataset created successfully!")
print("Total rows:", len(df))