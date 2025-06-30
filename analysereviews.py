import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

#csv data file
df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')

#first few rows
print(df.head())

#first review
review = df['reviews.text'][0]
print("Sample review:", review)

#sentiment polarity
blob = TextBlob(review)
print("Sentiment polarity:", blob.sentiment.polarity)
