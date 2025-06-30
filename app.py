import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk

#comment it after running once per setup
nltk.download('punkt')

#csv files 
df = pd.read_csv('1429_1.csv', low_memory=False)
df = df[['reviews.text']].dropna().head(200)
df = df.rename(columns={'reviews.text': 'text'})

#sentiment polarity
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

#sentiment label
def label_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(label_sentiment)

#pros and cons extractor
def extract_pros_cons(review):
    blob = TextBlob(review)
    pros = []
    cons = []
    for sentence in blob.sentences:
        polarity = sentence.sentiment.polarity
        subjectivity = sentence.sentiment.subjectivity
        if polarity > 0.1 and subjectivity > 0.3:
            pros.append(str(sentence))
        elif polarity < -0.1 and subjectivity > 0.3:
            cons.append(str(sentence))
    return pd.Series([pros, cons])



df[['pros', 'cons']] = df['text'].apply(extract_pros_cons)

#5 examples, can be changed later
for i in range(5):
    print(f"\nReview {i+1}:")
    print("Text:", df.iloc[i]['text'])
    print("Pros:", df.iloc[i]['pros'])
    print("Cons:", df.iloc[i]['cons'])

#sentiment distribution graph
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Review Count")
plt.tight_layout()
plt.show()

import nltk
from nltk.corpus import stopwords
from collections import Counter
import string

#comment it after running once
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def tokenize_and_clean(text_list):
    all_words = []
    for text in text_list:
        tokens = nltk.word_tokenize(" ".join(text))
        words = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        all_words.extend(words)
    return all_words

pros_words = tokenize_and_clean(df['pros'].dropna())
cons_words = tokenize_and_clean(df['cons'].dropna())

top_pros = Counter(pros_words).most_common(10)
top_cons = Counter(cons_words).most_common(10)

#bar graph
def plot_top_words(word_counts, title, color):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(8, 5))
    plt.barh(words, counts, color=color)
    plt.xlabel("Frequency")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

plot_top_words(top_pros, "Top 10 Pros (Most Frequent Words)", "green")
plot_top_words(top_cons, "Top 10 Cons (Most Frequent Words)", "red")

def is_fake_review(text):
    blob = TextBlob(text)
    length = len(text.split())
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    generic_phrases = ['very nice', 'great product', 'good', 'awesome', 'excellent', 'nice', 'love it']
    generic_count = sum(phrase in text.lower() for phrase in generic_phrases)

#1 too short
    if length < 8:
        return True

#2 contains two or more than two generic words
    if generic_count >= 2 and length < 30:
        return True

#3 polarity and subjectivity are very high and very short
    if polarity > 0.8 and subjectivity > 0.8 and length < 15:
        return True

#4 all generic and less than 20 words no real content
    if generic_count >= 1 and length < 20:
        return True

    return False


df['is_fake'] = df['text'].apply(is_fake_review)

#fake real reviews graph
df['is_fake'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.xticks([0, 1], ['Real', 'Fake'], rotation=0)
plt.title("Fake vs Real Reviews")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

#print sus fake reviews
#currently set to 5, can be increased to many
print("\nSuspected Fake Reviews:\n")
for i, review in enumerate(df[df['is_fake'] == True]['text'].head(5), start=1):
    print(f"{i}. {review}\n")
