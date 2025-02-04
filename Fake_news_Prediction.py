# Fake news Prediction
# 1 for fake
# 0 for real
# depencies
# !pip install --upgrade scikit-learn
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk


nltk.download("stopwords")
import streamlit as st


# data preprocessing
news_dataset = pd.read_csv("C:/Users/DELL/Downloads/train.csv (2)/train.csv")

# print(news_dataset.shape)
# print(news_dataset.columns)
# print(news_dataset.head())
# news_dataset.isnull().sum()

news_dataset = news_dataset.fillna("")
# cobine news author and tittle based on it news predict
news_dataset["content"] = news_dataset["author"] + " " + news_dataset["title"]
# print(news_dataset['content'])
# seprate label coulmn from other datas
x = news_dataset.drop(
    columns="label", axis=1
)  # axis 1 means column remove , it is 0 for rows
y = news_dataset["label"]


# Stemmming
port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub(
        "[^a-zA-Z]", " ", content
    )  # re search for alphabates when find any digits or sign give " "to it
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()  # convert into a list
    # example:  text = "Python is powerful"
    # words = text.split()
    # print(words)   ['Python', 'is', 'powerful']
    stemmed_content = [
        port_stem.stem(word)
        for word in stemmed_content
        if not word in stopwords.words("english")
    ]
    # port_stem for take out keywords from words and words which are not is stop words
    stemmed_content = " ".join(stemmed_content)
    # list to again in one sentance
    return stemmed_content


# apply function to each rowof column
news_dataset["content"] = news_dataset["content"].apply(stemming)


# seprate label and content data
x = news_dataset["content"].values
y = news_dataset["label"].values


# convert textual data into a numeric data
vectorizer = TfidfVectorizer()  # tf= term freq and id =inverse document freq
# a measure of how important a word is in a document relative to all other documents.
vectorizer.fit(x)  # convert respected feature and give tfid valies
x = vectorizer.transform(
    x
)  # This transforms the text data (x) into a sparse matrix where each term is represented by a numerical value based on its TF-IDF score in each document.


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)
# train data is 80% where test data is 20%
# stratify =y(label)the split preserves the proportion of labels (or classes) in the dataset.
# It ensures that if you run the code multiple times, you'll get the same split each time.


# logical regression
model = LogisticRegression()
model.fit(x_train, y_train)

# #accuracy score
# x_train_prediction=model.predict(x_train)
# training_data_accuracy=accuracy_score(x_train_prediction,y_train)
# print('accuracy score of training data :',training_data_accuracy)

# #accuracy score in test data
# x_test_prediction=model.predict(x_test)
# test_data_accuracy=accuracy_score(x_test_prediction,y_test)
# print('accuracy score of test data :',test_data_accuracy)

# x_new=x_test[1]
# prediction=model.predict(x_new)
# print(prediction)
# if (prediction==0):
#   print('the news is real')
# else:
#   print('the news is fake')

# website
# Streamlit is an open-source Python library used for building interactive web applications for machine learning and data science with minimal effort
# st.title('Fake news prediction ')
# st.markdown("<h1 style='color:black;'>Fake news prediction 📰</h1>", unsafe_allow_html=True)
# input= st.text_input("<p style='color:black';>Entre news article</p>",unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color:#A9A9A9;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# def prediction(input):
#     input_data=vectorizer.transform([input])
#     prediction1=model.predict(input_data)
#     return prediction1
# if input:
#     pred=prediction(input)
#     if pred == 1:
#       st.markdown("<h3 style='color: red;'>🚨 News is FAKE!</h3>", unsafe_allow_html=True)
#     else:
#         st.markdown("<h3 style='color: green;'>✅ News is REAL!</h3>", unsafe_allow_html=True)


# Streamlit UI
st.markdown(
    "<h1 style='color:black;'>Fake News Prediction 📰</h1>", unsafe_allow_html=True
)

# Text input field
input_text = st.text_input("Enter news article:")


# Function for prediction
def predict_news(text):
    input_data = vectorizer.transform([text])  # Transform input text
    prediction = model.predict(input_data)  # Predict
    return prediction


# Display result when input is given
if input_text:
    pred = predict_news(input_text)
    if pred == 1:
        st.markdown(
            "<h3 style='color: red;'>🚨 News is FAKE!</h3>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='color: green;'>✅ News is REAL!</h3>", unsafe_allow_html=True
        )

# Background color styling
st.markdown(
    """
    <style>
    .stApp {
        background-color:#A9A9A9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
