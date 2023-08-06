import pickle
import streamlit as st
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import sklearn
nltk.download('stopwords')
nltk.download('punkt')
model = pickle.load(open('model_mnb_rev,pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
stemmer=PorterStemmer()
def text_preprocess(text):

    # Lower-casing the words and splitting into a list of words
    # words = word_tokenize(text.lower())
    words=re.findall(r'\w+', text.lower())
    sw = stopwords.words('english')
    punctuation = string.punctuation
    processed_words = []
    for word in words:
      if word not in stopwords.words('english'):
          word=word.translate(str.maketrans('','',punctuation))
          word=stemmer.stem(word) # Stemming
          processed_words.append(word)
    processed_text = ' '.join(processed_words)
    return processed_text


st.title('IMDB Movie Sentiment Analysis')
text=st.text_area('Enter the review please')


def pipeline(text):
    if text:
        transformed_sms = text_preprocess(text)
        # 2. vectorize
        vector_input = vectorizer.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.subheader("Positive Sentiment")
        else:
            st.subheader("Negative Sentiment")
    else:
        st.subheader("Enter some text please")

if st.button('Predict'):
    pipeline(text)
