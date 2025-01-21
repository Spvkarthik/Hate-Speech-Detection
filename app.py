import streamlit as st
import pickle
# Load pre-trained model
with open('final_model1.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the fitted CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

st.title('Hate Speech Detection')

st.write("""
This app detects the hate speech in a text.
""")

text=st.text_area("Enter the text:")
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import html
stop_words=set(stopwords.words('english'))
wordnet=WordNetLemmatizer()
def data_cleaning(text):
    text=re.sub(r'@\w+', ' ', text)
    text= re.sub(r'\bRT\b', ' ', text)
    text=html.unescape(text)
    text= re.sub(r'http[s]?://\S+', '', text)
    text=re.sub('[^a-zA-Z]', ' ', text)
    text=text.lower()
    text=text.split()
    text=[wordnet.lemmatize(word) for word in text if not word in stop_words]
    text=' '.join(text)
    return text

if st.button("Detect"):
    cleaned_text = data_cleaning(text)
    transformed_text = cv.transform([cleaned_text]).toarray()  # Transform expects a list of texts
    # Predict using the model
    prediction = model.predict(transformed_text)
    # Since the prediction is an array like array(['hate speech'], dtype=object)
    st.write(f"Detection: {prediction[0]}")

st.write("""Note: The model is trained on raw data, so the predictions are made directly on the input values.""")