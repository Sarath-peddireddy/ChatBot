import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load your dataset and model
df = pd.read_csv('ipc_sections.csv')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the saved TF-IDF model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess the text
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''

# Fill NaN values and preprocess
df.fillna('', inplace=True)
df['Description_clean'] = df['Description'].apply(preprocess_text)
df['Offense_clean'] = df['Offense'].apply(preprocess_text)
df['combined_text'] = df['Description_clean'] + ' ' + df['Offense_clean']

# Function to find the closest matching section based on user query
def get_closest_section(query):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, vectorizer.transform(df['combined_text'])).flatten()
    closest_index = cosine_similarities.argmax()
    return df.iloc[closest_index][['Section', 'Offense', 'Punishment']]

# Streamlit app layout
st.title("Law Assistant Chatbot")
user_query = st.text_input("Ask your legal question:")

if st.button("Get Answer"):
    if user_query:
        result = get_closest_section(user_query)
        st.write(f"**Section:** {result['Section']}")
        st.write(f"**Offense:** {result['Offense']}")
        st.write(f"**Punishment:** {result['Punishment']}")
    else:
        st.write("Please enter a question.")
