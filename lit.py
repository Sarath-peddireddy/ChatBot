import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources quietly (only if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Function to load data with caching for deployment
@st.cache_data
def load_data():
    df = pd.read_csv('ipc_sections.csv')
    return df

# Function to load the vectorizer with caching for deployment
@st.cache_resource
def load_vectorizer():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return vectorizer

# Initialize the lemmatizer (does not need to be cached)
lemmatizer = WordNetLemmatizer()

# Preprocess the text function to clean and lemmatize
@st.cache_data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''

# Cache text vectorization to reduce repetitive computation
@st.cache_data
def vectorize_texts(df, _vectorizer):
    df.fillna('', inplace=True)
    df['Description_clean'] = df['Description'].apply(preprocess_text)
    df['Offense_clean'] = df['Offense'].apply(preprocess_text)
    df['combined_text'] = df['Description_clean'] + ' ' + df['Offense_clean']
    text_vectors = _vectorizer.transform(df['combined_text'])
    return text_vectors

# Function to get the closest matching law section
def get_closest_section(query, _vectorizer, df, text_vectors):
    query = preprocess_text(query)
    query_vec = _vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, text_vectors).flatten()
    closest_index = cosine_similarities.argmax()
    return df.iloc[closest_index][['Section', 'Offense', 'Punishment']]

# Load the dataset and vectorizer using caching
df = load_data()
vectorizer = load_vectorizer()

# Vectorize text data only once using caching
text_vectors = vectorize_texts(df, vectorizer)

# Streamlit app layout
st.title("Law Assistant Chatbot")
user_query = st.text_input("Ask your legal question:")

if st.button("Get Answer"):
    if user_query:
        result = get_closest_section(user_query, vectorizer, df, text_vectors)
        st.write(f"**Section:** {result['Section']}")
        st.write(f"**Offense:** {result['Offense']}")
        st.write(f"**Punishment:** {result['Punishment']}")
    else:
        st.write("Please enter a question.")
