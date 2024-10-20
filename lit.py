import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv('ipc_sections.csv')
    return df

# Cache model loading
@st.cache_resource
def load_model():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load saved TF-IDF model
    return vectorizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text function with caching for efficiency
@st.cache_data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = text.split()  # Basic tokenization
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''

# Cache the text vectorization step
@st.cache_data
def vectorize_texts(df, _vectorizer):
    df.fillna('', inplace=True)
    df['Description_clean'] = df['Description'].apply(preprocess_text)
    df['Offense_clean'] = df['Offense'].apply(preprocess_text)
    df['combined_text'] = df['Description_clean'] + ' ' + df['Offense_clean']
    text_vectors = _vectorizer.transform(df['combined_text'])
    return text_vectors

# Function to find the closest matching section based on user query
def get_closest_section(query, vectorizer, df, text_vectors):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, text_vectors).flatten()
    closest_index = cosine_similarities.argmax()
    return df.iloc[closest_index][['Section', 'Offense', 'Punishment']]

# Load data and model once using caching
df = load_data()
vectorizer = load_model()

# Vectorize the text data
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
