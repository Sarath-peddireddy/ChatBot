import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('ipc_sections.csv')
    return df

# Load vectorizer with caching
@st.cache_resource
def load_vectorizer():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return vectorizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text function to clean and lemmatize
@st.cache_data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''

# Cache vectorized texts
@st.cache_data
def vectorize_texts(df, _vectorizer):
    df.fillna('', inplace=True)
    df['Description_clean'] = df['Description'].apply(preprocess_text)
    df['Offense_clean'] = df['Offense'].apply(preprocess_text)
    df['combined_text'] = df['Description_clean'] + ' ' + df['Offense_clean']
    text_vectors = _vectorizer.transform(df['combined_text'])
    return text_vectors

# Handle greetings and friendly inquiries
def handle_greetings(query):
    query = query.lower()
    if any(greeting in query for greeting in ['hi', 'hello', 'how are you', 'hey']):
        return "Hi! I'm Sarath, your law assistant chatbot. I'm here to help you with your legal queries regarding the Indian Penal Code (IPC). What would you like to know?"
    return None

# Handle general legal queries
def handle_general_query(query):
    query = query.lower()
    if 'crime' in query:
        return "Crime refers to an unlawful act punishable by a state or other authority."
    elif 'ipc' in query or 'indian penal code' in query:
        return "The Indian Penal Code (IPC) is the official criminal code of India. It covers all substantive aspects of criminal law."
    elif 'bail' in query:
        return "Bail is the temporary release of a person awaiting trial, usually on condition of a sum of money being lodged to guarantee their appearance in court."
    elif 'punishment' in query:
        return "Punishment is the infliction of a penalty as retribution for an offense, typically defined in legal statutes."
    elif 'offense' in query or 'criminal offense' in query:
        return "An offense is a breach of a law or rule; an illegal act that may be prosecuted."
    elif 'civil law' in query:
        return "Civil law deals with the rights and duties of individuals and organizations, and addresses disputes between them."
    elif 'criminal law' in query:
        return "Criminal law is the body of law that relates to crime, and it prescribes punishments for those who violate the laws."
    return None

# Get the closest matching law section
def get_closest_section(query, _vectorizer, df, text_vectors):
    # Check for greetings first
    greeting_response = handle_greetings(query)
    if greeting_response:
        return greeting_response
    
    # Check if it's a general query
    general_response = handle_general_query(query)
    if general_response:
        return general_response
    
    # Preprocess the query and calculate similarity
    query = preprocess_text(query)
    query_vec = _vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, text_vectors).flatten()
    
    # If no good match is found, return a message
    if cosine_similarities.max() < 0.05:  # Lowered threshold slightly
        return "No relevant section found for your query."
    
    closest_index = cosine_similarities.argmax()
    result = df.iloc[closest_index][['Section', 'Offense', 'Punishment']]
    return f"**Section:** {result['Section']}\n**Offense:** {result['Offense']}\n**Punishment:** {result['Punishment']}"

# Load dataset and vectorizer
df = load_data()
vectorizer = load_vectorizer()

# Vectorize the text data
text_vectors = vectorize_texts(df, vectorizer)

# Streamlit app layout
st.title("Law Assistant Chatbot")
user_query = st.text_input("Ask your legal question:")

if st.button("Get Answer"):
    if user_query:
        result = get_closest_section(user_query, vectorizer, df, text_vectors)
        st.write(result)
    else:
        st.write("Please enter a question.")
