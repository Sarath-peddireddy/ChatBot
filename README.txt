# Law Assistant Chatbot

## Overview
The Law Assistant Chatbot is a Streamlit application designed to provide users with information about various Indian Penal Code (IPC) sections. It utilizes Natural Language Processing (NLP) techniques to understand user queries and retrieve relevant legal information, including offenses and their corresponding punishments.

## Features
- **Natural Language Understanding**: The chatbot can comprehend user queries and provide relevant responses based on the IPC dataset.
- **FAQs**: Users can receive predefined answers to common legal questions.
- **Interactive Interface**: The Streamlit interface allows users to easily input questions and view responses in real-time.

## Technologies Used
- **Streamlit**: For creating the web application interface.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning tasks, including TF-IDF vectorization and cosine similarity calculations.
- **NLTK**: For natural language processing tasks like tokenization and lemmatization.
- **Joblib**: For saving and loading the TF-IDF vectorizer model.

## Installation

### Prerequisites
Make sure you have Python 3.x installed on your machine.

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
