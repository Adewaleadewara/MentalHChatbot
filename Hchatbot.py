import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
lemmatizer = nltk.stem.WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.markdown("<h1 style = 'color: #201658; text-align: center; font-family: TaHoma'>MENTAL HEALTH PERSONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FAFAFA; text-align: center; font-family: Copperplate'>Built By: ADEWALE JOLAYEMI</h4>",unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

df = pd.read_csv('Mental_Health_FAQ.csv')
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


df['tokenized Questions'] = df['Questions'].apply(preprocess_text)
x = df['tokenized Questions'].to_list()
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(x)

robot_image, chat_response = st.columns(2)
with robot_image:
    robot_image.image('pngwing.com (15).png', caption= 'Hello there, ask me your questions')

with chat_response:
    user_word = chat_response.text_input('Hey,ask your questions')
    def get_response(user_input):
        user_input_processed = preprocess_text(user_input) 

        user_input_vector = tfidf_vectorizer.transform([user_input_processed])
        similarity_scores = cosine_similarity(user_input_vector, corpus) 

        most_similar_index = similarity_scores.argmax() 

        return df['Answers'].iloc[most_similar_index]
    greetings = ["Hey There.... I am a creation of Ade Coddy.... How can I help",
                "Hi, there.... How can I help",
                "Good Day .... How can I help", 
                "Hello There... How can I be useful to you today"]
    exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
    farewell = ['Thanks....see you soon', 'Bye, See you soon', 'Bye... See you later', 'Bye... come back soon', "It's a pleasure having you here"]

    random_farewell = random.choice(farewell) 
    random_greetings = random.choice(greetings) 
    # while True:
    #     user_input = input("You: ")
    if user_word.lower() in exits:
        chat_response.write(f"\nChatbot: {random_farewell}!")

    elif user_word.lower() in ['hi', 'hello', 'hey', 'hi there']:
        chat_response.write(f"\nChatbot: {random_greetings}!")

    elif user_word == '':
        chat_response.write('')
        
    else:   
        response = get_response(user_word)
        chat_response.write(f"\nChatbot: {response}")
st.header('Project Background Information',divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ.")
