#Text Data Preprocessing Lib
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    input_word_token_1 = 
    input_word_token_2 = 
    input_word_token_2 = 

    bag=[]
    bag_of_words = []
   
    # Input data encoding 
    
  
    return np.array(bag)

def bot_class_prediction(user_input):

    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  
   predicted_class = 

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
        

print("Hi I am Stella, How Can I help you?")

while True:
    user_input = input("Type your message here:")
    print("User Input: ", user_input)

    response = bot_response(user_input)
    print("Bot Response: ", response)