import streamlit as st
import speech_recognition as sr
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras import backend as K


@st.cache(allow_output_mutation=True)
def load_models():
    model_lr = load_model('model_l.hd/')
    model_sr = load_model('model_s.hd/')
    return model_lr, model_sr

model_lr, model_sr = load_models()

class WordVectorTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, model="en_trf_distilbertbaseuncased_lg"):    #put bert embeddings here
        self.model = model
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        nlp = spacy.load(self.model)
        return np.concatenate([nlp(doc).vector.reshape(1,-1) for doc in X])

bertvect = WordVectorTransformer()
analyzer = SentimentIntensityAnalyzer()

@st.cache
def combine_predict(sentence):
    sentence = [[sentence]]

    preds = np.array([])

    s_input_model_l = np.array([bertvect.transform(i) for i in sentence]).reshape(-1,768,1)
    s_input_model_s = pd.DataFrame([analyzer.polarity_scores(i) for i in sentence])

    preds_l = model_lr.predict(s_input_model_l)
    preds_s = model_sr.predict(s_input_model_s)
    
    for i in preds_l:
        for j in preds_s:
            if abs(i - j) > 0.5:
                preds = np.append(preds,j)
            else:
                preds = np.append(preds,(0.7*i + 0.3*j))
    
            
    return preds[0]


    