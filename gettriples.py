import nltk
import streamlit as st
import pandas as pd
import en_core_web_sm
import numpy as np
from collections import Counter
#from nltk.corpus import stopwords
#stopwords.words('english')

# Extract relations
def extract_relations(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()
    triples = []
    for ent in doc.ents:
        preps = [prep for prep in ent.root.head.children if prep.dep_ == "prep"]
        for prep in preps:
            for child in prep.children:
                triples.append((ent.text, "{} {}".format(ent.root.head, prep), child.text))
    return triples


# read input
def get_features(filename):
    f = open(filename,'r')
    lines = f.readlines();
    f.close()
    nlp = en_core_web_sm.load()
    head = ''
    text=''
    for line in lines:
        text += line
    # sentence boundary disambiguation
    doc = nlp(text)
    sents = list(doc.sents)
    print('There are ', len(sents), 'sentences ')
    
    # most common nouns
    tokens = nltk.tokenize.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    nouns_and_verbs = [token[0] for token in tagged_tokens if token[1] in ['NNP']]
    frequency = nltk.FreqDist(nouns_and_verbs)
    
    t = extract_relations(doc);
    return t
    
    
heads=[]
common_words=[]

#filename = 'data/t69.txt'
filename='/Users/kiran/Documents/my_py/my_tutorial/word2vec_input2.txt'
t = get_features(filename)



