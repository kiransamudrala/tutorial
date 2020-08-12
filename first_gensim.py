import requests
import en_core_web_sm
from gensim import models
from bs4 import BeautifulSoup
from newspaper import Article 
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import random


# Getting individual cities url
def geturls(motherurl):
    re = requests.get(motherurl)
    soup = BeautifulSoup(re.text, "html.parser")
    potential_urls = soup.find_all('div',class_='field-item even')
    focus_urls =[]
    for f in potential_urls:
        temp = f.find('h3')
        if temp != None:
            focus_urls.append(temp)
    urls=[]
    for p in focus_urls:
        temp = p.find('a')
        if temp == None:
            continue
        else:
            urls.append('https://www.dataguidance.com'+temp['href'])
    # print(urls)
    return urls

# get article urls
print('STEP 1: FETCHING URLS')
urls = []
pages=25
for x in range(pages):
    x=x+1
    motherurl = 'https://www.dataguidance.com/search/news?page='+str(x)
    urls = urls + geturls(motherurl)
    
text = ''
print('STEP 2: EXTRACTING CONTENT FROM EACH URL')
# consolidate articles into text
for url in urls:
    art = Article(url, language="en") 
    art.download() 
    art.parse() 
    nl = art.nlp() 
    text += art.text 
    
print('STEP 3: TOKENIZING THE TEXT TO WORDS AS LIST OF LISTS')
# convert the text into list of list of words
nlp = en_core_web_sm.load()
tokens = nlp(text)
sents = []
for sent in tokens.sents:
    sents.append(sent.string.strip())
sentwords = []
for words in sents:
    for word in words:
        sentwords.append(words.split(' '))
    
# train the word2vec model
print('STEP 4: ANALYZING THE CONTENT USING WORD2VEC')
model = models.Word2Vec(sentwords, min_count=1)

# visualize word embeddings
print('STEP 5: APPLYING PCA TO VIEW THE MODEL')
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)


words = list(model.wv.vocab)

def plot(n,words,result):
    seed = random.randint(0,len(words)-n-1)
    r = result[seed:seed+n-1,:]
    w = words[seed:seed+n-1]
    
    plt.scatter(r[:, 0], r[:, 1])
    for i, word in enumerate(w):
        plt.annotate(word, xy=(r[i, 0], r[i, 1]))

#plot n random points in words
plot(10,words,result)


print('STEP 7: STORE THE INPUT AND OUTPUT')
import numpy as np
import pandas as pd
df = pd.DataFrame(np.column_stack([words,result]),columns=['words','x','y'])
df = pd.DataFrame(np.column_stack([words,result]),columns=['words','x','y'])
df.to_csv('word2vec_output.csv',header=True)

f = open('word2vec_input.txt','w')
f.write(text)
f.close()



