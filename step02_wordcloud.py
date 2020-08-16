import re
import random
import numpy as np
import pandas as pd
import en_core_web_sm
from gensim import models
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer


# compute tfidf
def get_tfidf(data):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(data)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    #df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
    #print(df_idf.sort_values(by=['idf_weights'],ascending=True))
    
    count_vector=cv.transform(data)
    tf_idf_vector=tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names()
    fm = pd.DataFrame(columns=['tfidf'])
    for t in tf_idf_vector:
        df = pd.DataFrame(t.T.todense(), index=feature_names, columns=["tfidf"])
        fm = pd.concat([fm,df['tfidf']],axis=1)
    cols=[]
    for x in range(len(data)+1):
        cols.append( 'col'+str(x))
    fm.columns = cols
    fm = fm.drop(fm.columns[0],axis=1) 
    return fm


print('\n STEP 1: FETCH DATA FROM FILES AND ORGANIZE THE INPUTS')
dfinput = pd.read_csv('data/ot.txt')
lines = dfinput['text'].values
text = ''
for line in lines:
    text += line
# get all topics
topics = []
for t in dfinput['topics']:
    topics.append(t)
# convert a list of lists into a simple list (we lose order)
topiclist = []
for t in topics:
    for tsub in t:
        topiclist.append(tsub)
# remove duplicates from topiclist
topiclist = list(set(topiclist))



print('\n STEP 2: TOKENIZING THE TEXT TO WORDS AS LIST OF LISTS')
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
print('\nSTEP 3: CREATE WORD CLOUD: ANALYZING THE CONTENT USING WORD2VEC')
model = models.Word2Vec(sentwords, min_count=1)
X = model[model.wv.vocab]
words = list(model.wv.vocab)

# build word and topic clouds into dataframes
dfwc = pd.DataFrame(X,index=words)
dftc = dfwc[dfwc.index.isin(topiclist)]


# remove stopwords from each article and get tfidf feature matrix
print('\nSTEP 4: GET MOST IMPORTANT WORDS OF EACH ARTICLE')
stopsdf = pd.read_csv('data/stops.txt')
stopsdf = stopsdf[['0']]
stopsdf.columns=['stopwords']
stops = stopsdf['stopwords'].values.tolist()
text2 = []
for t in lines:
    t=t.lower()
    t=re.sub('[^a-zA-Z]', ' ', t )
    t=re.sub(r'\s+', ' ', t)
    for s in stops:
        t=t.replace(' '+s+' ',' ')
        t=t.replace('.'+s+' ',' ')
    text2.append(t)

fm = get_tfidf(text2)


# visualize word embeddings
print('\nSTEP 4: APPLYING PCA TO VIEW THE MODEL')
pca = PCA(n_components=min(X.shape))
result = pca.fit_transform(X)


print('\nSTEP 5: PLOT WORD CLOUD')
def plot(n,words,result):
    seed = random.randint(0,len(words)-n-1)
    r = result[seed:seed+n-1,:]
    w = words[seed:seed+n-1]
    plt.scatter(r[:, 0], r[:, 1])
    for i, word in enumerate(w):
        plt.annotate(word, xy=(r[i, 0], r[i, 1]))

#plot n random points in words
plot(20,words,result)
plt.show()

# screeplot
ev = pca.explained_variance_
evp = ev/sum(ev)

# plot scree cumulative
cumulative = np.flipud(np.cumsum(ev))
plt.plot(range(len(ev[1:])),ev[1:], c='blue')
plt.title('Eigenvalue Plot')
plt.show()

print('STEP 7: STORE THE INPUT AND OUTPUT')
df = pd.DataFrame(np.column_stack([words,result]),columns=['words','x','y'])
df.to_csv('data/wordcloud.csv',header=True)







