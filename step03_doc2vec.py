import gensim
import random
import smart_open
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.decomposition import PCA

# set variables
trainf = 'data/corpus.txt'
high_dim=100

# read from the dfs
df_train = pd.read_csv('data/train_ot.txt')
df_test = pd.read_csv('data/test_ot.txt')
df = pd.concat([df_train,df_test])

def df_to_csv(df,outfile):
    lines = df['text'].values
    clean_lines=[]
    for line in lines:
        if len(line)>=2 and line != '\n':
            clean_lines.append(line)
    fn = open(outfile,'w')
    for x in range(len(clean_lines)):
        fn.write(clean_lines[x])
        fn.write('\n')
    fn.close()

df_to_csv(df,trainf)


# function to create a tagged document format from the test and train files
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(trainf))

# apply document vector model
model = gensim.models.doc2vec.Doc2Vec(vector_size=high_dim, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
np.save('docvec.npy',model.docvecs.vectors_docs)
docvec_highdim = model.docvecs.vectors_docs

# apply PCA and visualize
pca = PCA(n_components=min(docvec_highdim.shape))
pcs = pca.fit_transform(docvec_highdim)
dfout = pd.DataFrame(pcs[:,0:5])
dfout.to_csv('data/doc_vectors.txt')

# get data labels
labels=df.title.values
topics=df.topics.values

##############################################


dfout1 = pd.read_csv('data/train_lowdim.csv')
df_train = pd.read_csv('data/train_clean.csv')
st.write('Training Data using Word2Vec')
        
#n = 500
#seed = random.randint(0,len(dfout)+n-1)
#df = dfout1.iloc[seed:seed+n-1,0:2]
df = dfout1.iloc[:,1:3]
df['words']=df_train['cleaner']+ " |||  " + df_train['Labels']
df['Label']=df_train['Labels']
df.columns = ['x','y','words','Label']
source = df
points = alt.Chart(source).mark_point().encode(
    x='x',
    y='y',
    tooltip='words',
    color='Label'
).properties(
    width=800,
    height=800
    
    )
points 



dfout2 = pd.read_csv('data/predict_lowdim.csv')
df_predicted = pd.read_csv('data/predicted.csv')
st.write('Predicted Data using Word2Vec + Logistic Regression')
        
#n = 500
#seed = random.randint(0,len(dfout)+n-1)
#df2 = dfout.iloc[seed:seed+n-1,0:2]
df2 = dfout2.iloc[:,1:3]
df2['words']=df_predicted['PII']+ " |||  " + df_predicted['Label']
df2['Label']=df_predicted['Label']
df2.columns = ['x','y','words','Label']
source = df2
points = alt.Chart(source).mark_point().encode(
    x='x',
    y='y',
    tooltip='words',
    color='Label'
).properties(
    width=800,
    height=800
    
    )
points 







#############################################

dfdc = pd.DataFrame(np.column_stack([pcs[:,0],pcs[:,1],labels,topics]),columns=['x','y','title','topics'])

st.markdown("""<p style="font-size:30px">Assigning Topics to OneTrust News Articles</p>""", unsafe_allow_html=True)
st.write('Assigning topics to news articles is a typical example of multi-label classification \
         where our objective is to train the machine to automatically understand and tag the articles\
        with multiple topics that the article is talking about.\
         Owing to the advancements in natural language processing field in the past decades, \
        and particularly past couple of years, there are multiple ways to achieve this. \
        For example: ')
st.markdown(
    """
    <ul>
    <li>How to model your data: 
        <ul>
            <li>Bag-of-words</li> 
            <li>TF-IDF</li> 
            <li><font color='green'>paragraph vector</font></li> 
            <li><font color='green'>word-to-vec</font></li> 
            <li> BERT</li>
        </ul>
    </li>
    <li>What Transformation type to choose for solution:
        <ul>
            <li>Binary Relevance</li> 
            <li>Classifier Chains</li> 
            <li> Powerset Labels </li>
        </ul>
    <li>What core model to choose: 
        <ul>
            <li>Logistic Regression</li> 
            <li>Supervised-Neural networks</li> 
            <li>unsupervised dimensionality reduction or Knn </li>
        </ul>
    </ul>
    """, unsafe_allow_html=True)
st.write('Below is the high-level summary of my proposed approach (green bullets are the ones I finished implementing in python):')
st.markdown(
    """
    <li> <font color='green'> Input: Scraped 240 news articles from dataguidance.com </font></li>
    <li><font color='green'> Preprocessing: Extracted the word-cloud from the entire corpus </font></li>
    <li><font color='green'> Target Variables: Converted the topics of full corpus into numeric vectors using word-cloud mapping </font></li>
    <li><font color='green'> Features: Converted each document into a vector using paragraph vector model or doc-to-vec model </font></li>
    <li><font color='green'> Split the data into training and testing data</li>
    <li>Feed the mappings of Features and Target Variables to a neural network model</li>
    <li>Train/Test/Predict from the model </li>
    <li>Measure the performance of the model using metrics like Recall, Precision, F1</li>
    """, unsafe_allow_html=True)

st.write('Document cloud or feature space: In the below chart, each dot is a news article.\
         Hover/Tap over to see what topic they are tagged by. This document cloud is a \
        reduced representation obtained by applying PCA - the unsupervised linear \
            dimensionality reduction technique. ')

source = dfdc

points = alt.Chart(source).mark_point().encode(
    x='x',
    y='y',
    tooltip='topics'
).properties(
    width=800,
    height=800
    )
points 



dfwc=pd.read_csv('data/wordcloud.csv')

st.write('Word cloud or target variable space: The below chart is a reduced dimensional representation of the \
         word cloud produced by analyzing 240 articles using the \
        word2vec model. \
        The proximity of the words on the chart is a function of \
        their meaning or their co-occurrence in the documents. \
        Since the first two components were only able to capture 14% of variability in the data \
        we can still see some words which are actually not next to eachother \
        seemingly very close to eachother in the chart.')

st.write('Hover over the points to see the words as a tooltip')

source = dfwc
points_all = alt.Chart(source).mark_point().encode(
    x='x',
    y='y',
    tooltip='words'
).properties(
    width=800,
    height=800
    
    )

points_all

st.write('Zoomed in View: The below chart is a subset of the above chart. Each time you refresh the page \
                 the chart shows a randomly picked 50 words.\
                 The proximity of the words on the chart is a function of \
                     their meaning or their co-occurrence in the documents.')
        
n = 50
seed = random.randint(0,len(dfwc)+n-1)
df = dfwc.iloc[seed:seed+n-1,:]
#chart = alt.Chart(df).mark_circle().encode(x='x', y='y',tooltip='words',text='words')

source = df

points = alt.Chart(source).mark_point().encode(
    x='x',
    y='y',
    tooltip='words'
).properties(
    width=800,
    height=800
    
    )

text = points.mark_text(
    align='left',
    baseline='middle',
    dx=7
).encode(
    text='words'
)

points + text

st.write('Copyrights of all the data sources belongs to OneTrust')







