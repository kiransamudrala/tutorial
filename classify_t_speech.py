from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import os
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans 
import altair as alt



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


# read all files in the folder
path = 'data/'
filenames = os.listdir(path)
head = []
text = []
fns = []
for filename in filenames:
    if filename[0] == 't':
        f = open(path+filename,'r')
        lines = f.readlines()
        f.close()
        fns.append(filename)
        head.append(lines[0])
        text.append(lines[1])

# remove stopwords from each text
stopsdf = pd.read_csv('stops.txt')
stopsdf = stopsdf[['0']]
stopsdf.columns=['stopwords']
stops = stopsdf['stopwords'].values.tolist()
# stops = stopwords.words('english')
# stops = stops +['also','under','many','back','applause','if','us','like','really','going','know','said','get','want','say','one','need','new','every']
# dfstop = pd.DataFrame(stops)
# dfstop.to_csv('stops.txt')
text2 = []
for t in text:
    t=t.lower()
    t=re.sub('[^a-zA-Z]', ' ', t )
    t=re.sub(r'\s+', ' ', t)
    for s in stops:
        t=t.replace(' '+s+' ',' ')
        t=t.replace('.'+s+' ',' ')
    text2.append(t)


fm = get_tfidf(text2)
fmt  = fm.transpose()
pca = PCA(n_components=74)
pcs = pca.fit_transform(fmt)


# eigenvalues
pdf = pd.DataFrame(data = pcs[:,0:5], columns = ['pc1', 'pc2','pc3','pc4','pc5'])
pdf.plot(x='pc1',y='pc2',style='o')

p1 = [-0.1,-0.3]
p2 = [0.18, 0.4]
x_values = [p1[0], p2[0]]
y_values = [p1[1], p2[1]]
plt.plot(x_values, y_values)

p3 = [0.1,-0.3]
p4 = [0.4, 0.3]
x_values = [p3[0], p4[0]]
y_values = [p3[1], p4[1]]
plt.plot(x_values, y_values)

plt.show()

ev = pca.explained_variance_
evp = ev/sum(ev)

# plot scree cumulative
cumulative = np.flipud(np.cumsum(ev))
plt.plot(cumulative,range(len(ev)), c='blue')
plt.title('Eigenvalue Plot')
plt.show()


def find_d(x1,y1,x2,y2,x,y):
    return (x-x1)*(y2-y1)-(y-y1)*(x2-x1)

# k-means clustering
km = KMeans(3,).fit(np.array(pdf).reshape(-1,1))
km_labels = km.labels_


# visual clustering
at=''
bt=''
ct=''
k1=''
k2=''
k3=''
clist=[]
for ind in range(len(head)):
    d1 = find_d(p1[0],p1[1],p2[0],p2[1],pdf.iloc[ind,0],pdf.iloc[ind,1])
    d2 = find_d(p3[0],p3[1],p4[0],p4[1],pdf.iloc[ind,0],pdf.iloc[ind,1])
    if d1<0:
        at = at+text2[ind]
        clss='blue'
    elif d2<0:
        bt = bt+text2[ind]
        clss='red'
    else:
        ct = ct+text2[ind]
        clss='green'
    if km.labels_[ind]==0:
        k1 += text2[ind]
    elif km.labels_[ind]==1:
        k2 += text2[ind]
    else:
        k3 += text2[ind]
    clist.append(clss)
    print(d1,d2)
plt.scatter(pdf.iloc[:,0],pdf.iloc[:,1],color=clist)
plt.show()

pdf2 = pdf.iloc[:,0:2]
pdf2 = pd.concat([pdf2,pd.DataFrame(clist,columns=['clist']),pd.DataFrame(head,columns=['speech_header'])],axis=1)

cdata = [at,bt,ct]
kdata = [k1,k2,k3]
fmc = get_tfidf(cdata)
fmk = get_tfidf(kdata)

blue = fmc[['col1']].sort_values(by=['col1'],ascending=False)
red = fmc[['col2']].sort_values(by=['col2'],ascending=False)
green = fmc[['col3']].sort_values(by=['col3'],ascending=False)

k1data = fmk[['col1']].sort_values(by=['col1'],ascending=False)
k2data = fmk[['col2']].sort_values(by=['col2'],ascending=False)
k3data = fmk[['col3']].sort_values(by=['col3'],ascending=False)

st.title(' Trump Speech Classification')
st.write('This natural language processing example displays the results of analyzing transcripts of seventy-four different speeches of President Trump ')
st.write('A) DATA MODEL: bag-of-words + tfidf')
st.write('tfidf = Term Frequency - Inverse Document Frequency')
st.write('\tThis tfidf is a measure of frequency of occurrence of a given word in the current document when compared to the entire corpus.So based on the tfidf calculated for each of the seventy-four transcripts, here are the most important words by their decreasing order of tf-idf:')
Total = fmt.sum().sort_values(ascending=False)
st.dataframe(Total.index,width=200)
st.write('If tfidf from the bag-of-words data representation model is the feature we choose, then our speech data is a point in 7000+ dimensional space. Plotting the files data by tfidf of two most important words:')
fmt2 = fmt[['people','clinton']]
fmt2['speech_header']=head
c1 = alt.Chart(fmt2).mark_circle().encode(x='people', y='clinton',tooltip='speech_header')
st.altair_chart(c1, use_container_width=True)
st.write('There are no evidently different clusters. Also, given that there are 7000+ dimensions, a natural first step is to apply dimensionality reduction to revisualize this high dimensional data.')

st.write('B) DIMENSIONALITY REDUCTION: PRINCIPAL COMPONENT ANALYSIS (PCA)')
st.write('We choose PCA due to its simplicity in being able to interpret the axes. Applying PCA renders somewhat visually disparate clusters. The scree plot of eigenvalues promised that we would only be covering the 13% of variability in the data by viewing the first two components of PCA. Nevertheless, PCA is always a great first step in the right direction when there are no labels on the data. ')
c2 = alt.Chart(pdf2).mark_circle().encode(x='pc1', y='pc2',color=alt.Color('clist', scale=None),tooltip='speech_header')
st.altair_chart(c2, use_container_width=True)
st.write('C) CLUSTERING: VISUAL AND MANUAL ')
st.write('In the above PCA output plot I have marked the visually distinct clusters in different colors. Please note that the k-means clustering was not doing a good job in our case - either because we have to improvise our features or because the data is lying on a non-linear manifold (See APPENDIX below). Now let us calculate the tfidf of each individual cluster to see what is marking them differently')
st.image('important_words.png',width=800)

st.write('D) DATA ANALYSIS:')
st.markdown("""
            <ul><li> Moving along from left to right the amount of focus on words related to opponents (\'Hillary\',\'Clinton\') kept going down </li>
            <li> From left to right the speeches started to transform from more formal to less formal 
            <ul>
            <li> Blue formal cluster: trade/world/government/president</li>
            <li> Red cluster: less formal with believe/cheers </li>
            <li> Green cluster: pep talk like words got/look/go/way/win </li> </ul> </li> </ul>
            """, unsafe_allow_html=True)
st.write('E) PERFORMANCE MEASURE: If more formal methods like k-means clustering worked, we could have leveraged cluster validity indices like Dunn/Silhouette to measure the performance of the model. Unfortunately, for a visual clustering method, we do not have a good way to measure the performance of the model.')
st.write('F) NEXT STEPS: Other directions to pursue are:')
st.markdown("""
            <li> explore rule-based classification </li> 
            <li> if we have labelled data, we can apply supervised techniques like decision trees </li>
            <li> apply non-linear dimensionality reduction techniques like isomap to extract a better reduced representation </li>
            <li> define richer features like triples or topics of the speech to define our text data instead of bag-of-words </li>
            """, unsafe_allow_html=True)
            
st.write('')
st.write('')
st.write('APPENDIX:')
st.write(' Here is why I did not use k-means clustering:')

kmcolor=[]
for k in km.labels_:
    if k==0:
        kmcolor.append('blue')
    elif k==1:
        kmcolor.append('red')
    else:
        kmcolor.append('green')
pdf2['kmcluster']=pd.DataFrame(kmcolor)
c3 = alt.Chart(pdf2).mark_circle().encode(x='pc1', y='pc2',color=alt.Color('kmcluster',scale=None),tooltip='speech_header')
st.altair_chart(c3, use_container_width=True)

# extract top words for each of the kmclusters
st.write(' Just like what we did with visual clusters, let us calculate tfidf of each individual k-means cluster to see what is making k-means algorithm mark the clusters differently')
st.image('kmeans_imp_words.png',width=800)

st.write('k-means is unable define clusters differently - atleast for the current combination of features and model. The two right boxes pretty much contain the same set of important words')
st.write('I have tried giving different values of k and was still able to not get a good distinct set of clusters. This strongly suggest we need to revise our choices of data model (bag-of-words) and method (PCA)')

st.write('I look forward to sharing new perspectives on this data and many others soon! Until then stay safe! :)')