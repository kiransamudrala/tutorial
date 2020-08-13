import streamlit as st
import altair as alt
import pandas as pd
import random

st.title('OneTrust Articles')

dfall= pd.read_csv('word2vec_output.csv')
st.write('The below chart is a reduced dimensional representation of the \
         output of word2vec model applied to a bunch of news articles \
        downloaded from http://www.dataguidance.com/news. 300 articles were read and analyzed using \
        word2vec model. And further PCA was applied on the results. \
        The proximity of the words on the chart is a function of \
        their meaning or their co-occurrence in the documents. \
        Since the first two components were only able to capture 14% of variability in the data \
        we can still see some words which are actually not next to eachother \
        seemingly very close to eachother in the chart.')

source = dfall
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
seed = random.randint(0,len(dfall)+n-1)
df = dfall.iloc[seed:seed+n-1,:]
#chart = alt.Chart(df).mark_circle().encode(x='x', y='y',tooltip='words',text='words')


source = df

points = alt.Chart(source).mark_point().encode(
    x='x',
    y='y'
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