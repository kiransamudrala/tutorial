
import streamlit as st
import altair as alt
import pandas as pd
import random

st.title('OneTrust Articles')

dfall= pd.read_csv('word2vec_output2.csv')
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
