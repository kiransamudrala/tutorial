import streamlit as st
import altair as alt
import pandas as pd
#import random

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


