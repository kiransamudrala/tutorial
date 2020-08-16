from bs4 import BeautifulSoup
from newspaper import Article 
import pandas as pd
import numpy as np
import requests
import math

#set number of pages to read
pages=20
offset=20

# Scrape the news results page for article urls
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
    # get topics/labels
    topics=[]
    topictags = soup.find_all('div',class_='field field-name-term-links field-type-ds field-label-inline inline')
    for t in topictags:
        temp = []
        for xx in t.find_all('a'):
            temp.append(xx.text)
        #print('****************************')
        topics.append(temp)
    #<div class="field field-name-term-links field-type-ds field-label-inline inline"><div class="field-label">Topics:&nbsp;</div><div class="field-items"><div class="field-item even"><a href="/search/news/topic/adequacy">Adequacy</a><a href="/search/news/topic/data-transfer">Data Transfer</a><a href="/search/news/topic/model-clauses">Model Clauses</a><a href="/search/news/topic/privacy-shield">Privacy Shield</a><a href="/search/news/topic/surveillance">Surveillance</a></div></div></div>
    return urls,topics

# get article urls
print('\nSTEP 1: FETCHING URLS')
urls = []
topics = []
for x in range(pages):
    x=x+offset
    motherurl = 'https://www.dataguidance.com/search/news?page='+str(x)
    curr_urls,curr_topics = geturls(motherurl)
    topics = topics + curr_topics
    urls = urls + curr_urls

text = ''
titles = []
articles = []
url_index = 0

print('\nSTEP 2: EXTRACTING CONTENT FROM EACH URL FROM A LIST OF '+ str(len(urls)))
# consolidate articles into text
for url in urls:
    url_index+=1
    print('\n reading url ',url_index,' / ', len(urls))
    art = Article(url, language="en") 
    art.download() 
    art.parse() 
    titles.append(art.title)
    nl = art.nlp() 
    # clean the full text from having unwanted lines 
    temp = art.text
    lines = temp.split('\n')
    cleantext=''
    # remove lines that start with 'You can' and contain 'here'
    # remove lines that start with 'Read' and contain 'here'
    for line in lines:
        if line.startswith('You can') and line.find('here')>-1 and len(line)<120:
            continue
        elif line.startswith('Read') and line.find('here')>-1 and len(line)<120:
            continue
        else:
            cleantext = cleantext + ' ' + line
    #print(cleantext)
    articles.append(cleantext)
    text += art.text 
    
print('\nSTEP 3: STACKING COLUMNS INTO A DATAFRAME')
df = pd.DataFrame(np.column_stack([urls,titles,articles,topics]),columns=['url','title','text','topics'])
n = len(df)
m = math.floor(n/2)

print('\nSTEP 4: SPLITTING THE DATASET INTO TRAIN AND TEST SETS')
df_train = df.iloc[0:m,:]
df_test = df.iloc[m:,:]


print('\nSTEP 5: STORING THE DATASET TO A FILE')
df.to_csv('data/ot.txt')
df_train.to_csv('data/train_ot.txt')
df_test.to_csv('data/test_ot.txt')
  




    
    
    
    
    
    