#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/karthikmahendra/Desktop/Salty/salty-trolls-cce24104ff9f.json'


# In[123]:


from google.cloud import bigquery
import time
import pandas as pd
import numpy as np


# In[9]:


bq_client = bigquery.Client()


# In[10]:


QUERY = '''SELECT  * 
        FROM `bigquery-public-data.hacker_news.full_201510`
        WHERE `bigquery-public-data.hacker_news.full_201510`.by = "danmaz74"
        '''


# In[29]:


df = bq_client.query(QUERY).to_dataframe()


# In[31]:


df.shape


# In[32]:


df.head()


# In[24]:


# job_config = bigquery.QueryJobConfig()
# now=time.time()
# query_job=bq_client.query(QUERY,location='US')
# res=query_job.result()
# print('query took:',round(time.time()-now,2),'s')


# In[13]:


# now=time.time()
# destination_uri = "gs://karthikkm028-ds/*hacker.csv"
# dataset_ref = bq_client.dataset("hacker_news", project='bigquery-public-data')
# table_ref = dataset_ref.table("full_201510")

# extract_job = bq_client.extract_table(
#     table_ref,
#     destination_uri)
# extract_job.result()  # Waits for job to complete
# print('create table and write to GCS took:',round(time.time()-now,2),'s')


# In[60]:


# gcs = gcsfs.GCSFileSystem(project='bigquery-public-data') 
# files=gcs.glob(destination_uri)
# df = pd.concat([pd.read_csv('gs://'+f) for f in files], ignore_index=True)
# print('read csv took:',round(time.time()-now,2),'s')


# In[101]:


df_drop = df.drop(columns= ['score','title','url','deleted','dead','descendants','ranking'])


# In[102]:


df_drop.head()


# In[103]:


df_drop.shape


# In[104]:


df_drop.dtypes


# In[61]:


#df1 = df_drop.head(100)


# In[105]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[106]:


df_drop['text']=df_drop['text'].apply(str)


# In[107]:


type(df_drop['text'][0])


# In[108]:


s = sia.polarity_scores(df_drop['text'][0])


# In[109]:


s


# In[110]:


s['neg']


# In[113]:


#df_drop['neg'][0] = sia.polarity_scores(df_drop['text'][0])['neg']


# In[116]:


sia.polarity_scores(df_drop['text'][0])


# In[129]:


df_drop['neg']=np.zeros(df_drop.shape[0])
df_drop['pos']=np.zeros(df_drop.shape[0])
df_drop['neu']=np.zeros(df_drop.shape[0])


# In[130]:


for idx,x in df_drop['text'].iteritems():
    #print (idx,x)
    df_drop['neg'][idx] = sia.polarity_scores(x)['neg']
    df_drop['pos'][idx] = sia.polarity_scores(x)['pos']
    df_drop['neu'][idx]= sia.polarity_scores(x)['neu']


# In[131]:


df_drop.head()


# In[146]:


df_neg = df_drop[df_drop['neg']> 0.5]
df_pos = df_drop[df_drop['pos']> 0.5]


# In[153]:


df_neg = df_neg.sort_values(['neg'],ascending=False)
df_pos =df_pos.sort_values(['pos'],ascending=False)


# In[154]:


df_neg.head()


# In[156]:


df_pos.head(10)


# In[ ]:




