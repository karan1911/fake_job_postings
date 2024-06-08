#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import tensorflow as tf
import keras 
from keras.models  import Sequential
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense , Dropout , Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense,LSTM 
from keras.layers import Bidirectional


# In[2]:


fake = pd.read_csv(r'E:\fake_job_postings.csv') #  read the file
fake


# In[3]:


df = pd.DataFrame()
df['X'] = fake['company_profile'].apply(str)+''+fake['description'].apply(str)+' '+fake['requirements'].apply(str)+' '+fake['required_experience'].apply(str)+'  '+fake['has_company_logo'].apply(str)+''+fake['benefits'].apply(str)+''+fake['industry'].apply(str)+''+fake['required_experience'].apply(str)


# In[4]:


df["Y"] = fake.fraudulent # target Variable & Y variable


# In[5]:


df.Y.value_counts()


# In[6]:


df


# In[7]:


df_x = df.iloc[: , 0]
df_y = df.iloc[: , 1]


# In[8]:


from sklearn.model_selection import  train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2 )


# In[9]:


train_y = to_categorical(train_y)


# In[10]:


max_num_words = 15000 # entire document (corpus)
seq_len = 100         # single documents
embedding_size = 200  # for each word embedding of size 100


# In[11]:


tokenizer = Tokenizer(num_words = max_num_words)
tokenizer.fit_on_texts(df.X)
train_x = tokenizer.texts_to_sequences(train_x)


# In[12]:


train_x = pad_sequences(train_x , maxlen=seq_len)


# In[13]:


test_x = tokenizer.texts_to_sequences(test_x)
test_x = pad_sequences(test_x , maxlen= seq_len)


# In[14]:


model = Sequential()
model.add(Embedding(input_dim = max_num_words,
#                    input_length = seq_len , 
                    output_dim = embedding_size))


# In[15]:


model.add(Bidirectional(LSTM(64)))
# model.add(LSTM(20))
model.add(Dense(2 , activation='softmax'))
from tensorflow.keras.optimizers import Adam
adam = Adam(learning_rate=.01)
model.compile(optimizer=adam , loss = 'categorical_crossentropy', metrics=['accuracy'])


# In[16]:


model.fit(train_x,train_y , epochs=10 ,validation_split=.1  , batch_size = 350 )


# In[17]:


pred = model.predict(test_x)


# In[18]:


pred_classes = pred.argmax(axis = 1)


# In[19]:


from sklearn.metrics import confusion_matrix ,accuracy_score, recall_score , precision_score ,f1_score,classification_report


# In[20]:


confusion_matrix(test_y,pred_classes)


# In[21]:


print(classification_report(test_y,pred_classes))


# In[ ]:





# In[ ]:




