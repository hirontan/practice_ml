#!/usr/bin/env python
# coding: utf-8

# ## ベクトル

# In[1]:


import numpy as np


# In[2]:


# ベクトルを定義
a = np.array([2, 1])
a


# In[3]:


type(a)


# In[5]:


# 縦ベクトル
c = np.array([[1, 2], [3, 4]])
c


# In[6]:


d = np.array([[1], [2]])
d


# In[7]:


# 転置
d.T


# In[9]:


d.T.T  # => 転置を２回繰り返すことで元に戻る


# In[10]:


# 足し算と引き算
b = np.array([1, 3])
a + b


# In[11]:


a - b


# In[12]:


# スカラーとの掛け算
2 * a


# In[14]:


# 内積
b = np.array([1, 3])
c = np.array([4, 2])
b.dot(c)


# In[16]:


# 二次元ベクトルの大きさ
a = np.array([1, 3])
np.linalg.norm(a)


# In[ ]:




