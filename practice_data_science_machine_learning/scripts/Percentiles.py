#!/usr/bin/env python
# coding: utf-8

# # パーセンタイル

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

vals = np.random.normal(0, 0.5, 10000)

plt.hist(vals, 50)
plt.show()


# In[5]:


## 全体の50%になる点を求める
np.percentile(vals, 50)


# In[6]:


## 全体の90%になる点を求める
np.percentile(vals, 90)


# In[7]:


## 全体の20%になる点を求める
np.percentile(vals, 20)


# In[ ]:




