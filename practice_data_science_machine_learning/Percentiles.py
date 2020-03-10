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


# In[2]:


## 全体の
np.percentile(vals, 50)


# In[3]:


np.percentile(vals, 90)


# In[4]:


np.percentile(vals, 20)


# ## アクティビティ

# テストデータを作る際に、異なるパラメータを使ってみよう。パーセンタイルにどのような影響を与えるでしょうか？

# In[ ]:




