#!/usr/bin/env python
# coding: utf-8

# # 標準偏差と分散

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

## 正規分布、60の中央値、70の標準偏差、10000の数値
incomes = np.random.normal(100.0, 70.0, 10000)

plt.hist(incomes, 50)
plt.show()


# In[23]:


## 標準偏差
incomes.std()


# In[24]:


## 分散
incomes.var()


# 
# 
