#!/usr/bin/env python
# coding: utf-8

# # モーメント: 平均、分散、歪度、尖度

# モーメント：確率密度函数の形状の定量化
# 
# - 平均：一次
# - 分散：二次
# - 歪度（分布がどれだけ偏っているか）：三次
# - 尖度（ピークがどれだけ尖っているか）：四次
# 
# ほぼ正規分布のランダムなデータセットを用意します。

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

vals = np.random.normal(0, 0.5, 10000)

plt.hist(vals, 50)
plt.show()


# 一次のモーメントは”平均”。データの平均値は０に近くなる。 

# In[17]:


np.mean(vals)


# 二次のモーメントは”分散”。

# In[18]:


np.var(vals)


# 三次のモーメントは”歪度（skewness）”。今回のデータは０を中心とした対称に近いので、歪度はほぼ０となる。

# In[19]:


import scipy.stats as sp
sp.skew(vals)


# 四次のモーメントは”尖度（kurtosis）”。ピークの鋭さを表す。大きいほど鋭い。

# In[20]:


sp.kurtosis(vals)


# In[ ]:




