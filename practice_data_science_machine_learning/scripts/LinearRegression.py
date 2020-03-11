#!/usr/bin/env python
# coding: utf-8

# # 線形回帰

# ページの表示速度と購入量の関係を線形に示すデータを作りましょう。

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from pylab import *

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

scatter(pageSpeeds, purchaseAmount)


# 二つしか特徴量がないので、scipy.state.linregressを使ってシンプルに行きましょう。

# In[2]:


from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)


# 驚くべきことではないのですが、R-2乗値がよいフィットを示しています。

# In[3]:


r_value ** 2


# この回帰の傾きと切片を用いて、直線を描画してみましょう。

# In[4]:


import matplotlib.pyplot as plt

def predict(x):
    return slope * x + intercept

fitLine = predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()


# ## アクティビティー

# テストデータの分散を大きくして、R-2乗値がどのような影響を受けるのか確認してみましょう。

# In[ ]:




