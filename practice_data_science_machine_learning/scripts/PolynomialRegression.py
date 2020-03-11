#!/usr/bin/env python
# coding: utf-8

# # 多項式回帰

# 実際のデータは、線形でない場合も多いです。多項式回帰を用いて、より現実に即したページ速度と購入量のデータを見ていきましょう。

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
import numpy as np

np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

scatter(pageSpeeds, purchaseAmount)


# numpyは簡単に使える多項式回帰の関数を持っています。 n次の多項式で、二乗誤差が最小になるようにフィッティングをしてみましょう。今回は4次の多項式を用います。

# In[2]:


x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

p4 = np.poly1d(np.polyfit(x, y, 4))


# 散布図とともに、ページ速度0-7の範囲で多項式を表示します。

# In[3]:


import matplotlib.pyplot as plt

xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()


# とてもよくフィットできていますね。R-二乗値を測定してみましょう。

# In[4]:


from sklearn.metrics import r2_score

r2 = r2_score(y, p4(x))

print(r2)


# ## アクティビティー

# 異なる次数の多項式回帰を試してみよう。高い次数の時、より良いフィッティングが得られるでしょうか？もしくは、R-二乗値が大きくても過剰適合の兆候が見られたりするでしょうか。

# In[ ]:




