#!/usr/bin/env python
# coding: utf-8

# # 平均値、中央値、モード

# ## 平均値 vs. 中央値

# 疑似的な収入分布を作りましょう。27000を中央にした正規分布で、標準偏差は15000、データ数は10000とします。（用語については後程解説します。）
# そして、平均値を計算してみましょう。27000に近くなるはずです。

# In[6]:


import numpy as np

## 正規分布、27000の中央値、15000の標準偏差、10000の数値
incomes = np.random.normal(27000, 15000, 10000)
np.mean(incomes)


# 収入データを50個にセグメント化し、ヒストグラムにプロットすることもできます。

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

## ヒストグラム化
plt.hist(incomes, 50)
plt.show()


# 中央値を計算してみましょう。今回はきれいな分布をしているため、中央値は平均値と同様に27000に近くなります。

# In[9]:


## 中央値の計算：中央値は平均と近しい結果になる
np.median(incomes)


# 大富豪をデータに混ぜてみましょう。所得格差を実感してみましょう。

# In[10]:


incomes = np.append(incomes, [1000000000])


# 中央値は大きく変わりませんが、平均値は大きく変わります。

# In[11]:


np.median(incomes)


# In[13]:


## 平均値を大きく歪ませる可能性があるから、データの異常値を検知する必要がある
np.mean(incomes)


# ## モード

# 次に、疑似的な年齢のデータを500人分作りましょう。

# In[17]:


## 18〜90の乱数を500作る
ages = np.random.randint(18, high=90, size=500)
ages


# In[19]:


from scipy import stats

## モードの値（最頻値）を取得する
stats.mode(ages)


# In[ ]:




