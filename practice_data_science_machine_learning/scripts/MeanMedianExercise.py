#!/usr/bin/env python
# coding: utf-8

# # 練習: 購入金額の平均値と中央値

# 疑似的な電子商取引における購入金額の分布を作るコードです。乱数を使って生成しています。

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(100.0, 20.0, 10000)

plt.hist(incomes, 50)
plt.show()


# このデータの、平均値と中央値を見つけてください。以下のコードブロックにコードを書いて、実行し正しい実行結果が得られることを確認しましょう。

# In[3]:


# 平均値
np.mean(incomes)


# In[4]:


# 中央値
np.median(incomes)


# 簡単な例から始めて、徐々にJupiterにコードを書くこととnumpyを扱うことに慣れていきましょう。
# ここまでのコードを用いて自由に遊んでみましょう。様々な分布を表示させたり、外れ値を追加してその影響を見たりしてみましょう。

# In[5]:


incomes = np.append(incomes, [1000000])


# In[7]:


plt.hist(incomes, 20)
plt.show()


# In[8]:


# 平均値
np.mean(incomes)


# In[9]:


# 中央値
np.median(incomes)


# In[ ]:




