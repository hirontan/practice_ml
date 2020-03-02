#!/usr/bin/env python
# coding: utf-8

# # Numpy：数値計算

# In[1]:


import numpy as np


# In[2]:


# ベクトルの定義
x = np.array([1, 2, 3])


# In[3]:


x


# In[4]:


y = np.array([2, 3.9, 6.1])


# In[5]:


y


# # データの中心化

# In[6]:


# 平均の算出
x.mean()


# In[7]:


y.mean()


# # 中心化

# In[8]:


xc = x- x.mean()


# In[9]:


xc


# In[10]:


yc = y - y.mean()


# In[11]:


yc


# # パラメータaの計算

# In[12]:


# 要素ごとの掛け算（要素積）
xx = xc * xc


# In[13]:


xx


# In[14]:


xy = xc * yc


# In[15]:


xy


# In[16]:


xx.sum()


# In[17]:


xy.sum()


# In[18]:


a = xy.sum() / xx.sum()


# In[19]:


# 傾き
a


# # Pandas：データベースの操作

# In[20]:


import pandas as pd


# In[21]:


# CSVファイルの読み込み
# df: data frame
df = pd.read_csv('./data/sample.csv')


# In[22]:


print(df)


# In[23]:


# dfだけだと綺麗に整理される
df


# In[24]:


# 3つだけ表示したい
df.head(3)


# In[25]:


# データの抽出（辞書型のようにカラムから取得できる）
x = df['x']
y = df['y']


# # Matplotlib：グラフの描画

# In[26]:


import matplotlib.pyplot as plt


# In[27]:


# 横軸をx、縦軸をyの散布図（scatter）をプロット 
plt.scatter(x, y)
plt.show()


# # 単回帰分析の実装

# ## データの中心化

# In[28]:


# データの概要を表示
df.describe()


# In[29]:


x.mean()


# In[30]:


df.mean()


# In[31]:


# 中心化：データフレームごと平均を引くことができる
df_c = df  - df.mean()


# In[32]:


df_c.head(3)


# In[33]:


df_c.describe()


# In[34]:


# データの抽出
x = df_c['x']
y = df_c['y']


# In[35]:


# xとyの散布図をプロット 
plt.scatter(x,y)
plt.show()


# ## パラメータaの計算
# 傾き$a$の計算式
# $$
# a = \dfrac{\displaystyle{\sum_{n=1}^{N}}x_{n}y_{n}}
# {\displaystyle{\sum_{n=1}^{N}}x_{n}^{2}}
# $$

# In[36]:


xx = x * x # * は要素積


# In[37]:


xy = x * y


# In[38]:


a = xy.sum() / xx.sum()


# In[39]:


a


# ## プロットして確認
# 予測値
# $$
# \hat{y} = ax
# $$

# In[40]:


plt.scatter(x, y, label='y') # 実測値
plt.plot(x, a*x, label='y_hat', color='red') # 予測値
plt.legend() # 凡例の表示
plt.show()


# ## 予測値の計算
# $$
# \begin{align}
# y - {\bar y} &= a (x-{\bar x}) \\
# \rightarrow y &= a (x-{\bar x}) + {\bar y}
# \end{align}
# $$

# In[41]:


x_new = 40 # 40平米の部屋


# In[42]:


mean = df.mean()


# In[43]:


mean['x']


# In[44]:


# 中心化
xc = x_new - mean['x']


# In[45]:


xc


# In[46]:


# 単回帰分析による予測
yc = a * xc


# In[47]:


yc


# In[48]:


# 元のスケールの予測値
y_hat = a * xc + mean['y']


# In[49]:


y_hat


# ## 予測値を計算する関数の作成

# In[50]:


# predict：予測する
def predict(x):
    # 定数項
    a = 10069.022519284063
    xm = 37.62222
    ym = 121065.0
    # 中心化
    xc = x - xm
    # 予測値の計算
    y_hat = a * xc + ym
    # 出力
    return y_hat


# In[51]:


# 予測値
predict(40)


# In[52]:


predict(30)


# In[53]:


# 元々持っているデータが30平米の範囲、外挿であるとマイナスの値が出力される
# もし予測をするなら、データを入力する必要がある
predict(10)


# In[ ]:




