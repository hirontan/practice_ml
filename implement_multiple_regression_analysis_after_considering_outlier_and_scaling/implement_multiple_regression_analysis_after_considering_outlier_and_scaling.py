#!/usr/bin/env python
# coding: utf-8

# # データの読み込み

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


# CSVファイルの読み込み
df = pd.read_csv('./data/housing.csv')


# In[4]:


df.head(3)


# # 分布の確認

# In[25]:


import seaborn as sns


# In[26]:


sns.distplot(df['x6'], bins=20)


# # 外れ値除法（３σ法）

# In[27]:


col = 'x6'


# In[28]:


# 平均
mean = df.mean()
mean


# In[29]:


mean[col]


# In[30]:


# 標準偏差（standard deviation）
sigma = df.std()
sigma


# In[31]:


low = mean[col] - 3 * sigma[col]
low


# In[32]:


high = mean[col] + 3 * sigma[col]
high


# In[33]:


df2 = df[(df[col] > low) & (df[col] < high)]


# In[34]:


len(df)


# In[35]:


len(df2)


# In[36]:


# 分布の確認
sns.distplot(df['x6']) # オリジナル


# In[38]:


# 外れ値の除去ができている
sns.distplot(df2['x6'])


# In[39]:


# 正規分布ではないので、外れ値を検出しにくい
sns.distplot(df2['x1'])


# In[41]:


cols = df.columns
cols


# In[43]:


_df = df
for col in cols:
    # 3σ法の上下限値を設定
    low = mean[col] - 3 * sigma[col]
    high = mean[col] + 3 * sigma[col]
    # 条件での絞り込み
    _df = _df[(_df[col] > low) & (_df[col] < high)]


# In[44]:


_df


# In[45]:


# オリジナル
len(df)


# In[47]:


# ３σ法適用後（正規分布に沿っていなかったため、20%ぐらい減ってしまっている。）
len(_df)


# サンプルが減る場合の対処法
# 
# - 外れ値は取り除く
# - 外れ値を平均もしくは中央値などで埋める
# - 主成分分析等を使って、潜在変数に変換した後に３σ法を適用 ← 高度

# In[49]:


# 入力変数と出力変数に分割
_df.head(3)


# In[50]:


# df.iloc[行, 列]
X = _df.iloc[:, :-1]
y = _df.iloc[:, -1]


# # 訓練データと検証データに分割

# In[51]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# # 重回帰分析

# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


# モデルを宣言
model = LinearRegression()


# In[65]:


# モデルの学習
model.fit(X_train, y_train)


# In[66]:


# 検証 ← 訓練データ
model.score(X_train, y_train)


# In[67]:


# 検証 ← 検証データ
model.score(X_test, y_test)


# In[68]:


# 過学習（オーバーフィッティング）


# # スケーリング

# In[70]:


from sklearn.preprocessing import StandardScaler


# In[71]:


# scalerの宣言
scaler = StandardScaler()


# In[72]:


# scalerの学習 ← 平均と標準偏差を計算
scaler.fit(X_train)


# In[74]:


# scaling
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)


# In[76]:


# ３σ法だと-3〜3の間になるので、1を超えていることもある
X_train2


# In[77]:


# モデルの宣言
model = LinearRegression()


# In[78]:


# モデルの学習
model.fit(X_train2, y_train)


# In[79]:


# 検証 ← 訓練データ
model.score(X_test2, y_test)


# In[80]:


# 重みの確認
model.coef_


# In[82]:


np.set_printoptions(precision=3, suppress=True)


# In[83]:


model.coef_


# In[84]:


sns.distplot(_df['x6'])


# In[ ]:




