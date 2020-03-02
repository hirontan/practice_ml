#!/usr/bin/env python
# coding: utf-8

# # データの読み込み

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


# CSVファイルの読み込み
df = pd.read_csv('./data/housing.csv')


# In[5]:


df.head()


# In[6]:


# レコード数の確認
len(df)


# In[8]:


# 統計量の算出：データの不備があるかどうかのチェック（欠損値）
df.describe()


# # 分布の確認

# In[9]:


import seaborn as sns


# In[16]:


# 分布の確認
sns.distplot(df['y'])

### 何も表示されなければ
# %matplotlib inline
### どこかに入力する


# In[26]:


sns.distplot(df['x6'], bins=20)


# In[24]:


sns.distplot(df['x4'], bins=20)


# # 相関関係の確認

# In[20]:


# 相関係数（crrelation）の算出：-1〜1（0であれば、無相関）
df.corr()


# In[21]:


# 相関関係を目視で確認（数値にどこが相関関係があるのかわからない可能性があるので、図で表す）
sns.pairplot(df)


# # 入力変数と出力変数の切り分け

# In[27]:


df.head(3)


# In[28]:


# df.iloc[行, 列]


# In[29]:


df.iloc[0, 0]


# In[30]:


df.iloc[:, 0]


# In[32]:


df.iloc[:, 0:13]


# In[38]:


# 最後から一つ目の列まで
X = df.iloc[:, :-1]


# In[39]:


y = df.iloc[:, -1]


# # モデル構築と検証

# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


# モデルの宣言
model = LinearRegression()


# In[43]:


# モデルの学習
model.fit(X, y)


# In[44]:


# 検証（決定係数の計算）
model.score(X, y)


# # 訓練データ（train）と検証データ（test）
# 例）受験勉強用10年分の過去問を購入
# 
# ダメなケース
# - 10年分で勉強（学習）
# - 10年分で実力テスト（検証）
# 
# 良いケース
# - 前半の5年分で勉強（学習）← 訓練データ
# - 後半の5年分で実力テスト（検証）← 検証データ

# In[45]:


from sklearn.model_selection import train_test_split


# In[53]:


# 訓練データと検証データの分割（検証データを40%利用し、乱数を1に設定する）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1) # randam_stateは乱数のシードを固定（再現性の確保）


# In[54]:


# モデルの学習
model.fit(X_train, y_train)


# In[55]:


# 検証 ← 検証データ
model.score(X_test, y_test)


# In[56]:


# 検証 ← 訓練データ
model.score(X_train, y_train)


# # 予測値の計算

# In[59]:


x = X.iloc[0, :]
x


# In[60]:


# 予測値の計算
y_pred = model.predict([x])[0]
y_pred


# # モデルの保存

# In[62]:


from sklearn.externals import joblib


# In[64]:


# モデルの保存（ピックル）
joblib.dump(model, 'model.pkl')


# # モデルの読み込み

# In[65]:


model_new = joblib.load('model.pkl')


# In[66]:


x


# In[68]:


model_new.predict([x])[0]


# # パラメータの確認

# In[70]:


# パラメータwの値（重みが大きい場合、変数に影響を与えている）
model.coef_


# In[72]:


# 有効桁数を指定する。指数関数での表示も禁止
np.set_printoptions(precision=3, suppress=True)


# In[73]:


model.coef_


# In[74]:


df.head(3)


# 重み（パラメータw）を見るだけでは、どの変数が影響を与えているかわからない

# In[ ]:




