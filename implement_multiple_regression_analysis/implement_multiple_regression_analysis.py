#!/usr/bin/env python
# coding: utf-8

# # 行列演算の基礎

# $\boldsymbol{w} = (\boldsymbol{X}^{T}\boldsymbol{X})^{-1}\boldsymbol{X}^{T}\boldsymbol{y}$

# - ベクトルの定義
# - 行列の定義
# - 転置
# - 逆行列
# - 行列積

# In[1]:


import numpy as np


# In[5]:


# ベクトルの定義
x = np.array([[1], [2], [3]])
print(x)


# In[7]:


# 行列の定義
X = np.array([[1, 2], [3, 4]])
print(X)


# In[9]:


# 転置（transposeのT）
Xt = X.T


# In[12]:


# 逆行列
# linear algebra: 線形代数
X_inv = np.linalg.inv(X)
print(X_inv)


# In[14]:


# 行列積
XX_inv = np.dot(X, X_inv)
print(XX_inv)


# # よくある間違い

# In[39]:


# 転置しているつもりができないケース
x = np.array([1, 2, 3])
x


# In[17]:


x.T


# In[19]:


x = np.array([[1, 2, 3]])
x


# In[20]:


x.T


# In[21]:


x = np.array([[1], [2], [3]])
x


# # Numpyでよく使う処理

# In[24]:


X = np.array([
  [2, 3, 4],
  [1, 2, 3]
])
print(X)


# In[25]:


X.shape


# In[26]:


row, col = X.shape


# In[27]:


row


# In[28]:


col


# In[29]:


for x in X:
    print(x)
    print('--')


# # 演習問題

# $\boldsymbol{X} = \begin{bmatrix}
# 1 & 2 & 3 \\
# 1 & 2 & 5 \\
# 1 & 3 & 4 \\
# 1 & 5 & 9 \\
# \end{bmatrix}
# , \ \boldsymbol{y} = \begin{bmatrix}
# 1\\
# 5\\
# 6\\
# 8\\
# \end{bmatrix}
# $のとき
# 
# $\boldsymbol{w} = (\boldsymbol{X}^{T}\boldsymbol{X})^{-1}\boldsymbol{X}^{T}\boldsymbol{y}$
# 

# - Step1: $\boldsymbol{X}^{T}\boldsymbol{X}$
# - Step2: $(\boldsymbol{X}^{T}\boldsymbol{X})^{-1}$
# - Step3: $\boldsymbol{X}^{T}\boldsymbol{y}$
# - Step4: $\boldsymbol{w} = (\boldsymbol{X}^{T}\boldsymbol{X})^{-1}\boldsymbol{X}^{T}\boldsymbol{y}$
# 

# In[32]:


# Xの定義
X = np.array([
    [1, 2, 3],
    [1, 2, 5],
    [1, 3, 4],
    [1, 5, 9]
])
print(X)


# In[34]:


y = np.array([
    [1],
    [5],
    [6],
    [8]
])
print(y)


# In[35]:


# Step1
XtX = np.dot(X.T, X)
print(XtX)


# In[36]:


# Step2
XtX_inv = np.linalg.inv(XtX)
print(XtX_inv)


# In[37]:


# Step3
Xty = np.dot(X.T, y)
print(Xty)


# In[38]:


# Step4
w = np.dot(XtX_inv, Xty)
print(w)


# # Scilit-learnで実装

# In[40]:


import sklearn


# In[42]:


# 重回帰分析のみ読み込み
from sklearn.linear_model import LinearRegression


# In[47]:


# モデルの宣言
# model = LinearRegression()
model = LinearRegression(fit_intercept=False)


# In[48]:


# モデルの学習 ← パラメータの調整
model.fit(X, y)


# In[49]:


# 調整後のパラメータ
model.coef_


# In[50]:


model.intercept_


# In[52]:


# 予測精度 ← 決定係数：0.7があれば最初としては問題ない
model.score(X, y)


# In[53]:


# 予測値の計算
x = np.array([[1, 2, 3]])


# In[55]:


y_pred = model.predict(x)
y_pred


# In[ ]:




