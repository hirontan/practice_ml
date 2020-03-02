#!/usr/bin/env python
# coding: utf-8

# # グラフの描画

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# グラフを表示させるためのおまじない
get_ipython().run_line_magic('matplotlib', 'inline')

# data 作成
np.random.seed(1)            # 乱数を固定
x = np.arange(10)
y = np.random.rand(10)

# グラフ表示
plt.plot(x, y)               # 折れ線グラフの登録
plt.show()                   # グラフを描写


# In[2]:


# これまでの履歴をリセットする
get_ipython().run_line_magic('reset', '')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 関数を作成する
def f(x):
    return (x - 2) * x * (x + 2)


# In[4]:


print(f(1))


# In[5]:


f(1)


# In[6]:


print(f(np.array([1, 2, 3])))


# In[7]:


# 描画する範囲を決める
x = np.arange(-3, 3.5, 0.5)


# In[8]:


x


# In[9]:


# linspace(n1, n2, n)　　n1〜n2を等間隔でn個の点に分けた点を返す
x = np.linspace(-3, 3, 10)
np.round(x, 2)


# In[10]:


plt.plot(x, f(x))
plt.show()


# In[11]:


def f2(x, w):
    return (x - w) * x * (x + 2)      #(A) 関数定義

# x を定義
x = np.linspace(-3, 3, 100)           # (B) x を100 分割にする

# グラフ描写
plt.plot(x, f2(x, 2), color='black', label='$w=2$') # (C)
plt.plot(x, f2(x, 1), color='cornflowerblue',
         label='$w=1$')                # (D)
plt.legend(loc="upper left")           # (E) 凡例表示
plt.ylim(-15, 15)                      # (F) y 軸の範囲
plt.title('$f_2(x)$')                  # (G) タイトル
plt.xlabel('$x$')                      # (H) x ラベル
plt.ylabel('$y$')                      # (I) y ラベル
plt.grid(True)                         # (J) グリッド
plt.show()


# In[12]:


# グラフを並べる
plt.figure(figsize=(10, 3))                 # (A) figure を指定
plt.subplots_adjust(wspace=0.5, hspace=0.5) # (B) グラフの間隔を指定
for i in range(6):
    plt.subplot(2, 3, i + 1)                # (C) グラフ描写の位置を指定
    plt.title(i + 1)
    plt.plot(x, f2(x, i), 'k')
    plt.ylim(-20, 20)
    plt.grid(True)
plt.show()


# In[13]:


get_ipython().run_line_magic('reset', '')


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 関数f3 を定義
def f3(x0, x1):
    ans = (2 * x0**2 + x1**2) * np.exp(-(2 * x0**2 + x1**2))
    return ans

# 各x0,x1 でf3 を計算
xn = 9
x0 = np.linspace(-2, 2, xn)            # (A)
x1 = np.linspace(-2, 2, xn)            # (B)
y = np.zeros((len(x0), len(x1)))       # (C)
for i0 in range(xn):
    for i1 in range(xn):
        y[i1, i0] = f3(x0[i0], x1[i1]) # (D)


# In[15]:


x0


# In[16]:


np.round(y, 1)


# In[ ]:


plt.figure(figsize=(3.5, 3))
plt.gray()                     # (A)
plt.pcolor(y)                  # (B)
plt.colorbar()                 # (C)
plt.show()


# In[ ]:




