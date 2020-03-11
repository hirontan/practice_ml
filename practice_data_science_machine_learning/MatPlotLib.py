#!/usr/bin/env python
# coding: utf-8

# # MatPlotLibの基礎

# ## グラフの描画

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# arange：-3 ≦ n < 3で間隔は0.01
x = np.arange(-3, 3, 0.01)

# Normal Distribution（正規分布）
plt.plot(x, norm.pdf(x))
plt.show()


# ## 複数のグラフの描画

# In[2]:


plt.plot(x, norm.pdf(x))

# 1.0の位置、区間0.5
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()


# ## ファイルに保存

# In[3]:


plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.savefig('./images/MyPlot.png', format='png')


# ## 軸の調整

# In[5]:


axes = plt.axes()

# x軸の範囲
axes.set_xlim([-5, 5])

# y軸の範囲
axes.set_ylim([0, 1.0])

# x軸の目盛り
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

# y軸の目盛り
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()


# ## グリッドの追加

# In[6]:


axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# グリッドの追加
axes.grid()
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()


# ## 線の種類と色の変更

# In[7]:


axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()

# plotの第３引数、b = blue / r = red / - = 線 / : = 点線
plt.plot(x, norm.pdf(x), 'b-')
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')
plt.show()


# ## 軸のラベルと凡例

# In[8]:


axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()
plt.xlabel('Greebles')
plt.ylabel('Probability')
plt.plot(x, norm.pdf(x), 'b-')
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')

# 凡例 loc = 4：グラフの右下
plt.legend(['Sneetches', 'Gacks'], loc=4)
plt.show()


# ## XKCDスタイル

# In[9]:


# アメリカで人気のコミック。漫画調のものを使える
plt.xkcd()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-30, 10])

data = np.ones(100)
data[70:] -= np.arange(30)

plt.annotate(
    'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
    xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

plt.plot(data)

plt.xlabel('time')
plt.ylabel('my overall health')


# ## 円グラフ

# In[10]:


# Remove XKCD mode:
plt.rcdefaults()

values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
explode = [0, 0, 0.2, 0, 0]
labels = ['India', 'United States', 'Russia', 'China', 'Europe']

plt.pie(values, colors= colors, labels=labels, explode = explode)
plt.title('Student Locations')
plt.show()


# ## 棒グラフ

# In[11]:


values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
plt.bar(range(0,5), values, color= colors)
plt.show()


# ## 散布図

# In[12]:


from pylab import randn

# randn ：　正規分布に従った乱数を発生できる
X = randn(500)
Y = randn(500)
plt.scatter(X,Y)
plt.show()


# ## ヒストグラム

# In[13]:


# 平均27000 標準偏差15000 サイズ10000
incomes = np.random.normal(27000, 15000, 10000)
# 50はセグメントの数
plt.hist(incomes, 50)
plt.show()


# ## 箱ひげ図

# データの広がりと歪みを視覚化するのに便利。赤い線はデータの平均値を表す。
# 箱の中には、下位1/4より大きく、上位3/4より小さいデータが入る。従って、半数のデータは箱の中に存在する。
# 点線のひげはデータの範囲を表す。外れ値はひげの外側にプロットされる。外れ値は、箱の上端、下端から箱の高さの1.5倍以上離れた値とする。
# 下の例は、-40から60までの均一に分布したデータに100以上、及び-100以下の外れ値を加えたもの。

# In[16]:


uniformSkewed = np.random.rand(100) * 100 - 40
high_outliers = np.random.rand(10) * 50 + 100
low_outliers = np.random.rand(10) * -50 - 100
data = np.concatenate((uniformSkewed, high_outliers, low_outliers))

# 外れ値は、点で描かれる
plt.boxplot(data)
plt.show()


# In[ ]:




