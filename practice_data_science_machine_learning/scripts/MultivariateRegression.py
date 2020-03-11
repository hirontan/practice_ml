#!/usr/bin/env python
# coding: utf-8

# # 多変量回帰

# 中古車の属性と価格の関係を見ていきましょう。

# In[1]:


import pandas as pd

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')


# In[2]:


df.head()


# pandasを使うことで、目的の属性のベクトルを取り出すことができます。そして、値の予想を行うことができます。
# 
# pandas.Categoricalを使うことで、テキストベースのカテゴリデータ（Model）を順序を表す整数に変更することができます。これにより、多変量解析で扱うことができるようになります。

# In[3]:


import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print (X)

est = sm.OLS(y, X).fit()

est.summary()

上記の表のcoefから、以下のように多変量回帰の式を決めることができます。
    B0 + B1 * Mileage + B2 * model_ord + B3 * doors
    
上記の例では標準誤差（std err）を見る限り走行距離（Mileage）が他の属性と比較して一番重要です。
# In[5]:


y.groupby(df.Doors).mean()


# 上記ではドア数別の平均価格を求めています。2ドアの方が価格が高いという結果になりました。

# ## アクティビティー

# 上記のコードに、Type属性を加えて多変量解析をしてみよう。Type属性は、価格の要因として重要でしょうか？

# In[ ]:




