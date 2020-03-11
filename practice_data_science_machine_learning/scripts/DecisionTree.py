#!/usr/bin/env python
# coding: utf-8

# # 決定木

# 最初に、フェイクの過去の雇用データを読み込みます。pandasを用いて、csvファイルをDataFrameに変換します。

# In[2]:


import numpy as np
import pandas as pd
from sklearn import tree

input_file = "c:/DataScience/DataScience/PastHires.csv"
df = pd.read_csv(input_file, header = 0)


# In[3]:


df.head()


# scikit-learnは決定木を扱うために全てのデータを数値データにする必要があります。したがって、Y、Nは1、0とし、Levels of Educationも0-2の数値とします。現実には、想定外のデータや失われたデータの扱い方も考慮する必要があります。map()関数を用いることで、想定外のデータはNaNとして扱うことができます。

# In[4]:


d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
df.head()


# 次に、決定木を構築するのに必要な特徴量を取り出します。

# In[5]:


features = list(df.columns[:6])
features


# 決定木を構築します

# In[7]:


y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# 決定木を図示しましょう。pyplot2がインストールされている必要があります。
# 
# value = [0. 5.]は、その点における0の不採用と5の採用を意味し、value = [3. 0.]はその点における3の採用と0の不採用を意味します。

# In[ ]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# ## ランダムフォレストを用いたアンサンブル学習

# 10個の決定木から成るランダムフォレストを用いて、特定の候補者が採用されるかどうか予想してみましょう。

# In[11]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (clf.predict([[10, 0, 4, 0, 0, 0]]))


# ## アクティビティ

# テストデータを修正し、あなたが好みの候補者を採用にして、決定木を作成してみよう。嫌いな候補者を採用にするケースも試してみよう。そして、作成した決定木を元の決定木と比較してみよう。

# In[ ]:




