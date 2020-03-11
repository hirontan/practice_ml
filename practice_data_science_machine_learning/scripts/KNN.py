#!/usr/bin/env python
# coding: utf-8

# # KNN (K近傍法)

# KNN（K近傍法）はシンプルなコンセプトです。データセットの中から、K個の最も近いアイテムを見つけます。それらのアイテムに”投票”させることにより、新しいデータの属性を予測します。 
# 
# 例として、MovieLensのデータを用いましょう。ある映画にジャンルと人気度で最も近い10個の映画を用いて、評点の予測を行いましょう。
# 
# 最初に、全てのデータをpandasのDataFrameに読み込みます。

# In[1]:


import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('c:/DataScience/DataScience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3))
ratings.head()


# movie_IDでグループ化しましょう。そして、評点の総数（各映画の人気度）と評点の平均値を計算しましょう。

# In[2]:


import numpy as np

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
movieProperties.head()


# 評点の元のデータは、映画同士の距離を計算するための使い勝手がよくありません。従って、正規化された評点の数値を含む新しいDataFrameを作ります。0は誰も評点していないことを意味し、1はもっとも人気の高い映画の値です。

# In[3]:


movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
movieNormalizedNumRatings.head()


# それでは、u.itemファイルからジャンルの情報を取得しましょう。 19個のフィールドがあり、それぞれが特定のジャンルに対応しています。0はそのジャンルではないことを意味し、1はそのジャンルであることを意味します。各映画は一つ以上のジャンルを持つことがあります。
# 
# 関連する全てのデータは、ディクショナリmovieDictに格納します。各要素は映画の名前、ジャンルのリスト、正規化された人気度のスコア、平均の評点を含みます。

# In[4]:


movieDict = {}
with open(r'c:/DataScience/DataScience/ml-100k/u.item') as f:
    temp = ''
    for line in f:
        #line.decode("ISO-8859-1")
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))


# 例えば、movie IDが1の, "Toy Story"があります。

# In[5]:


print(movieDict[1])


# それでは、ジャンルがどれだけ似ているか、人気度がどれだけ似ているかで”距離”を計算するための関数を定義しましょう。この関数が機能することを確認するために、IDが2の映画とIDが4の映画の距離を計算してみましょう。

# In[7]:


from scipy import spatial

def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance
    
ComputeDistance(movieDict[2], movieDict[4])


# 距離が大きいほど、映画が似ていないことになります。2と4の映画の内容を見て、同じ点や相違点を確かめてみましょう。

# In[8]:


print(movieDict[2])
print(movieDict[4])


# ある映画（この場合はトイストーリー）と、データセット内の他の全ての映画との距離を計算します。そして、距離でソートし、K個の最も近い映画を取得します。

# In[9]:


import operator

def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

K = 10
avgRating = 0
neighbors = getNeighbors(1, K)
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print (movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))
    
avgRating /= K


# 10個の最近傍の映画の平均評点は以下の通りです。

# In[10]:


avgRating


# これは、トイストーリーの実際の平均評点と比較していかがでしょうか？

# In[11]:


movieDict[1]


# いい感じですね。
# 

# ## アクティビティ

# Kに10を設定しましたが、10というのは適当な値です。Kの値を変更すると、結果にどのような影響があるでしょうか？
# 
# 距離を求めるロジックも適当なものです。ロジックを改善することは可能でしょうか？

# In[ ]:




