#!/usr/bin/env python
# coding: utf-8

# # アイテムベース協調フィルタリング

# 以前と同じように、MovieLensの100KデータセットをpandasのDataFameに入れるところから始めます。 

# In[1]:


import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('e:/sundog-consult/udemy/datascience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('e:/sundog-consult/udemy/datascience/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

ratings.head()


# pivot_table関数によりユーザーとユーザーが評価した映画のマトリックスを作ります。NaNはデータが無いか、ユーザーが鑑賞していないことを意味します。

# In[2]:


userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()


# pandasのcorr()関数で、マトリックスの全ての映画のペアの相関スコアを計算できます。少なくとも一人のユーザーが両者の映画を評価している必要があり、そうでないとNaNが表示されます。

# In[3]:


corrMatrix = userRatings.corr()
corrMatrix.head()


# しかしながら、少数のユーザーによって評価された映画のペアによって引き起こされるおかしな結果を避ける必要があります。結果を多くの人に鑑賞された映画に絞るために、min_periodsの引数を設定して評価数が100未満の映画のペアは除くようにします。

# In[4]:


corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()


# ユーザーID 0のユーザーに、映画のレコメンドを行ってみましょう。このユーザーはテストケースとして追加した架空のユーザーで、スターウォーズと帝国の逆襲が大好きです。しかしながら、風と共に去りぬが大嫌いです。userRatingsのDataFrameから彼の評価を取り出し、dropna()を用いて存在しないデータを取り除きます。これにより、実際に評価を行ったデータのみが残ります。

# In[5]:


myRatings = userRatings.loc[0].dropna()
myRatings


# それでは、このユーザーが評価を行った映画をもとに、レコメンドを作ります。
# 
# ユーザーが評点を行った各映画に関して、相関マトリックスから映画の類似度のリストを取り出します。そして、それらの類似度とユーザーの評点を掛け合わせてお勧め度とし、候補に加えます。これにより、ユーザーが高く評価した映画に似た映画のお勧め度は高くなります。

# In[9]:


simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))


# かなりいい感じになってきましたね。いくつかの映画が、二回以上見えます。これは、ユーザーが評点した映画が二つ以上の映画と似ていたからです。groupby()関数を用いて、同じ映画のスコアを足し合わせます。

# In[10]:


simCandidates = simCandidates.groupby(simCandidates.index).sum()


# In[11]:


simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)


# 最後にすべきことは、既に評価した映画を除くことです。以前に観た映画をお勧めすることは望ましくありません。

# In[12]:


filteredSims = simCandidates.drop(myRatings.index)
filteredSims.head(10)


# これで完成です。

# ## 演習

# 今回の結果をさらに良くすることは可能でしょうか？異なる関数を用いたり、min_periodsの値を変えることはさらに興味深い結果をもたらすかもしれません。
# 
# 先ほどのユーザーが嫌いであった、風と共に去りぬに似た映画が最後のリストに残っています。ユーザーが嫌いな映画は、評点を下げる代わりにペナルティを与えるべきなのでしょうか？
# 
# 評価のデータセットには外れ値があるかもしれません。何人かのユーザーは、極端に多くの映画を評点して結果のバランスを崩しているかもしれません。以前のレクチャーに戻って、外れ値を検出する方法を復習し、外れ値の除去による改善が期待できるか検討してみましょう。
# 
# これまで学習してきた様々な手法を踏まえて、より良いレコメンドシステムの構築にはまだまだ議論の余地があるかと思います。

# In[ ]:




