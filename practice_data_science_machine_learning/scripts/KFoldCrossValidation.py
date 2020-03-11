#!/usr/bin/env python
# coding: utf-8

# # K分割交差検証
Iris data setを再び用いましょう。
# In[1]:


import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()


# 単一の訓練/テスト用のデータは、cross_validation libraryのtrain_test_split関数で簡単に作ることができます。

# In[2]:


# Split the iris data into train/test data sets with 40% reserved for testing
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# Build an SVC model for predicting iris classifications using training data
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
clf.score(X_test, y_test)   


# K分割交差検証も同様に簡単です。Kに5を設定します。

# In[4]:


# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

# Print the accuracy for each fold:
print(scores)

# And the mean accuracy of all 5 folds:
print(scores.mean())


# いい結果ですね。さらに良くすることは可能でしょうか。異なるカーネル（多項式）を試してみましょう。

# In[5]:


clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print(scores.mean())


# より複雑な多項式のカーネルは、シンプルな線形のカーネルよりも正確性を下げるようです。多項式のカーネルは過剰適合のようです。しかしながら、多項式のカーネルであっても単一の訓練/テストでは過剰適合しないようです。

# In[6]:


# Build an SVC model for predicting iris classifications using training data
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
clf.score(X_test, y_test)   


# 線形のカーネルを用いた単一の訓練/テストと同じスコアですね。

# ## アクティビティ

# SVCにおける多項式のカーネルは多項式の次数を設定できますが、デフォルトでは3です。例えば、以下のようなコードで設定できます。
# svm.SVC(kernel='poly', degree=3, C=1)
# 
# デフォルトの3次の多項式は、上記の結果により過剰適合と考えられます。しかしながら、次数を2にしてみるとどうなるのでしょうか。試してみて、線形のカーネルとも比較してみましょう。

# In[ ]:




