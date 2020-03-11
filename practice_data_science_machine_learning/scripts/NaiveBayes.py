#!/usr/bin/env python
# coding: utf-8

# # 単純ベイズ

# sklearn.naive_bayesを用いて、スパム分類器を簡単に作りましょう。大部分は、訓練用のデータをpandasのDataFrameに入れるコードです。

# In[1]:


import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('e:/sundog-consult/Udemy/DataScience/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('e:/sundog-consult/Udemy/DataScience/emails/ham', 'ham'))


# DataFrameを見てみましょう。

# In[2]:


data.head()


# CountVectorizerを用いて、各メッセージを各単語数を表すベクトルに変換します。これを各メッセージに対応するclassのデータと共にMultinomialNB分類器に入れて、fit関数により訓練を行います。 これにより、訓練済みのスパムフィルターを得ることができます。

# In[3]:


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)


# それでは、適当なメッセージを分類してみましょう！

# In[4]:


examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions


# ## アクティビティ

# 今回はデータが少ないため、スパム分類器の精度はあまりよくありませんが、examplesに他のメッセージを追加してみて、期待するような結果が得られるかどうか試してみましょう。
# もしやる気があるのであれば、訓練とテストをこのスパム分類器で試してみましょう。スパム分類器の精度が数値化されるでしょう。

# In[ ]:




