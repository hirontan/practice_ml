#!/usr/bin/env python
# coding: utf-8

# # データクリーニング

# Webのアクセスログを取って、最も見られたページを特定しましょう。
# 
# Apacheのアクセスログラインを解析するために、正規表現のセットアップをしましょう。

# In[1]:


import re

format_pat= re.compile(
    r"(?P<host>[\d\.]+)\s"
    r"(?P<identity>\S*)\s"
    r"(?P<user>\S*)\s"
    r"\[(?P<time>.*?)\]\s"
    r'"(?P<request>.*?)"\s'
    r"(?P<status>\d+)\s"
    r"(?P<bytes>\S*)\s"
    r'"(?P<referer>.*?)"\s'
    r'"(?P<user_agent>.*?)"\s*'
)


# 解析すべきログファイルのパスです。DataScienceフォルダの場所に合わせて変更しましょう。

# In[2]:


logPath = "c:/DataScience/DataScience/access_log.txt"


# 各アクセスからURLを取り出すためのコードを用意します。そして、ディクショナリを用いて各URLの出現回数をカウントします。そして、ソートしてTop20のページを表示します。

# In[3]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            (action, URL, protocol) = request.split()
            if URLCounts.has_key(URL):
                URLCounts[URL] = URLCounts[URL] + 1
            else:
                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# 何か間違っているのでしょうか。'request'の内容は以下のようになっているはずです。
# 
# GET /blog/ HTTP/1.1
# 
# HTTPアクションと、URLと、プロトコルがあるはずです。しかしながら、常にこのようになっていないようです。3つのアイテムを含まないリクエストを表示してみましょう。

# In[4]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) != 3):
                print(fields)


# 空のフィールドに加えて、不要なデータが含まれているようです。これらをチェックするコードに修正してみましょう。

# In[7]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) == 3):
                URL = fields[1]
                if URL in URLCounts:
                    URLCounts[URL] = URLCounts[URL] + 1
                else:
                    URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# 機能しているようですね。しかしながら、結果はまだ不十分です。欲しいものは、ウェブサイトの人がアクセスしたページです。xmlrpc.php?って何でしょうか？ログには以下のようなものが含まれているようです。 
# 
# 46.166.139.20 - - [05/Dec/2015:05:19:35 +0000] "POST /xmlrpc.php HTTP/1.0" 200 370 "-" "Mozilla/4.0 (compatible: MSIE 7.0; Windows NT 6.0)"
# 
# GETアクションのみを処理しているわけではないようです。POSTは要らないので、フィルタリングしてしまいましょう。

# In[8]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) == 3):
                (action, URL, protocol) = fields
                if (action == 'GET'):
                    if URL in URLCounts:
                        URLCounts[URL] = URLCounts[URL] + 1
                    else:
                        URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# 良くなってますね。しかしながら、これはニュースサイトです。訪問者は本当にニュースの代わりにブログを読んでいるのでしょうか。おかしいですね。典型的な /blog/ entry のログを見てみましょう。
# 
# 54.165.199.171 - - [05/Dec/2015:09:32:05 +0000] "GET /blog/ HTTP/1.0" 200 31670 "-" "-"
# 
# なぜユーザーエージェントが空なのでしょうか。ある種の悪意のあるスクレイパーか何かでしょう。どんなユーザーエージェントがいるのか明らかにしてみましょう。

# In[10]:


UserAgents = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if agent in UserAgents:
                UserAgents[agent] = UserAgents[agent] + 1
            else:
                UserAgents[agent] = 1

results = sorted(UserAgents, key=lambda i: int(UserAgents[i]), reverse=True)

for result in results:
    print(result + ": " + str(UserAgents[result]))


# なんと、'-'に加えて、無数の異なるWebロボットがこのサイトにアクセスし、データを汚染しているようですね。これらを全て排除するのはとても難しいですが、'-'、もしくは"bot"あるいは"spider"を含むもの、W3 Total Cacheを取り除けば大きくデータの汚染を取り除けそうです。

# In[11]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if (not('bot' in agent or 'spider' in agent or 
                    'Bot' in agent or 'Spider' in agent or
                    'W3 Total Cache' in agent or agent =='-')):
                request = access['request']
                fields = request.split()
                if (len(fields) == 3):
                    (action, URL, protocol) = fields
                    if (action == 'GET'):
                        if URL in URLCounts:
                            URLCounts[URL] = URLCounts[URL] + 1
                        else:
                            URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# そして、ウェブページでないものにヒットしてしまうのが新たな問題です。今回はこれらに関心はないので、/で終わらないURLは除いてしまいましょう。（今回のサイトではこの方法でアクセスされます）

# In[13]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if (not('bot' in agent or 'spider' in agent or 
                    'Bot' in agent or 'Spider' in agent or
                    'W3 Total Cache' in agent or agent =='-')):
                request = access['request']
                fields = request.split()
                if (len(fields) == 3):
                    (action, URL, protocol) = fields
                    if (URL.endswith("/")):
                        if (action == 'GET'):
                            if URL in URLCounts:
                                URLCounts[URL] = URLCounts[URL] + 1
                            else:
                                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# 信頼できそうに見えますね。しかしながら、もっと信頼性を上げるためには /feed/ のページは疑うべきですし、ロボットが紛れ込んでいる可能性もまだあります。それでも、Orlandoのニュースと、worldニュース、comicsニュースに人が多く訪れたのは本当のことです。
# 学ぶべきことは、データを知ることです。そして、結論の前に結果に疑問を抱き、結果を再構築してみることが大事です。もし、間違ったデータにに基づいた解析によりビジネスに悪い結果がもたらされたのであれば、大きなトラブルとなります。
# そして、データクリーニングは公正に行いましょう。欲しい結果を得るために、恣意的なデータのフィルタリングを行ってはいけません。

# ## アクティビティ

# 今回の結果はまだ完璧ではありません。"feed"を含むURLは人間によって見られたページではありません。コードを修正し、"/feed"を含むURLを除くようにしましょう。これらのページのログをいくつか抽出し、誰によってページが見られたかを確認するとベターです。

# In[ ]:




