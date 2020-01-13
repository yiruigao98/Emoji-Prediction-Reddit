# import re

# m = re.search('[a-zA-Z]+',"???")
# print(m)
# import nltk
# print(nltk.data.path)

# import praw
# r = praw.Reddit(client_id = "jojUq2jbLlQQnw",
#     client_secret='1PooR7o5_XyJH60X7VJmFrAMX7s',
#     user_agent='/u/yiruigao98')
# sub = r.subreddit('python').hot(limit=5)
# for s in sub:
#     print(s.title)
# dir(thing) # prints out big long list of attributes on thing.
# redditor = r.redditor("chuggo_tuggans").comments.controversial('week')
# for s in redditor:
#     print(s.body)
# print(redditor)
# thing = r.get_subreddit('t5_2qh11')
# dir(thing)
# print(list(r.info(['t5_2qh11']))[0].display_name)

# for s in r.redditor('chuggo_tuggans').comments.top('all'):
#     print(s.body)
# for s in r.redditor('phailhaus').:
#     print(s.selftext)

# print(r.redditor('chuggo_tuggans').comments.top('all'))



# import pandas as pd
# from sklearn import preprocessing

# data = {'score': [234,24,14,27,-74,46,73,-18,59,160]}
# df = pd.DataFrame(data)

# x = df[['score']].values.astype(float)

# # Create a minimum and maximum processor object
# min_max_scaler = preprocessing.MinMaxScaler()

# # Create an object to transform the data to fit minmax processor
# x_scaled = min_max_scaler.fit_transform(x)

# # Run the normalizer on the dataframe
# df_normalized = pd.DataFrame(x_scaled)

# print(df_normalized[0])


import numpy as np
import pandas as pd

x = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
print(x)


l = [1, 2, 3]
p = np.array([1, 3, 4]).reshape((x.shape[0],1))
print(x.shape)
print(p.shape)
y = np.append(x, p, axis = 1)
print(y)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
z = pd.DataFrame(['ni','you','nsd'])
X_2 = z.apply(le.fit_transform)
ohe = OneHotEncoder(sparse=False)
x_ohe = ohe.fit_transform(X_2)
print(x_ohe)


