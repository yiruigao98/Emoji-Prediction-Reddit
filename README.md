# Emoji-Prediction-Reddit

Files Descriptions:

Data manipulation:

* dataset2db.py: get the reddit data from great lake server and transform the data from bz2 to databases. 
* db2csv.py: Extract subreddit id, author, create time and content as the main features and store into csv files. "run_personal" function utilizes the personality-related and karma related functions and create new csv files with the addition of karmasn and personalities.
* emoji_list.csv: a csv file containing all the emojis scrapped online.

Preprocessing:

* data_preprocessing.py: preprocess the data by removing stopwords, stemming and lemmatizing and normalizing. Do pre steps for word2vec and LDA constructions.

Feature engineering:

* feature_importance.py: show importance of features.
* inspection.py: allow users to inspect the features of the data frame, show top topics after LDA operations.

* file_combine.py: combine monthly csv files to one single csv file for the whole year's data.

Modeling:

* models.py: train shallow ML models and make predictions.
* train.py: train word2vec and LDA models.
* nn.py: neural network model.

User-subreddit information:

* user_info.py: get a user's personality and karma and subreddit info.
* subreddit_info.py: get user number given a subreddit id.
* MBPT.py: a model to train and get users' personalities.

* main.py: put everything together.


