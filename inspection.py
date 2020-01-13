import pandas as pd
import numpy as np
from collections import Counter

# Write a function to allow feature inspection:
def feature_inspect(df):
    print("The shape of the dataframe is: ")
    print(df.shape)
    print("Display the top 5 rows: ")
    print(df.head(5))
    # Check null value:
    print(df.isnull().sum())
    # Check the number of any column:
    print(df.Emoji.value_counts().index)
    # Check text info:
    content_lengths = np.array(list(map(len, df.Content.str.split(' '))))
    print("The average number of words in a discussion is: {}.".format(np.mean(content_lengths)))
    print("The minimum number of words in a discussion is: {}.".format(min(content_lengths)))
    print("The maximum number of words in a discussion is: {}.".format(max(content_lengths)))
    # Find out how many rows that have only one emoji as the content:
    print("There are {} discussions with over one emoji as the sole content.".format(sum(df.Content == df.Emoji)))
    # Look at the emoji that is mostly and leastly used:
    print("The mostly used emoji is\n {}".format(df.Emoji.value_counts()[df.Emoji.value_counts() == df.Emoji.value_counts().max()]))
    print("The leastly used emoji is\n {}".format(df.Emoji.value_counts()[df.Emoji.value_counts() == df.Emoji.value_counts().min()]))



# See top word frequencies:
def top_word(df):
    # Test the top words after doing LDA preprocessing:
    tokenized_only_dict = Counter(np.concatenate(df.Tokenized_Content.values))
    tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
    tokenized_only_df.rename(columns={0: 'count'}, inplace=True)
    tokenized_only_df.sort_values('count', ascending=False, inplace=True)
    print(tokenized_only_df)


# See topics' top words:
def get_top_topic(model, topic_id, num_word = 10):
    id_tuples = model.get_topic_terms(topic_id, topn = num_word)
    word_ids = np.array(id_tuples)[:, 0]
    words = map(lambda id_: model.id2word[id_], word_ids)
    return words

def LDA_example_top_topic(df, model):
    # Example:
    ex_distribution = df.loc[df.Emoji == '™', 'LDA_features'].mean()
    print("LDA: Looking up top words from top topics from %s"%'™')
    for x in sorted(np.argsort(ex_distribution)[-5:]):
        top_words = get_top_topic(model, x)
        print("For topic {}, the top words are: {}".format(x, ','.join(top_words)))
    ex2_distribution = df.loc[df.Emoji == '♥', 'LDA_features'].mean()
    print("LDA: Looking up top words from top topics from %s"%'♥')
    for x in sorted(np.argsort(ex2_distribution)[-5:]):
        top_words = get_top_topic(model, x)
        print("For topic {}, the top words are: {}".format(x, ','.join(top_words)))




        