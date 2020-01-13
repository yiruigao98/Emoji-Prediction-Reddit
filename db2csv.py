import sqlite3
import pandas as pd
import csv
import re
import praw
from MBPT import *


data = pd.read_csv("BYOtrain.csv")
data_type = data.join(data.apply(lambda row: get_type (row),axis=1))
list_posts, list_personality = preprocessing(data_type)
X_tfitf = vectorize(list_posts)
train_individual(X_tfitf, list_posts, list_personality)


def run(Wfile, DBNAME, emojis):

    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

    results = c.execute("SELECT * FROM Discussion").fetchall()
    row_list = []
    for row in results:
        for emoji in emojis:
            if emoji in row[4]:
                # Test the effect of TM:
                if emoji == "™":
                    continue
                # Eliminate rows with only one emoji as texts:
                if row[4] == emoji or set(row[4]) == emoji:
                    continue
                if re.search('[a-zA-Z]+',row[4]) == None:
                    continue
                row_list.append([str(row[3]), str(row[1]), str(row[2]), str(row[4]).replace('\n',' '), emoji])
                break

    all_df = pd.DataFrame(row_list, columns=('Subreddit_Id', 'Author', 'CreateTime', 'Content','Emoji'))
    conn.close()     
    all_df.to_csv(Wfile, encoding='utf-8', index=False, header= True)



# Add personal info into the csv files:
def run_personal(Wfile, DBNAME, emojis):
    reddit = praw.Reddit(client_id = "jojUq2jbLlQQnw", client_secret='1PooR7o5_XyJH60X7VJmFrAMX7s',
        user_agent='/u/yiruigao98')
    
    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

    results = c.execute("SELECT * FROM Discussion").fetchall()
    row_list = []
    for row in results:
        for emoji in emojis:
            if emoji in row[4]:
                # Test the effect of TM:
                if emoji == "™":
                    continue
                # Eliminate rows with only one emoji as texts:
                if row[4] == emoji or set(row[4]) == emoji:
                    continue
                if re.search('[a-zA-Z]+',row[4]) == None:
                    continue
                try:
                    comment_karma, link_karma = get_karma(str(row[1]), reddit)
                    personality = get_personality(str(row[1]), reddit)
                    encoding_list = translate_personality(personality)
                    row_list.append([str(row[3]), str(row[1]), str(row[2]), str(row[4]).replace('\n',' '), comment_karma, link_karma, encoding_list[0],encoding_list[1],encoding_list[2],encoding_list[3], emoji])
                    print("This user exists!")
                    print(str(row[0]))
                    break
                except:
                    print("This user doesn't exist!")
                    break

    all_df = pd.DataFrame(row_list, columns=('Subreddit_Id', 'Author', 'CreateTime', 'Content','Comment_karma','Link_karma','Personality_IE','Personality_NS','Personality_TF','Personality_PJ','Emoji'))
    conn.close()     
    all_df.to_csv(Wfile, encoding='utf-8', index=False, header= True)\



if __name__ == "__main__":

    Wfile_base = "text_2013-"
    DBNAME_base = "2013/reddit_2013-"

    with open("emoji_list.csv", "r", encoding = "utf-8") as file:
        emojis = list(csv.reader(file, delimiter=','))[0]
    # for emoji in emojis:
    #     print('%s'%emoji)
    # for i in range(12):
    #     if i < 9:
    #         Wfile_path = Wfile_base + '0%s_noTM.csv'%str(i+1)
    #         DBNAME = DBNAME_base + '0%s.db'%str(i+1)
    #     else:
    #         Wfile_path = Wfile_base + '%s_noTM.csv'%str(i+1)
    #         DBNAME = DBNAME_base + '%s.db'%str(i+1)
    #     run(Wfile_path, DBNAME, emojis)

    # With personal info:
    for i in range(10,12):
        if i < 9:
            Wfile_path = Wfile_base + '0%s_noTM_personal.csv'%str(i+1)
            DBNAME = DBNAME_base + '0%s.db'%str(i+1)
        else:
            Wfile_path = Wfile_base + '%s_noTM_personal.csv'%str(i+1)
            DBNAME = DBNAME_base + '%s.db'%str(i+1)
        run_personal(Wfile_path, DBNAME, emojis)   

