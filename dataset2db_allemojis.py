import bz2 
import json
import sqlite3
import csv



def run(file_path, DNAME, Wfile, emojis):
    bz_file = bz2.BZ2File(file_path)
    data = bz_file.readlines()

    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

    c.execute("DROP table if exists 'Discussion'")
    # c.execute("DROP table if exists 'User'")
    conn.commit()

    c.execute('''CREATE TABLE 'Discussion' ( 
        'Id'    INTEGER PRIMARY KEY AUTOINCREMENT, 
        'Author'   TEXT NOT NULL, 
        'CreateTime' TEXT,
        'Subreddit_Id'   TEXT,
        'Content' TEXT
    );''')

    conn.commit()

    for row in data:
        row = json.loads(row.decode('utf-8'))
        comment_string = row['body']
        emoji_list = []
        for emoji in emojis:
            if emoji in comment_string:
                emoji_list.append(emoji)
        if len(set(emoji_list)) > 0 and row['author'] != "[deleted]":
            insertion = (None, row['author'], row['created_utc'], row['subreddit_id'], row['body'])
            statement = "INSERT INTO 'Discussion' VALUES (?,?,?,?,?)"
            c.execute(statement, insertion)

    conn.commit()
    c.execute("SELECT Author, COUNT(Author) as Num FROM Discussion GROUP BY Author ORDER BY Num DESC")
    with open(Wfile, 'a') as w_file:
        for row in c:
            string = "The author with the most comment is %s with %d comments in this month, 2012.\n"%(row[0], row[1])
            print(string)
            w_file.write(string)
            break

    conn.close()


if __name__ == "__main__":
    file_path_base = "/scratch/qmei_root/qmei/shiyansi/reddit_data/2012/RC_2012-"
    DBNAME_base = 'all_emoji_reddit_2012-'
    Wfile = "all_emoji_result.csv"
    with open("emoji_list.csv", "r", encoding = "utf-8") as file:
        emojis = list(csv.reader(file, delimiter=','))[0]

    for i in range(12):
        if i < 9:
            file_path = file_path_base + '0%s.bz2'%str(i+1)
            DBNAME = DBNAME_base + '0%s.db'%str(i+1)
        else:
            file_path = file_path_base + '%s.bz2'%str(i+1)
            DBNAME = DBNAME_base + '%s.db'%str(i+1)
        run(file_path, DBNAME, Wfile, emojis)