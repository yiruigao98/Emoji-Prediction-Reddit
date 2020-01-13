import csv
import requests
from bs4 import BeautifulSoup

base_url = 'https://unicode.org/emoji/charts/full-emoji-list.html'

emoji_list = []

html = requests.get(base_url).text
td_tags = BeautifulSoup(html, 'html.parser').find_all(class_ = 'chars')

for emoji in td_tags:
    e = emoji.text
    if e != "🤎":
        emoji_list.append(e)

with open("emoji_list.csv", 'w', encoding= "utf-8") as file:
    wr = csv.writer(file)
    wr.writerow(emoji_list)




  