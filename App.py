from PIL import Image
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import pipeline

url = "https://news.yahoo.co.jp/topics/top-picks"

res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')

target_elements = soup.find_all("a", class_="newsFeed_item_link")
news_list = []
for element in target_elements:
  #本文の取得
  response = requests.get(element['href'])
  data = response.text
  soup = BeautifulSoup(data, "html.parser")
  content_links = soup.select('#uamods-pickup > div:nth-of-type(2) > div > p > a') #記事全文を読む
  content_response = requests.get(content_links[0]["href"])
  content_data = content_response.text
  content_soup = BeautifulSoup(content_data, "html.parser")
  content_element = content_soup.find(class_="article_body highLightSearchTarget")
  content_text = content_element.get_text()

  text2text_pipeline = pipeline(
      model="Zolyer/ja-t5-base-summary"
  ) 

  news = {
      'title': element.find(class_="newsFeed_item_title").text,
      'url': element['href'],
      'content': content_text,
      'summary': text2text_pipeline(content_text)[0]["generated_text"]

  }
  news_list.append(news)


#WEBページに表示する内容を記載
image = Image.open('新聞の無料アイコン素材 6 .jpeg')
st.set_page_config(
    page_title="ニュース要約アプリ", 
    page_icon=image, 
    layout="wide", 
    initial_sidebar_state="auto", 
    menu_items={
         'About': """
         Yahooニュースのトップページの最初に表示されるニュースを、T5のファインチューニング済みのモデルを使用して要約しています。
         読み込みが遅い場合は、しばらくお待ちください。
         """
     })

# 現在の日付を取得
current_date = datetime.now()
# 日付を指定した形式で表示
formatted_date = current_date.strftime('%Y/%m/%d')

st.title("ニュース要約")
st.subheader("概要")
st.write("Yahooニュースのトップページからニュースを取得し、T5による要約を行うアプリです。")
st.subheader(f"今日のニュースと要約({formatted_date})")

for i, news in enumerate(news_list):
    st.markdown(f"[{news['title']}]({news['url']})")
    st.markdown(f"{news['summary']}")



