from PIL import Image
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime

tokenizer = T5Tokenizer.from_pretrained("Zolyer/ja-t5-base-summary")  
model = T5ForConditionalGeneration.from_pretrained("Zolyer/ja-t5-base-summary")

url = "https://news.yahoo.co.jp"

news_list = []

# APIからニュースデータを取得
response = requests.get(url)
data = response.text

# BeautifulSoupを使ってHTMLを解析
soup = BeautifulSoup(data, "html.parser")


# sc-aiIEM gCShmHの部分を取り出す
target_element = soup.find(class_="sc-aiIEM gCShmH")
target_text = target_element.text
target_elements = soup.find_all(class_="sc-dtLLSn dpehyt")
target_texts = [element.text for element in target_elements]

for element in target_elements:
    # ニュースの本文を得られたURLから取得
    response = requests.get(element['href'])
    data = response.text
    soup = BeautifulSoup(data, "html.parser")
    read_all_url = soup.find(class_="sc-jWuRkY kZUDoE").get('href')
    url_response = requests.get(read_all_url)
    url_data = url_response.text
    url_soup = BeautifulSoup(url_data, "html.parser")
    target_url_content_element = url_soup.find(class_="article_body highLightSearchTarget")
    target_url_content_text = target_url_content_element.get_text()

    inputs = tokenizer.encode("要約: " + target_url_content_text, return_tensors="pt")  
    outputs = model.generate(inputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


    news = {
        'title': element.text,
        'url': element['href'],
        'content': target_url_content_text,
        'summary': summary

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



