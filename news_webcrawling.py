# 기업 ESG 경영 정보 뉴스크롤링 정보 추출(출처 NAVER)
from selenium import webdriver
from bs4 import BeautifulSoup
from openai import OpenAI
# 네이버 뉴스(https://search.naver.com/search.naver?ssc=tab.news.all&where=news&sm=tab_jum&query=ESG)
import pandas as pd
import time
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np

# 기업 ESG 경영 정보 뉴스크롤링 정보 추출(출처 NAVER)
# 크롬드라이버 실행
chrome_option = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=chrome_option) # 잘못된 코드 그래서 고쳐야 됨!
driver.maximize_window()

#크롬 드라이버에 url 주소 네이버 넣고 실행
driver.get('https://www.naver.com/')

# 페이지가 완전히 로딩되도록 3초동안 기다림
time.sleep(3)

# 검색어 타이핑하여서 이를 검색키 엔터 누르기
search_box = driver.find_element(webdriver.common.by.By.XPATH,'/html/body/div[2]/div[1]/div/div[3]/div/div/form/fieldset/div/input')
search_box.send_keys('ESG')
search_box.send_keys(webdriver.common.keys.Keys.ENTER)

# 뉴스검색란 클릭하기
news_box = driver.find_element(webdriver.common.by.By.XPATH,'//*[@id="lnb"]/div[1]/div/div[1]/div/div[1]/div[1]/a')
news_box.click()

# 필터하기(2023~2024년)
option_box = driver.find_element(webdriver.common.by.By.XPATH, '/html/body/div[3]/div[2]/div/div[1]/div[1]/div[1]/div/div[2]/a')
option_box.click()
# 1시간 필터로 해보기(직접 실험)
one_hour_box1 = driver.find_element(webdriver.common.by.By.XPATH, '/html/body/div[3]/div[2]/div/div[1]/div[1]/div[2]/ul/li[3]/div/div[1]/a[2]')
one_hour_box1.click()
one_hour_box2 = driver.find_element(webdriver.common.by.By.XPATH, '/html/body/div[3]/div[2]/div/div[1]/div[1]/div[2]/ul/li[3]/div/div[2]/div/div/div/div/div/ul/li[1]/a')
one_hour_box2.click()
# 2023년 필터로 해보기
# box1_2023 = driver.find_element(webdriver.common.by.By.XPATH, '/html/body/div[3]/div[2]/div/div[1]/div[1]/div[2]/ul/li[3]/div/div[1]/a[8]')
# box1_2023.click()
# 끝까지 최대로 스크롤하기

# 스크롤 높이 가져옴
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # 끝까지 스크롤 내리기
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 대기
    time.sleep(3)

    # 스크롤 내린 후 스크롤 높이 다시 가져옴
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# beautifulsoup 이용하여서 소스 추출하기
source_soup = BeautifulSoup(driver.page_source, 'html.parser')
li_title_list = source_soup.select('div.news_area > div.news_contents > a.news_tit')
title_list = []
for i in li_title_list:
    if i == None:
        title_list.append('')
    if i.attrs['title'] == None:
        title_list.append('')
    title_list.append(i.attrs['title'])    
    
print(title_list)
