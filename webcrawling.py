# 기업 ESG 경영 정보 뉴스크롤링 정보 추출(출처 NAVER)
from selenium import webdriver
from openai import OpenAI
# 네이버 뉴스(https://search.naver.com/search.naver?ssc=tab.news.all&where=news&sm=tab_jum&query=ESG)
import pandas as pd
import time
from webdriver_manager.chrome import ChromeDriverManager

# 기업 ESG 경영 정보 뉴스크롤링 정보 추출(출처 NAVER)
# 크롬드라이버 실행
chrome_option = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=chrome_option) # 잘못된 코드 그래서 고쳐야 됨!

#크롬 드라이버에 url 주소 네이버 넣고 실행
driver.get('https://www.naver.com/')

# 페이지가 완전히 로딩되도록 3초동안 기다림
time.sleep(3)

# 검색어 타이핑하여서 이를 검색키 엔터 누르기
search_box = driver.find_element_by_xpath('//*[@id="query"]')
search_box.send_keys('ESG')
search_box.send_keys(webdriver.common.keys.Keys.ENTER)

# 뉴스검색란 클릭하기
news_box = driver.find_element_by_xpath('//*[@id="lnb"]/div[1]/div/div[1]/div/div[1]/div[1]/a')
news_box.click()

# 뉴스 제목링크 클릭하기 
# 뉴스 제목, 내용 추출

# Dataframe에 기사, 제목, 내용 저장하기

