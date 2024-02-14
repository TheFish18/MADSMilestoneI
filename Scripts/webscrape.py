from selenium import webdriver
import os
import mechanicalsoup

browser = mechanicalsoup.StatefulBrowser()
import requests
from bs4 import BeautifulSoup
import sys
import time
import shutil
import glob
driver = webdriver.Chrome()

def sort_time_key(p, dirname=None):
    if dirname is not None:
        p = os.path.join(dirname, p)
    return os.path.getmtime(p)

def list_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all("a")
    return links

def download_wait_move(url, save_dir, filename):
    download_dir = '/Users/joshfisher/Downloads'
    pre_count = len(os.listdir(download_dir))
    print(url)
    driver.get(url)
    time.sleep(5)
    while pre_count == len(os.listdir(download_dir)):
        time.sleep(0.5)
    time.sleep(2)
    while len(glob.glob(os.path.join(download_dir, '*.crdownload'))) != 0:
        time.sleep(0.5)
    download_paths = glob.glob(download_dir + "/*XPT")
    download_path = sorted(download_paths, key=lambda x: sort_time_key(x, download_dir), reverse=True)[0]
    shutil.move(download_path, os.path.join(save_dir, filename + '.XPT'))

if __name__ == "__main__":
    save_dir = '/Data/NHANES'

    url = 'https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017'
    links = list_links(url)
    years = []
    for link in links:
        if "Demographics" in str(link.string) and 'Cycle' in link.get("href"):
            year = link.get("href").split('&')[1]
            years.append(year)

    root = 'https://wwwn.cdc.gov'
    subdir = '/nchs/nhanes/search/datapage.aspx?'
    demo_subdir = 'Component=Demographics&'
    dietary_subdir = 'Component=Dietary&'
    years = set(years)

    'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&'
    demo_dir = os.path.join(save_dir, "Demographic")
    indiv_dir = os.path.join(save_dir, 'Dietary/IndividualFoods')
    nuts_dir = os.path.join(save_dir, 'Dietary/TotalNutrients')

    for year in years:
        # Demographic URLs
        demo_url = root + subdir + demo_subdir + year
        demo_links = list_links(demo_url)
        for link in demo_links:
            if "XPT" in str(link.string):
                url = link.get('href')
                url = root + url
                download_wait_move(url, demo_dir, year)

    for year in years:
        dietary_url = root + subdir + dietary_subdir + year
        dietary_links = list_links(dietary_url)
        for link in dietary_links:
            s = str(link.string)
            if "XPT" in s:
                if "DR1IFF" in s:
                    url = link.get('href')
                    url = root + url
                    download_wait_move(url, os.path.join(indiv_dir, 'Day1'), year)
                elif "DR2IFF" in s:
                    url = link.get('href')
                    url = root + url
                    download_wait_move(url, os.path.join(indiv_dir, 'Day2'), year)
                if "DR1TOT" in s:
                    url = link.get('href')
                    url = root + url
                    download_wait_move(url, os.path.join(nuts_dir, 'Day1'), year)
                elif "DR2TOT" in s:
                    url = link.get('href')
                    url = root + url
                    download_wait_move(url, os.path.join(nuts_dir, 'Day2'), year)
