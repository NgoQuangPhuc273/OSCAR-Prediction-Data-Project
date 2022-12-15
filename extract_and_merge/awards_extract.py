from bs4 import BeautifulSoup
import requests as rq
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
import traceback
pd.options.display.max_columns = None

def academy():
    oscar_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture').text, 'lxml')

    oscar_results = []
    current_year = 1
    for table in oscar_soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) == 1:
                current_year = int(re.search('[\d]{4}', columns[0].text).group(0))
            elif len(columns) == 2:
                film_col = columns[0]
                if row.get('style') == 'background:#FAEB86':
                    winner = 1
                else:
                    winner = 0
                try:
                    a = film_col.find('i').find('a')
                    oscar_results.append((current_year, a.get_text('title'), winner))
                except:
                    print("Extracting Movie Awards Data...")
            else:
                continue

    oscar = pd.DataFrame(oscar_results, columns = ['year','film', 'oscar_winner'])

    oscar.to_csv('csv/oscar.csv', index=False)

def dga():
    dga_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/Directors_Guild_of_America_Award_for_Outstanding_Directing_%E2%80%93_Feature_Film').text, 'lxml')

    dga_results = []
    current_year = 1
    for table in dga_soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) == 4:
                current_year = int(re.search('[\d]{4}', columns[0].text).group(0))
                film_col = columns[2]
            else:
                film_col = columns[1]
            if columns[1].get('style') == 'background:#FAEB86;':
                winner = 1
            else:
                winner = 0
            try:
                a = film_col.find('i').find('a')
                dga_results.append((current_year, a.get('title'), winner))
            except:
                # print(f"Problem with {row}")
                continue
                traceback.print_exc()

    pd.DataFrame(dga_results, columns = ['year','film','winner']).to_csv('csv/DGA.csv', index = False)

def bafta():
    bafta_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/BAFTA_Award_for_Best_Film').text, 'lxml')

    bafta_results = []
    current_year = 1
    for table in bafta_soup.find_all('table', {'class': 'wikitable'})[2:]:
        year = 1947
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) == 1:
                if current_year != 2019: 
                    # 2019 throws an error because a winner has not been picked as of 2/1
                    current_year = int(re.search('[\d]{4}', columns[0].text).group(0))
                continue
            elif len(columns) == 5:
                film_col = columns[1]
            elif len(columns) == 4:
                film_col = columns[0]
            else:
                # print(f"Wrong number of columns in {row}")
                continue
                
            winner = film_col.get('style') == 'background:#ccc;'
            if winner == True:
                winner = 1
            else:
                winner = 0
            try:
                a = film_col.find('a')
                bafta_results.append((current_year, a.get('title'), winner))
            except:
                # print(f"Problem with {row}")
                continue
                traceback.print_exc()
    pd.DataFrame(bafta_results, columns = ['year','film','winner']).to_csv('csv/BAFTA.csv', index = False)

def pga():
    pga_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/Producers_Guild_of_America_Award_for_Best_Theatrical_Motion_Picture').text, 'lxml')

    pga_results = []
    current_year = 1
    for table in pga_soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) == 4:
                current_year = int(re.search('[\d]{4}', columns[0].text).group(0))
                film_col = columns[1]
            else:
                film_col = columns[0]
            if columns[1].get('style') == 'background:#FAEB86;':
                winner = 1
            else:
                winner = 0
            try:
                if film_col.find('i') is not None:
                    a = film_col.find('i').find('a')
                    pga_results.append((current_year, a.get('title'), winner))
            except:
                traceback.print_exc()

    pd.DataFrame(pga_results, columns = ['year','film','winner']).to_csv('csv/PGA.csv', index = False)

def globes():
    globes_drama_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Motion_Picture_%E2%80%93_Drama').text, 'lxml')
    globes_comedy_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/Golden_Globe_Award_for_Best_Motion_Picture_%E2%80%93_Musical_or_Comedy').text, 'lxml')
    
    globes_drama_results = []
    globe_comedy_results = []
    current_year = 1

    for table in globes_drama_soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) == 4:
                current_year = columns[0].text.split('[')[0]
                film_col = columns[1]
            else:
                film_col = columns[0]
            if columns[1].get('style'):
                winner = 1
            else:
                winner = 0
            a = film_col.find('i').find('a')
            globes_drama_results.append((current_year, a.get('title'), winner))

    for table in globes_comedy_soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) == 4:
                current_year = columns[0].text.split('[')[0]
                film_col = columns[1]
            else:
                film_col = columns[0]
            if columns[1].get('style'):
                winner = 1
            else:
                winner = 0
            try:
                if film_col.find('i') is not None:
                    a = film_col.find('i').find('a')
                    globe_comedy_results.append((current_year, a.get('title'), winner))
            except:
                traceback.print_exc()
                
    pd.DataFrame(globes_drama_results, columns = ['year','film','winner']).to_csv('csv/GG_drama.csv', index = False)
    pd.DataFrame(globe_comedy_results, columns = ['year','film','winner']).to_csv('csv/GG_comedy.csv', index = False)


def cannes():
    GPalm_soup = BeautifulSoup(rq.get('https://en.wikipedia.org/wiki/Palme_d%27Or').text, 'lxml')
    GPalm_elements = GPalm_soup.find('div', {'id': 'Palme_d&#039;Or_winning_films'}).findNext('ul').find_all('li')

    GPalm_elements = GPalm_soup.find('div', {'id': 'Palme_d&#039;Or_winning_films'}).findNext('ul').find_all('li')
    winners = dict()
    for wel in GPalm_elements:
        year = int(re.search('[\d]{4}', wel.text).group(0))
        a = wel.find('a')
        href = a.get('href')
        title = a.get('title')
        winners[href] = (year, title)
    table_years = set([1991, 1993, 1994] + list(range(2007, 2020)))
    GPalm_results = []
    for year in range(1970, 2020):
        soup = BeautifulSoup(rq.get(f'https://en.wikipedia.org/wiki/{year}_Cannes_Film_Festival').text, 'lxml')
        tag = next(x for x in soup.find_all('span', {'class': 'mw-headline'}) if x.text.lower().startswith('in competition'))
        if not tag:
            raise
        if year in table_years:
            elements = tag.findNext('tbody').find_all('tr')[1:]
        else:
            elements = tag.findNext('ul').find_all('li')
        for el in elements:
            a = el.findNext('a')
            winner = href in winners
            GPalm_results.append((year, title, winner))
    GPalm = pd.DataFrame(GPalm_results, columns = ['year','film','winner'])

    GPalm['winner'] = GPalm['winner']*1
    GPalm.to_csv('csv/GPalm.csv', index = False)

def ccma():
    CCMA_result=list()

    for i in range(1999,2020):
        
        url="https://www.filmaffinity.com/en/awards.php?award_id=critics_choice_awards&year=%s" %(i)
        page_soup = BeautifulSoup(rq.get(url).content,'lxml')
        
        for tag in page_soup.find_all('div',class_='aw-mc2 winner-border'):
            movie=tag.find('a').get('title')
            winner=1
            year=url[-4:]
            CCMA_result.append((movie, winner, year))
              
        for tag in page_soup.find_all('div',class_="aw-mc2"):
            for i in tag.find_all('a'):
                movie=i.get('title')
                winner=0
                year=url[-4:]
                CCMA_result.append((year, movie, winner))
                
    fields = ['year','film','winner']
    CCMA = pd.DataFrame(CCMA_result,columns=fields).to_csv('csv/CCMA.csv', index=False)

def main():
    academy()
    dga()
    bafta()
    pga()
    ccma()

# main()
