from bs4 import BeautifulSoup
import requests as rq
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
pd.options.display.max_columns = None
from time import sleep
from random import randint


header = ['release_date', 'movie_name', 'budget', 'domestic_gross',
             'world_wide_gross']

box_office = pd.DataFrame(columns=header)

def get_budget_gross(list_tr):
    budget_dict = {'release_date':[], 'movie_name':[], 'budget':[], 
                       'domestic_gross':[], 'world_wide_gross':[]}

    for tr in list_tr:
        try:
            x_list = tr.find_all('td', {'class':'data'})
            budget_dict['release_date'].append(tr.find('a').get_text()) 
            # movie name 
            budget_dict['movie_name'].append(tr.find('b').find('a').get_text()) 
            # budget 
            budget_dict['budget'].append(x_list[1].get_text())  
            # domestic
            budget_dict['domestic_gross'].append(x_list[2].get_text())  
            # international
            budget_dict['world_wide_gross'].append(x_list[3].get_text())  

        except:
            continue
    
    # print(budget_dict)
    return budget_dict

def main():
    print("Extracting Box Office Data...")
    for i in range(1, 5700, 100):
        url = (f"https://www.the-numbers.com/movie/budgets/all/{i}")
        response = rq.get(url)
        html_soup = BeautifulSoup(response.content,"lxml")

# main()