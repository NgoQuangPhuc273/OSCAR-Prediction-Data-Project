from bs4 import BeautifulSoup
import requests as rq
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
pd.options.display.max_columns = None

def get_url_data(url):
    
    response = rq.get(url)
    html_soup = BeautifulSoup(response.content,"lxml")

    movie_containers = html_soup.find_all('div', {'class':'lister-item mode-advanced'})
    
    movie_dict = get_data(movie_containers)
    imdb_1 = pd.DataFrame(movie_dict)
    imdb_2 = imdb_1

    return imdb_2

def get_data(movie_containers):
    
    movie_dict = {'Movie':[], 'Year':[], 'Runtime':[], 'Certificate':[], 'Genre':[], 
                  'IMDb_rating':[], 'IMDb_votes':[], 'Metascore':[], 'Directors':[], 'Actors':[]}
    
    for first_movie in movie_containers:
        
        movie_dict['Movie'].append(first_movie.find('h3').find('a')
                                   .get_text() if first_movie.find('h3').find('a') else None)
        
        movie_dict['Year'].append(first_movie.find('h3')
                                  .find('span', {'class':'lister-item-year text-muted unbold'})
                                  .get_text() if first_movie.find('h3')
                                  .find('span', {'class':'lister-item-year text-muted unbold'}) else None)
        
        movie_dict['Runtime'].append(first_movie.find('p')
                                     .find('span', class_='runtime').get_text() if first_movie.find('p')
                                     .find('span', class_='runtime') else None)
        
        if first_movie.find('p').find('span', {'class':'certificate'}) is not None:
            movie_dict['Certificate'].append(first_movie.find('p').find('span', {'class':'certificate'}).get_text())
        else:
            movie_dict['Certificate'].append(None)
        
        movie_dict['Genre'].append(first_movie.find('p')
                                   .find('span', {'class':'genre'}).get_text()
                                   .split() if first_movie.find('p').find('span', {'class':'genre'}) else None)
        
        movie_dict['IMDb_rating'].append(first_movie
                                         .find('div', {'class':'inline-block ratings-imdb-rating'})
                                         .find('strong').get_text() if first_movie
                                         .find('div', {'class':'inline-block ratings-imdb-rating'}) else None)
        
        movie_dict['IMDb_votes'].append(first_movie.find_all('span', {'name':'nv'})[0]
                                        .get('data-value') if first_movie.find_all('span', {'name':'nv'}) else None)
        
        if first_movie.find('div', class_ = 'ratings-metascore') is not None:
            movie_dict['Metascore'].append(first_movie.find('div', {'class':'inline-block ratings-metascore'}).find('span').get_text().strip())
        else:
            movie_dict['Metascore'].append(None)
        
        directors = []
        actors=[]
        state = False
            
        for i in first_movie.find('p',class_='').find_all('a'):
            if state == False:
                directors.append(i.get_text('href'))
            else:
                actors.append(i.get_text('href'))
                        
            if i.next_sibling.next_sibling != None:
                if i.next_sibling.next_sibling.get('class') != None and len(i.next_sibling.next_sibling.get('class')) > 0:
                    state = i.next_sibling.next_sibling.get("class")[0] =='ghost'          
        
        movie_dict['Directors'].append(directors)
        movie_dict['Actors'].append(actors)

    return movie_dict

def main():
    header = ['Movie', 'Year', 'Runtime', 'Certificate', 'Genre', 'IMDb_rating',
                    'IMDb_votes', 'Metascore', 'Directors', 'Actors']
    imdb = pd.DataFrame(columns=header)

    for year in range(1999,2020):
        url = ("https://www.imdb.com/search/title/?title_type=feature&release_date=%s&sort=num_votes,desc&count=100" %year)
        print("Getting best movies data in", year)
        imdb_1 = get_url_data(url)
        imdb = imdb.append(imdb_1, ignore_index=True)

    imdb['Movie'] = imdb['Movie'].replace({'Gisaengchung': 'Parasite', 
                                        'Once Upon a Time... in Hollywood': 'Once Upon a Time in Hollywood'})

    temp = imdb.copy()
    imdb = imdb.join(temp.Genre.apply(pd.Series).add_prefix('Genre_'))

    imdb['Year'] = imdb['Year'].astype(str).str.replace(r'\D+', '').astype(int)

    #CLEANING DATA

    # change runtime to runtime (min)
    imdb['Runtime'] = imdb['Runtime'].astype(str).str.replace('min', '').astype(int)
    imdb = imdb.rename(columns={'Runtime': 'Runtime (min)'})

    # change rating to float
    imdb['IMDb_rating'] = imdb['IMDb_rating'].fillna(0).astype(float)

    # change metascore to int
    imdb['Metascore'] = imdb['Metascore'].fillna(0).astype(int)

    # change imdb_votes to int
    imdb['IMDb_votes'] = imdb['IMDb_votes'].str.replace(',', '').fillna(0).astype(int)

    # remove list from director and join
    imdb['Directors'] = imdb['Directors'].apply(lambda x: " ".join(x))

    # remove ',' commas from genre
    imdb['Genre_0'] = imdb['Genre_0'].str.replace(',', '')
    imdb['Genre_1'] = imdb['Genre_1'].str.replace(',', '')
    imdb['Genre_2'] = imdb['Genre_2'].str.replace(',', '')

    imdb.drop('Genre', axis=1, inplace=True)
    imdb['Genre'] = imdb['Genre_0'].fillna('') + "," + imdb['Genre_1'].fillna('') + "," + imdb['Genre_2'].fillna('')


    dummy_genre = imdb.Genre.str.get_dummies(',')

    # merge dummies
    imdb = imdb.merge(dummy_genre, how = 'inner', left_index=True, right_index = True)

    # merge music and musical

    imdb.loc[imdb['Music'] == 1, 'Musical'] = imdb['Music']
    imdb.drop(columns='Music', inplace=True)
    imdb.drop(['Genre_0', 'Genre_1', 'Genre_2'], axis=1, inplace=True)

    print(imdb)
    imdb.to_csv("csv/imdb.csv", index=False)

# main()