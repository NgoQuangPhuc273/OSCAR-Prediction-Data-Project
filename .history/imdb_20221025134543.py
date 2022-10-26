# from rotten_tomatoes_scraper.rt_scraper import MovieScraper
# from imdb import IMDb
import imdb
# import re
# from time import sleep
# from random import randint
# from datetime import datetime
# import matplotlib
# import seaborn


# creating an instance of the IMDB()
ia = imdb.IMDB()
# Using the Search movie method
items = ia.search_movie('Data')

for i in items:
    print(i)

