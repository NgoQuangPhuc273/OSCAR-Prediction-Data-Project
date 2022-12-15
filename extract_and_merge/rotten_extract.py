from rotten_tomatoes_scraper.rt_scraper import MovieScraper
import pandas as pd
import numpy as np
import re
from time import sleep
from datetime import datetime


def get_RT_ratings(movie_title):

    # Initialize
    RT_search = MovieScraper()
    search_res = RT_search.search(movie_title)

    # Matching entities
    url_list = [
        movie_dict["url"]
        for movie_dict in search_res["movies"]
        if movie_dict["name"].lower() == movie_title.lower()
    ]
    if len(url_list) == 1:
        url = url_list[0]

    # No exact name for the movie:
    elif not url_list:
        url_list = sorted(
            [
                (movie_dict["url"], movie_dict["year"])
                for movie_dict in search_res["movies"]
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        url = url_list[0][0]
        print(f"No exact match found. Going with {url}")

    # More than one exact match - return the latest one:
    elif len(url_list) > 1:
        url_list = sorted(
            [
                (movie_dict["url"], movie_dict["year"])
                for movie_dict in search_res["movies"]
                if movie_dict["name"].lower() == movie_title.lower()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        url = url_list[0][0]
        print(f"More than one exact match found. Going with {url}")

    #Running
    movie_scraper = MovieScraper(movie_url="https://www.rottentomatoes.com" + url)
    movie_scraper.extract_metadata()

    rt_critics_score = int(movie_scraper.metadata["Score_Rotten"])
    rt_audience_score = int(movie_scraper.metadata["Score_Audience"])
    print(movie_title)
    print(" Audience Score: ", rt_audience_score,"%\n ","Critics Score: ", rt_critics_score, "%")
    rotten_score = pd.DataFrame()

    return movie_title, rt_critics_score, rt_audience_score

def main():
    movie_list = []
    title_list = []
    rt_critics_score_list = []
    rt_audience_score_list = []
    
    print("Extracting Rotten Tomatoes Data...")
    for movie in movie_list:
        movie_title, rt_critics_score, rt_audience_score = get_RT_ratings(movie)
        
        title_list.append(movie_title)
        rt_critics_score_list.append(rt_critics_score)
        rt_audience_score_list.append(rt_audience_score)
# main()