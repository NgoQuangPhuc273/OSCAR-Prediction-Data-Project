# ECON 312: Predict The Academy Award Results Through Machine Learning

Phuc Ngo, Zach Moskowitz, Ryan Blasberg


## I. Introduction:

## II. Description:
In this project, we focus on deeply analyzing the Best Picture category to predict the next Oscar Winners by the distributed percentage of winning.

Our plan is to extract the top 100 movie data ranked according to the number of votes from IMDB for each year from 1999 to 2020. Then, we scrape other websites and match the new data to existed data frame according to the movie title.

IMDB data types:
Movie title, Year, Runtime, Certificate, Genre, IMDB ratings, IMDB number of votes, Metascore, Directors, Actors

Rotten Tomatoes data types:
Critics Rating, Number of reviews

The numbers:
Domestic box office 
International box office

Next, we extract the awards data from Wikipedia. List of awards:
Director Guild Awards (DGA)
British Academy Film Awards (BAFTA)
Producer Guild Awards (PGA)
Golden Globes Awards for Drama (GG_drama)
Golden Globes Awards for Comedy (GG_comedy)
Cannes Film Festival (Golden Palm)
Critics' Choice Movie Awards (CCMA)

And the most important award:
Academy Award - Best Picture

Next, we perform some data cleaning (unformatted names and numbers, general stemming, and lemmatization) and merge different tables generated through data wrangling into one big data set. 

#### Machine Learning Model:
Random Forest is one of the best approaches when dealing with classification. In this project, we primarily concentrate on implementing Random Forest to create a model that can:

- Classify and rank the importance of different metrics (54 attributes)
- Produce the highest accuracy possible
- Successfully predict the future Oscar winner by learning through past candidates.

We will attempt to apply the model to each year, look at the results of them and finally predict the 2022 Oscar winner!


## III. Installation:
You will need to install these python libraries to run the program.

```bash
pip install pandas
pip install bs4
pip install sklearn
pip install matplotlib
pip install seaborn
pip install plotly
```

## IV. Running:
(This is the update for the first implementaion. The final write up will be completely different.)

1. Run extract_main.py to extract data.

2. Run merging.py to merge different data sets into a main one.

3. Run implement_{algorithm}.py (decision tree, random forest or logistic regression) to execute the program.

4. Run winner_prediction.py to see the predicted winners for the next Oscar!


### However, that is when the project is finished!

Initally, we used the alternate IMDB_API (or CinemaGoer) and OMDB to extract data. However, both apis are too inconsistent so we have decided to switch to scrape and extract data ourselves. 

This was a very confusing and complicated at first since since we have to deal with many data types at once. However, the process becomes more and more exciting when we get used to it due to the fact that we can deeply understand how a data flows (or pipelines) can work (more similar the work of data engineers). 

#### At the moment, we have successfully self-extract, scraping, cleaning and merging the movies data from 1990 to 2020 through several websites such as:

https://en.wikipedia.org/ (awards data)

https://www.the-numbers.com/movie/budgets/all (box office)

https://editorial.rottentomatoes.com/guide/best-movies/ (rotten tomatoes score)

https://www.metacritic.com/feature/film-critics-pick-10-best-movies-of-2020 (metacritics score)

https://www.imdb.com/search/title/?title_type=feature&release_date=2020&sort=num_votes,desc&count=100 (imdb data in specific year)

#### as well as some useful APIs:

```python
rotten_tomatoes_scraper
Cinemagoer
```
#### to create separate datasets in csv folder:

Rating and information datasets:

```
imdb.csv
box_office.csv
rotten.csv
```

Awards datasets:

```
oscar.csv
BAFTA.csv
CCMA.csv
DGA.csv
GG_comedy.csv
GG_drama.csv
PGA.csv
```

#### then merge all of them together to form a complete data set: 
```
first_movie_dataset.csv
```
Initially, the merging process works well whenever we run the extract_main.py. 

However, we currently face some trouble with extracting the box office data as well as the data for the Cannes (Golden Palm) award when we carry out several data wrangling and cleaning new metrics. Hence, the merging process is also affected. 

The "first_movie_dataset" is the first dataset that we collected before adding new metrics. It contains almost all data that we think is neccessary for prediction. Thus, we can still run implement_random_forest.py and calculating_algo_accuracy.py to see the needed attributes and there importance ranking (or weight).


## The correct way to run the program is:

1. Run extract_main.py to extract data.

2. Do not run this new version of merging.py and go the next step!

3. Run implement_random_forest.py to execute the program and see some first results (important metrics ranking).


## V. Future Work:
The hardest and most frustrate work for us in this project so far is properly extracting the data from different websites, cleaning and merging data to create a perfect unbias primary data set.

Hence, our future work will focus on:

- Fixing bugs resulted from box office extracting and add cannes data to the dataset.
- Finish prediction file based on results from Random Forest Classifier.
- Provide more helpful visualizations through out the process so that the audiences can can understand more clearly.
- Implement Decision Trees and Logistic Regression.
- Automation using Apache Airflow to schedule the program to run before the next Oscars happens!




