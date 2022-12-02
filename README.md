# ECON 312: Predict The Academy Award Results Through Machine Learning

Phuc Ngo, Zach Moskowitz, Ryan Blasberg

## I. Introduction:

## II. Description:


## III. Installation:
You will need to install these libraries to run the program.

```bash
pip install pandas
pip install bs4
pip install sklearn
pip install matplotlib
pip install seaborn
pip install plotly
```

## IV. Running:

1. Run extract_main.py to extract data from.

2. Run merging.py to merge different data sets into a main one.

3. Run implement_{algorithm}.py (decision tree, random forest or logistic regression) to execute the program.

4. Run winner_prediction.py to see the predicted winners for the next Oscar!

However, that is when the project is finished!

At the moment, we have successfully self-extract, scraping, cleaning and merging the movies data through several websites such as:

https://en.wikipedia.org/

https://www.imdb.com/search/title/?title_type=feature&release_date=2020&sort=num_votes,desc&count=100

as well as some APIs such as:
```python
rotten_tomatoes_scraper
Cinemagoer
```


