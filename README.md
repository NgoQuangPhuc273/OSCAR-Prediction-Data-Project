# **ECON 312: Predict The Academy Award Results Through Machine Learning**

Phuc Ngo, Zach Moskowitz, Ryan Blasberg


## **I. Introduction:**
The Academy Awards, also known as the OSCARs, is the most prestigious annual event that recognizes outstanding achievements in the film industry across a variety of categories. The first Academy Awards ceremony was held in 1928, and since then it has taken place around late February each year. For professionals in the film industry, receiving an OSCAR can lead to increased recognition and opportunities, such as higher salaries and a wider range of roles or films to choose from in the future. The Academy of Motion Picture Association (AMPAS) organizes the OSCARs every year, and the ceremony is highly anticipated by both industry insiders and movie fans. Since the award's inception, there have been 563 movies nominated for Best Picture and 92 winners.

In addition to the financial benefits for the film industry, the general public also has an interest in predicting the outcomes of the OSCARs. Moviegoers may participate in polls and discussions about who they think will win, and it is even possible to place bets on the OSCARs on various websites. Therefore, accurately forecasting the OSCAR winners and identifying any trends can be of great interest to many people. In this project, we aim to predict which film will win the OSCAR for the Best Picture category based on various factors and metrics that could potentially influence the final decision of film critics.

## III. Installation & Requirements:

Please go to requirements.txt for needed libraries.

## IV. Running:

1. Go to extract_and_merge folder and run extract_main.py to extract data.
2. Run merging.py to merge different data sets into a main one.
3. Go to implement_algorithms_and_predictions folder and run run_predictions.py to execute the predicting process.
4. Go to final_predictions_graphs/predictions to see the predicted Oscar winners for each year!

You can also go to implement_algorithms_and_predictions/implement_algo and run each of the models to see the actuall Classification accuracy, True negative rate, False negative rate True positive rate, and False positive rate for each model.

## **II. Description & Explanation:**

**1. Important Questions:**

- Which prediction model would be the most effective with the data we have?
- What characteristics of a film contribute to its potential to win an Oscar?
- Is it possible to accurately predict the winners of the Oscars based on the features of a film?

To investigate these questions, we will attempt to predict the winners of the Academy Award for Best Picture, and as part of that process, we will evaluate the features of a film that make it stand out compared to others.

Our plan is to extract the top 100 movie data ranked according to the number of votes from IMDB for each year from 1999 to 2020. Then, we scrape other websites and match the new data to existed data frame according to the movie title.

**2. Metrics:**

Film Elements:
- Runtime
- Genre

Movie Critics Ratings:
- IMDb User Rating
- IMDB User Votes
- Rotten Tomatoes Critics Rating
- Rotten Tomatoes Critics Review
- Metascore

Commercial
- Budget
- Domestic (US) gross
- International gross
- Worldwide gross

Film Awards
- The British Academy of Film and Television Arts Film Awards (BAFTA)  
- Director Guild Awards (DGA)  
- Producer Guild Awards (PGA)  
- Golden Globes - Comedy  
- Golden Globes – Drama 
- Cannes International Film Festival (Golden Palm)  
- Berlin International Film Festival (Golden Bear)  
- Venice Film Festival (Golden Lion)  
- Toronto Film Festival – People Choice’s Award  
- New York Film Critics Circle (NYFCC) Award for Best Picture  
- Critics’ Choice Movie Award (CCMA) for Best Picture  
- Online Film Critics Society Award (OFCSA) for Best Picture  

**3. Extracting & cleaning data:**

For this project, we decided to self-extract the needed data for our models using BeautifulSoup, requests, and rotten_tomatoes_scraper to build our datasets from scratch. Meanwhile, we will also perform data-cleaning whenever it is necessary.

We studied the HTML structure of these websites:

https://www.imdb.com/search/title/?title_type=feature&release_date=%s&sort=num_votes,desc&count=100

https://www.the-numbers.com/movie/budgets/all/

https://en.wikipedia.org/wiki/


https://www.rottentomatoes.com/top/bestofrt/year

**4. Implement Machine Learning Models:**

We classified all 53 features we have into 4 main categories:
Movie Elements, Critics Ratings, Box Office and Awards.

We chose to use four machine learning approaches:
- Decision Tree
- Random Forest
- Logistic Regression
- Light Gradient-Boost Machine

to examine how well is each feature category in predicting the Oscar winner (accuracy, true & false negative/positve rate) and also find out which model will best suit the final movie dataset.

**5. Predicting the winners:**

For the last part of our project, we will try to predict the actual Oscar winner directly using the 'predict_proba' function from each model to determine the predicted winner based on the probability of winning. 

We will also be using the 'feature_importance_' function to visualise the weightage of importance of each feature in different classifiers. We will split our movie dataset to training set (1999-2014) and testing set (2015-2019).


## V. Results and Analysis:

We were quite sucessfully in answering the questions raised at the beginning of the project!

- Which prediction model would be the most effective with the data we have?

Random Forest and Light Gradient Boosting Machine (LGBM) were equally good.

Logistic Regression came second.

Decision Tree was the worst one.

- What characteristics of a film contribute to its potential to win an Oscar?

For Random Forest: Director Guild Awards, Producer Guild Award or the Critics' Choice Movie Awards

For LGBM: critical ratings (IMDb_ratings, Tomatometer, Metascore)

- Is it possible to accurately predict the winners of the Oscars based on the features of a film?

The answers will depend on many factors. Indeed, no model can predict 5/5 correctly. The maximum accuracy we can get is 3/5 (Random Forest and LGBM).

