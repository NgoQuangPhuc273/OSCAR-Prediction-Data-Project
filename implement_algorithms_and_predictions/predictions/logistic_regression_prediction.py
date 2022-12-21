import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import time
pd.set_option("display.max_colwidth", 200)

# Classifiers 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('csv/final_movie_dataset.csv')

# Train on 16 years, predict on recent 5 years
train = data[data['Year'] < 2015]
test = data[data['Year'] >= 2015]

train['Oscar_winner'].value_counts()

movie_name = np.array(test["Movie"])
year = np.array(test["Year"])
oscar_w = np.array(test["Oscar_winner"])
oscar_n = np.array(test["Oscar_nominee"])

# feat. = feature
feat_film_elements = ['Runtime (min)', 'Action', 'Adventure', 'Animation', 'Biography', 
                      'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Musical', 'Mystery', 
                      'Romance', 'Sci-Fi', 'Sport','Thriller', 'War', 'Western']  # Genre - binary  #20

feat_movie_critics = ['IMDb_rating', 'IMDb_votes','RT_rating','RT_review','Metascore']  #5

feat_commercial = ['Budget','Domestic (US) gross', 'International gross', 'Worldwide gross'] #4

feat_awards = ['GG_drama_winner', 'GG_drama_nominee', 'GG_comedy_winner', 'GG_comedy_nominee',
               'BAFTA_winner', 'BAFTA_nominee', 'DGA_winner', 'DGA_nominee',
               'PGA_winner', 'PGA_nominee', 'CCMA_winner', 'CCMA_nominee',
               'Golden_Palm_winner', 'Golden_Palm_nominee', 'Golden_Bear_winner', 'Golden_Bear_nominee',
               'Golden_Lion_winner', 'Golden_Lion_nominee', 'PCA_winner', 'PCA_nominee',
               'NYFCC_winner', 'NYFCC_nominee', 'OFCS_winner', 'OFCS_nominee'] #24

all_features = []

all_features = feat_film_elements + feat_movie_critics + feat_commercial  + feat_awards

X_train = train[all_features]
X_test = test[all_features]
y_train = train['Oscar_winner']
y_test = test['Oscar_winner']

# transform the data to standardize the values in the data 
preprocessor = ColumnTransformer(transformers=[('scale', StandardScaler(), all_features)])

def get_scores(model, X_train, y_train, X_test, y_test, show = True):
    
    if show: 
        print("Training error:   %.2f" % (1-model.score(X_train, y_train)))
        print("Validation error: %.2f" % (1-model.score(X_test, y_test)))
        print('\n')
    return (1-model.score(X_train, y_train)), (1-model.score(X_test, y_test))

def diff_class_ml(X_train, X_test, y_train, y_test):
    
    # Lets create an empty dictionary to store all the results
    results_dict = {}
    
    models = {
            # The Trees
            'Decision Tree': DecisionTreeClassifier(),
            
            'Random Forest' : RandomForestClassifier(),

            'Light Gradient Boosting Machine': LGBMClassifier(),

            'Logistic Regression': LogisticRegression(), 
              }

    for model_name, model in models.items():
        t = time.time() 
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train);
        tr_err, valid_err = get_scores(clf, X_train, y_train, X_test, y_test, show = False)
        elapsed_time = time.time() - t
        results_dict[model_name] = [round(tr_err,3), round(valid_err,3), round(elapsed_time,3)]
    
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ["Train error", "Validation error", "Elapased Time (s)"]
    return results_df

diff_class_ml(X_train, X_test, y_train, y_test)

def logistic_regression():
    LR = LogisticRegression()

    my_LR = LR.fit(X_train, y_train)
    
    LR_importances = pd.DataFrame(my_LR.feature_importances_.round(3), all_features, columns=["Importances Weightage"])

    LR_importances['Features'] = all_features
    fig = px.bar(LR_importances, x='Features', y='Importances Weightage', 
                title='Logistic Regression Features Importances', height=600)
    fig.show()

    pred_LR = my_LR.predict_proba(X_test)[:, 1]

    LR_prediction = pd.DataFrame(year, columns=["Year"])
    LR_prediction["Movie"] = movie_name
    LR_prediction["Oscar_nominee"] = oscar_n
    LR_prediction["Oscar_winner"] = oscar_w
    LR_prediction['Predicted Win Rate'] = pred_LR

    normalized_prediction = LR_prediction.copy()

    for index, row in normalized_prediction.iterrows():
        normalized_prediction.loc[index, "Predicted Win Rate"] = \
            (row["Predicted Win Rate"] / LR_prediction["Predicted Win Rate"][LR_prediction["Year"] == row["Year"]].sum()).round(3)
    
    years = [2015, 2016, 2017, 2018, 2019]
    
    for single_year in years:
        predicted_result = normalized_prediction[normalized_prediction["Year"] == single_year].sort_values("Predicted Win Rate", ascending=True).tail(10)
        predicted_result.to_csv("final_prediction_csv/logistic_regression/predict_result_{}.csv".format(str(single_year)))
        
        predicted_result.plot(kind='barh',x='Movie',y='Predicted Win Rate',color='gold',figsize=(10, 5)) 
    
        plt.title("Logistic Regression Predictions") 
        plt.xlabel("Win rate") 
        plt.ylabel("") 
        plt.legend(['Predicted Win Rate'], loc = 'lower right')
        plt.tight_layout()
        plt.savefig('final_predictions_graphs/predictions/{}/{}_logistic_regression.png'.format(str(single_year),str(single_year)))
logistic_regression()