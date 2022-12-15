import numpy as np
import pandas as pd
import pandas_profiling
import plotly.express as px
import pickle 
import graphviz
import matplotlib.pyplot as plt
import time
pd.set_option("display.max_colwidth", 200)

# Classifiers
from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Extra
from sklearn.preprocessing import normalize, scale, Normalizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

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
            'Extra Trees' : ExtraTreesClassifier(),
        
            # Boosting
            'AdaBoost Classifier' : AdaBoostClassifier(),
            'XGBoost Classifier' : XGBClassifier(),
            'Gradient Boosting Classifier' : GradientBoostingClassifier(),
            'Light Gradient Boosting Machine': LGBMClassifier(),
        
            # Naive Bayes
            # 'Multinomial Naive Bayes' : MultinomialNB(), input error
            'Gaussian Naive Bayes' : GaussianNB(),
            'Bernoulli Naive Bayes' : BernoulliNB(),
            
            # Others
            'Linear Support Vector Classifier' : LinearSVC(dual=False),
            'Support Vector Classifier' : SVC(),
            'K-Nearest Neighbors Classifier' : KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(), 
            'Bagging Classifier' : BaggingClassifier(),
            'Multi Layer Perceptron Classifier' : MLPClassifier(),
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
    
    # LR_importances = pd.DataFrame(my_LR.feature_importances_.round(3), all_features, columns=["Importances Weightage"])

    # LR_importances['Features'] = all_features
    # fig = px.bar(LR_importances, x='Features', y='Importances Weightage', 
    #             title='Logistic Regression Features Importances', height=600)
    # fig.show()

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
            
    predict_result_2015 = normalized_prediction[normalized_prediction["Year"] == 2015].sort_values("Predicted Win Rate", ascending=False).head(10)
    predict_result_2015.to_csv("final_prediction/logistic_regression/predict_result_2015.csv")

    predict_result_2016 = normalized_prediction[normalized_prediction["Year"] == 2016].sort_values("Predicted Win Rate", ascending=False).head(10)
    predict_result_2016.to_csv("final_prediction/logistic_regression/predict_result_2016.csv")

    predict_result_2017 = normalized_prediction[normalized_prediction["Year"] == 2017].sort_values("Predicted Win Rate", ascending=False).head(10)
    predict_result_2017.to_csv("final_prediction/logistic_regression/predict_result_2017.csv")

    predict_result_2018= normalized_prediction[normalized_prediction["Year"] == 2018].sort_values("Predicted Win Rate", ascending=False).head(10)
    predict_result_2018.to_csv("final_prediction/logistic_regression/predict_result_2018.csv")

    predict_result_2019 = normalized_prediction[normalized_prediction["Year"] == 2019].sort_values("Predicted Win Rate", ascending=False).head(10)
    predict_result_2019.to_csv("final_prediction/logistic_regression/predict_result_2019.csv")