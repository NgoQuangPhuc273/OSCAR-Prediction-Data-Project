import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
sb.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree


data = pd.read_csv('csv/first_movie_dataset.csv')

feat_awards = ['GG_drama_winner', 'GG_drama_nominee', 'GG_comedy_winner', 'GG_comedy_nominee',
               'BAFTA_winner', 'BAFTA_nominee', 'DGA_winner', 'DGA_nominee',
               'PGA_winner', 'PGA_nominee', 'CCMA_winner', 'CCMA_nominee',
               'Golden_Palm_winner', 'Golden_Palm_nominee', 'Golden_Bear_winner', 'Golden_Bear_nominee',
               'Golden_Lion_winner', 'Golden_Lion_nominee', 'PCA_winner', 'PCA_nominee',
               'NYFCC_winner', 'NYFCC_nominee', 'OFCS_winner', 'OFCS_nominee'] #24

oscar_nominee = ['Oscar_nominee']

film_elements = ['Runtime (min)', 'Action', 'Adventure', 'Animation', 'Biography', 
                      'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Musical', 'Mystery', 
                      'Romance', 'Sci-Fi', 'Sport','Thriller', 'War', 'Western']  # Genre - binary  #20

movie_critics = ['IMDb_rating', 'IMDb_votes','RT_rating','RT_review','Metascore']  #5

commercial = ['Budget','Domestic (US) gross', 'International gross', 'Worldwide gross'] #4

awards = ['GG_drama_winner', 'GG_drama_nominee', 'GG_comedy_winner', 'GG_comedy_nominee',
               'BAFTA_winner', 'BAFTA_nominee', 'DGA_winner', 'DGA_nominee',
               'PGA_winner', 'PGA_nominee', 'CCMA_winner', 'CCMA_nominee',
               'Golden_Palm_winner', 'Golden_Palm_nominee', 'Golden_Bear_winner', 'Golden_Bear_nominee',
               'Golden_Lion_winner', 'Golden_Lion_nominee', 'PCA_winner', 'PCA_nominee',
               'NYFCC_winner', 'NYFCC_nominee', 'OFCS_winner', 'OFCS_nominee']

oscar_nominee = ['Oscar_nominee']

all_feature = []
all_feature = film_elements + movie_critics + commercial + awards

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def get_rate(conArray):
    con = conArray.ravel()
    TN = con[0]
    FP = con[1]
    FN = con[2]
    TP = con[3]
    
    TNR = TN/(TN+FP)
    FNR = FN/(FN+TP)
    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)
    
    print(color.RED + "True Positive Rate  \t " + color.RED +": ", TPR.round(3), color.END)
    print("True Negative Rate  \t : ", TNR.round(3))
    print("False Positive Rate \t : ", FPR.round(3))
    print("False Negative Rate \t : ", FNR.round(3))

#prep the x variables and y variables
X_set_movie_critics = data[feat_awards]
y_set_movie_critics = data['Oscar_winner']

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_set_movie_critics, y_set_movie_critics, test_size = 0.25)

forest = RandomForestClassifier(
    max_depth=25,
    min_samples_split=15,
    n_estimators=1000,
    random_state=1)

my_forest = forest.fit(X_train, y_train)

forest_importances = pd.DataFrame(my_forest.feature_importances_.round(3), feat_awards, columns=["Importances Weightage"])

print(forest_importances)
print('Score', my_forest.score(X_train, y_train))

forest_importances['Features'] = feat_awards
fig = px.bar(forest_importances, x='Features', y='Weight', 
             title='Features Importances', height=600)
fig.show()

