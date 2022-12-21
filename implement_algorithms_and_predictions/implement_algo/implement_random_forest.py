import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

#Function to calculate Accuracy Parameters (TPR,TNR,FPR,FNR)
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
    print("True negative rate: ", TNR.round(2))
    print("False negative rate: ", FNR.round(2))
    print("True positive rate: ", TPR.round(2))
    print("False positive rate: ", FPR.round(2))
    
df = pd.read_csv('csv/final_movie_dataset.csv')

df['Metascore'] = df['Metascore'].fillna(0)

#prepare your ingredients
X = ['Runtime (min)', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
     'Drama', 'Family','Fantasy', 'History', 'Horror', 'Musical', 'Mystery', 'Romance', 
     'Sci-Fi', 'Sport','Thriller', 'War', 'Western','Budget','Domestic (US) gross',
     'International gross','Worldwide gross','Metascore', 'IMDb_rating', 'IMDb_votes', 'RT_rating', 'RT_review',
     'GG_drama_winner', 'GG_drama_nominee', 'GG_comedy_winner', 'GG_comedy_nominee',
     'BAFTA_winner', 'BAFTA_nominee', 'DGA_winner', 'DGA_nominee',
     'PGA_winner', 'PGA_nominee', 'CCMA_winner', 'CCMA_nominee',
     'Golden_Palm_winner', 'Golden_Palm_nominee', 'Golden_Bear_winner', 'Golden_Bear_nominee',
     'Golden_Lion_winner', 'Golden_Lion_nominee', 'PCA_winner', 'PCA_nominee',
     'NYFCC_winner', 'NYFCC_nominee', 'OFCS_winner', 'OFCS_nominee']

#prep the x variables and y variables
X_set = df[X]
y_set = df['Oscar_winner']

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size = 0.25)

RF = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 4)
RF_xTrain = X_train
RF_xTest = X_test
RF_yTrain = y_train
RF_yTest = y_test

#fit data
RF.fit(RF_xTrain, RF_yTrain)

#predict response
RFy_trainPred = RF.predict(RF_xTrain)
RFy_testPred = RF.predict(RF_xTest)

#visualise the result
f,axes = plt.subplots(1,2,figsize = (20,8))
sb.heatmap(confusion_matrix(RF_yTrain, RFy_trainPred), annot = True, annot_kws={"size":20}, ax = axes[0], fmt='.0f')
sb.heatmap(confusion_matrix(RF_yTest, RFy_testPred), annot = True, annot_kws={"size":20}, ax = axes[1], fmt = '.0f')

print("Classification accuracy for train set: ", RF.score(RF_xTrain, RF_yTrain).round(3))
print("Classification accuracy for test set: ", RF.score(RF_xTest, RF_yTest).round(3))
print()
print("For train set:")
get_rate(confusion_matrix(RF_yTrain, RFy_trainPred))
print()
print("For test set:")
get_rate(confusion_matrix(RF_yTest, RFy_testPred))