import decision_tree_prediction
import lgbm_prediction
import random_forest_prediction
import logistic_regression_prediction

def run():
    try:
        decision_tree_prediction.decision_tree()
        print("")
    except:
        print("An error has occurred in decision_tree_prediction.py")
        print("")
        
    try:
        random_forest_prediction.random_forest()
        print("")
    except:
        print("An error has occurred in random_forest_prediction.py")
        print("")
        
    try:
        lgbm_prediction.lgbm()
        print("")
    except:
        print("An error has occurred in lgbm_prediction.py")
        print("")
    
    try:
        logistic_regression_prediction.logistic_regression()
        print("")
    except:
        print("An error has occurred in logistic_regression_prediction.py")
        print("")

run()