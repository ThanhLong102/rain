import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import model.logisticRegresstion as LR
import db.Db as db

np.random.seed(0)
def getWeatherProcessed():
    result, column_name = db.getWeatherProcessed()
    data = pd.DataFrame(data=result, columns=column_name)
    return data

def dropFeaturesAndSplit(features: pd.DataFrame, featuresDrop: list):
    featuresDrop.append("RainTomorrow")
    X = features.drop(featuresDrop, axis=1)
    y = features.RainTomorrow.values

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=75)
    return X_train, X_test, y_train, y_test

def predict(X_train, y_train,x_test):
    dimension = X_train.shape[0]
    w, b = LR.initialize_weight_bias(dimension) 
    parameters, gradients, cost_list = LR.update(w, b, X_train, y_train, learning_rate=1, nu_of_iteration=400)

    # Lets use x_test for predicting y:
    y_test_predictions = LR.prediction(parameters['weight'], parameters['bias'], x_test) 

    # Investigate the accuracy:
    return y_test_predictions

def savePrediction(y_pred, y_test):
    database_connection = db.getConnectDataFrame()
    result = pd.DataFrame(
        {'id': [*range(1, y_pred[0].size + 1)], 'prediction': y_pred[0], 'actual': y_test})
    result.to_sql(con=database_connection, name='result_logisticregresstion', if_exists='replace', index=False)

def getPredictionWithFeatures(featuresDrop):
    features = getWeatherProcessed()
    X_train, X_test, y_train, y_test = dropFeaturesAndSplit(features, featuresDrop)
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    y_pred = (X_train, y_train,X_test)
    acccuray = format(100 - np.mean(np.abs(y_pred - y_test))*100)
    savePrediction(y_pred, y_test)
    return acccuray