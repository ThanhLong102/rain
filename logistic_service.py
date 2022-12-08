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
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test