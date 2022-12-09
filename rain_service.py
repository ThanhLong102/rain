import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import model.nn as nn
from model.optimizer import optimizer as optimizer
import logistic_service
from NMKHDL.rain.model import decisionTree
from NMKHDL.rain.model import RandomForest
from NMKHDL.rain.model import linear_regresion
import db.Db as db

np.random.seed(0)


def encode(data, col, max_val):
    # function to encode datetime into cyclic parameters.
    # As I am planning to use this data in a neural network I prefer the months and days in a cyclic continuous feature.
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data


def preprocessor():
    result, column_name = db.getWeather()
    data = pd.DataFrame(data=result, columns=column_name)

    # There don't seem to be any error in dates so parsing values into datetime
    data['Date'] = pd.to_datetime(data["Date"])
    # Creating a collumn of year
    data['year'] = data.Date.dt.year

    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)

    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)

    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)

    # Filling missing values with mode of the column in value
    for i in object_cols:
        data[i].fillna(data[i].mode()[0], inplace=True)

    # Get list of numeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    # Filling missing values with median of the column in value
    for i in num_cols:
        data[i].fillna(data[i].median(), inplace=True)

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    for i in object_cols:
        data[i] = label_encoder.fit_transform(data[i])

    features = data.drop(['RainTomorrow', 'Date', 'day', 'month'], axis=1)  # dropping target and extra columns
    target = data['RainTomorrow']

    # Set up a standard scaler for the features
    col_names = list(features.columns)
    s_scaler = preprocessing.StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names)

    # full data for
    features["RainTomorrow"] = target
    return features


def droppingWithOutlier(features):
    # Dropping with outlier
    features = features[(features["MinTemp"] < 2.3) & (features["MinTemp"] > -2.3)]
    features = features[(features["MaxTemp"] < 2.3) & (features["MaxTemp"] > -2)]
    features = features[(features["Rainfall"] < 4.5)]
    features = features[(features["Evaporation"] < 2.8)]
    features = features[(features["Sunshine"] < 2.1)]
    features = features[(features["WindGustSpeed"] < 4) & (features["WindGustSpeed"] > -4)]
    features = features[(features["WindSpeed9am"] < 4)]
    features = features[(features["WindSpeed3pm"] < 2.5)]
    features = features[(features["Humidity9am"] > -3)]
    features = features[(features["Humidity3pm"] > -2.2)]
    features = features[(features["Pressure9am"] < 2) & (features["Pressure9am"] > -2.7)]
    features = features[(features["Pressure3pm"] < 2) & (features["Pressure3pm"] > -2.7)]
    features = features[(features["Cloud9am"] < 1.8)]
    features = features[(features["Cloud3pm"] < 2)]
    features = features[(features["Temp9am"] < 2.3) & (features["Temp9am"] > -2)]
    features = features[(features["Temp3pm"] < 2.3) & (features["Temp3pm"] > -2)]
    return features


def updateDataProcessed(features):
    database_connection = db.getConnectDataFrame()
    features.to_sql(con=database_connection, name='weather_processed', if_exists='replace', index=False)


def processedAndUpdate():
    features = preprocessor()
    features = droppingWithOutlier(features)
    updateDataProcessed(features)


def saveCorrelation(features):
    database_connection = db.getConnectDataFrame()
    corr_data = features.corr()["RainTomorrow"].sort_values(ascending=False)
    df_corr = pd.DataFrame({
        'id': [*range(1, corr_data.index.size + 1)],
        'labels': corr_data.index,
        'corr': corr_data.values})
    df_corr.set_index('id')
    print(df_corr)
    df_corr.to_sql(con=database_connection, name='correlation', if_exists='replace', index=False)


def getWeatherProcessed():
    result, column_name = db.getWeatherProcessed()
    data = pd.DataFrame(data=result, columns=column_name)
    return data


def getWeatherProcessedLr():
    result, column_name = db.getWeatherProcessedLR()
    data = pd.DataFrame(data=result, columns=column_name)
    return data


def dropFeaturesAndSplit(features: pd.DataFrame, featuresDrop: list):
    featuresDrop.append("RainTomorrow")
    X = features.drop(featuresDrop, axis=1)
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


def trainModel(X_train, y_train):
    X = np.array(X_train).T
    Y = np.array([y_train])

    # optimize using  ADAM
    net = nn.nn([X.shape[0], 32, 32, 1], ['relu', 'relu', 'sigmoid'], epochs=1)
    net.cost_function = 'MSELoss'
    print('net architecture :')
    print(net)

    optim = optimizer.AdamOptimizer
    optim(X, Y, net, alpha=0.00009, print_at=1)
    return net


def getPredictionFromTest(net, X_test, y_test):
    X_test = np.array(X_test).T
    prediction = net.forward(X_test)
    y_actual = np.array([y_test])

    prediction = 1 * (prediction >= 0.5)
    accuracy = np.sum(prediction == y_actual[0]) / prediction.shape[1]

    return accuracy * 100


def savePrediction(net, X_test, y_test):
    X_test = np.array(X_test).T
    prediction = net.forward(X_test)
    y_actual = np.array([y_test])
    result = pd.DataFrame(
        {'id': [*range(1, prediction[0].size + 1)], 'prediction': prediction[0], 'actual': y_actual[0]})
    result.to_sql(con=db.getConnectDataFrame(), name='result', if_exists='replace', index=False)


def saveMetrics(net):
    result = pd.DataFrame(net.cache_metrics)
    result.to_sql(con=db.getConnectDataFrame(), name='metric', if_exists='replace', index=False)


def getMSELoss(net, X_test, y_test):
    X_test = np.array(X_test).T
    prediction = net.forward(X_test)
    y_actual = np.array([y_test])
    return net.MSELoss(prediction, y_actual)


def getFeatures():
    features = db.getFeatures()
    listFeatures = []
    for i in range(1, len(features)):
        listFeatures.append(features[i][0])

    return listFeatures


def getPredictionWithFeatures(featuresDrop):
    features = getWeatherProcessed()
    X_train, X_test, y_train, y_test = dropFeaturesAndSplit(features, featuresDrop)
    net = trainModel(X_train, y_train)
    saveMetrics(net)
    accuracy = getPredictionFromTest(net, X_test, y_test)
    mseLoss = getMSELoss(net, X_test, y_test)
    savePrediction(net, X_test, y_test)
    return {"accuracy": str(accuracy), "mseLoss": str(mseLoss)}


def getPredictionWithFeaturesDesicitionTree(featuresDrop):
    features = getWeatherProcessed()
    features = features.iloc[0:1000]
    X_train, X_test, y_train, y_test = dropFeaturesAndSplit(features, featuresDrop)
    X_train.index = [x for x in range(1, len(X_train.values) + 1)]
    y_train.index = [x for x in range(1, len(y_train.values) + 1)]
    tree = decisionTree.DecisionTreeID3(max_depth=1, min_samples_split=2)
    tree.fit(X_train, y_train)
    a = y_test.to_numpy()
    b = tree.predict(X_test)
    print(a, b)
    result = pd.DataFrame(
        {'id': [*range(1, a.size + 1)], 'prediction': a, 'actual': b})
    result.to_sql(con=db.getConnectDataFrame(), name='result_decision', if_exists='replace', index=False)
    print(decisionTree.accuracy_score(a, b))


def getPredictionWithFeaturesRandoForest(featuresDrop):
    features = getWeatherProcessed()
    features = features.iloc[0:10000]
    X_train, X_test, y_train, y_test = dropFeaturesAndSplit(features, featuresDrop)

    clf = RandomForest.RandomFores(n_trees=3, min_samples_split=2, max_depth=5)
    clf.fit(X_train.values, y_train.values)
    y_pred = clf.predict(X_test.values)
    # y_pred =rd.predict_rf(model, X_test)
    acc = clf.accuracy(y_test.values, y_pred)
    y_pred = (y_pred > 0.5)
    result = pd.DataFrame(
        {'id': [*range(1, y_pred.size + 1)], 'prediction': y_pred, 'actual': y_test.values})
    result.to_sql(con=db.getConnectDataFrame(), name='result_randomforest', if_exists='replace', index=False)
    return acc


def getPredictionWithFeaturesLinear(featuresDrop):
    features = getWeatherProcessed()
    features = features.iloc[0:10000]
    X_train, X_test, y_train, y_test = dropFeaturesAndSplit(features, featuresDrop)

    # Model training
    model = linear_regresion.LinearRegression(iterations=1000, learning_rate=0.01)
    model.fit(X_train.values, y_train.values)

    # Prediction on test set
    Y_pred = model.predict(X_test.values)
    result = pd.DataFrame(
        {'id': [*range(1, Y_pred.size + 1)], 'prediction': Y_pred, 'actual': y_test.values})
    result.to_sql(con=db.getConnectDataFrame(), name='result_linear', if_exists='replace', index=False)
    return 100 - np.mean(np.abs(Y_pred - y_test)) * 100


def getPredictionWithFeatureLr(featuresDrop):
    return logistic_service.getPredictionWithFeatures(featuresDrop)


if __name__ == "__main__":
    print(getPredictionWithFeaturesLinear([]))
    # print(getFeatures())
