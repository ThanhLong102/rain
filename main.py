import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import model.nn as nn
from model.optimizer import optimizer as optimizer

np.random.seed(0)


def run():
    data = pd.read_csv("./weatherAUS.csv")
    data.head()
    data.info()

    # first of all let us evaluate the target and find out if our data is imbalanced or not
    cols = ["#C2C4E2", "#EED4E5"]
    sns.countplot(x=data["RainTomorrow"], palette=cols)

    # exploring the length of date objects
    lengths = data["Date"].str.len()
    lengths.value_counts()

    # There don't seem to be any error in dates so parsing values into datetime
    data['Date'] = pd.to_datetime(data["Date"])
    # Creating a collumn of year
    data['year'] = data.Date.dt.year

    # function to encode datetime into cyclic parameters.
    # As I am planning to use this data in a neural network I prefer the months and days in a cyclic continuous feature.

    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
        return data

    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)

    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)

    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)

    print("Categorical variables:")
    print(object_cols)

    # Missing values in categorical variables
    for i in object_cols:
        print(i, data[i].isnull().sum())

    # Filling missing values with mode of the column in value
    for i in object_cols:
        data[i].fillna(data[i].mode()[0], inplace=True)

    # Get list of numeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    print("Numeric variables:")
    print(num_cols)

    # Missing values in numeric variables
    for i in num_cols:
        print(i, data[i].isnull().sum())

    # Filling missing values with median of the column in value
    for i in num_cols:
        data[i].fillna(data[i].median(), inplace=True)

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    for i in object_cols:
        data[i] = label_encoder.fit_transform(data[i])

    # Preparing attributes of scale data
    features = data.drop(['RainTomorrow', 'Date', 'day', 'month'], axis=1)  # dropping target and extra columns

    target = data['RainTomorrow']

    # full data for
    features["RainTomorrow"] = target

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

    X = features.drop(["RainTomorrow"], axis=1)
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X = np.array(X_train).T
    Y = np.array([y_train])

    # optimize using  ADAM
    net = nn.nn([26, 32, 32, 1], ['relu', 'relu', 'sigmoid'], epochs=10)
    net.cost_function = 'CrossEntropyLoss'
    print('net architecture :')
    print(net)

    optim = optimizer.AdamOptimizer
    optim(X, Y, net, alpha=0.00009, lamb=0.05, print_at=1)


if __name__ == "__main__":
    run()
