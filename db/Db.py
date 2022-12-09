import mysql.connector as con
# Import dataframe into MySQL
import sqlalchemy


def getConnectDataFrame():
    database_username = 'root'
    database_password = 'anhhieu123'
    database_ip = 'localhost'
    database_name = 'rain'

    database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                                   format(database_username, database_password,
                                                          database_ip, database_name))
    return database_connection


def connect():
    mn = con.connect(host="localhost",
                     user='root',
                     password='anhhieu123',
                     database="rain")
    return mn


def getWeather():
    mn = connect()
    cur = mn.cursor()
    cur.execute('SELECT * From weather;')
    column_name = cur.column_names
    result = cur.fetchall()
    cur.close()
    mn.close()
    return result, column_name


def getWeatherProcessed():
    mn = connect()
    cur = mn.cursor()
    cur.execute('SELECT * From weather_processed;')
    column_name = cur.column_names
    result = cur.fetchall()
    cur.close()
    mn.close()
    return result, column_name


def getWeatherProcessedLR():
    mn = connect()
    cur = mn.cursor()
    cur.execute('SELECT * From weather_processed_logistic;')
    column_name = cur.column_names
    result = cur.fetchall()
    cur.close()
    mn.close()
    return result, column_name


def insertData():
    mn = connect()
    cur = mn.cursor()
    sql = "INSERT INTO weather (Date,Location,MinTemp,MaxTemp,Rainfal,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RainTomorrow) VALUES (%s, %s)"


def getFeatures():
    mn = connect()
    cur = mn.cursor()
    cur.execute('SELECT labels From correlation;')
    column_name = cur.fetchall()
    cur.close()
    mn.close()
    return column_name
