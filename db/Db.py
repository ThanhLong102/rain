import mysql.connector as con


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


def insertData():
    mn = connect()
    cur = mn.cursor()
    sql = "INSERT INTO weather (Date,Location,MinTemp,MaxTemp,Rainfal,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RainTomorrow) VALUES (%s, %s)"
