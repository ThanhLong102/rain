from flask import Flask
from flask import request
import rain_service
import logistic_service
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)


@app.route("/features", methods=['GET'])
@cross_origin()
def getFeatures():
    return rain_service.getFeatures()


@app.route("/prediction", methods=['POST'])
@cross_origin()
def getPredictionWithFeatures():
    featuresDrop = request.get_json()
    return rain_service.getPredictionWithFeatures(featuresDrop)

@app.route("/prediction/logistic", methods=['POST'])
@cross_origin()
def getPredictionWithFeatures():
    featuresDrop = request.get_json()
    return logistic_service.getPredictionWithFeatures(featuresDrop)

if __name__ == "__main__":
    app.run()
