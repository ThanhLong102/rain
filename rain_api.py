from flask import Flask
from flask import request
import rain_service
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
    modelSelect = request.args.get('model')
    if modelSelect=='decisionTree':
        return rain_service.getPredictionWithFeaturesDesicitionTree(featuresDrop)
    elif modelSelect=='linear_regresion':
        return rain_service.getPredictionWithFeaturesLinear(featuresDrop)
    elif modelSelect=='decision_tree_rd':
        return rain_service.getPredictionWithFeaturesRandoForest(featuresDrop)
    elif modelSelect=='logisticRegression':
        return rain_service.getPredictionWithFeatureLr(featuresDrop)
    else:
        return rain_service.getPredictionWithFeatures(featuresDrop)

if __name__ == "__main__":
    app.run()
