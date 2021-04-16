import flask
import threading
from flask import jsonify
from preprocessors.initial_preprocess import init_preprocessing
from models.XGBoost import XGBoostModel
from models.Kmeans import KmeansModel
from datetime import datetime
import numpy as np

app = flask.Flask(__name__)
xg_model = XGBoostModel()
k_means_model = KmeansModel()


def activate_job():
    def run_job():
        print("Preprocessing thread started at ", datetime.now())
        init_preprocessing(xg_model, k_means_model)
        print("Preprocessing thread finished at ", datetime.now())

    thread = threading.Thread(target=run_job)
    thread.start()


@app.route('/api/v1/predict', methods=['POST'])
def predict_xgb():
    req = flask.request
    ltd = req.args.get('driver_lat')
    lng = req.args.get('driver_lng')
    f = np.array([ltd, lng]).reshape((1, -1))
    prediction = xg_model.predict(f)
    return jsonify(prediction.tolist())


@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    print("Request accepted")
    xg_model.train()


@app.route('/api/v1/cluster', methods=['POST'])
def predict_cluster():
    req = flask.request
    ltd = req.args.get('driver_lat')
    lng = req.args.get('driver_lng')
    f = np.array([ltd, lng]).reshape((1, -1))
    prediction = k_means_model.predict(f)
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    activate_job()
    app.run()
