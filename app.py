import flask
import threading
from flask import jsonify
from preprocessors.preprocessing import start_processing
from models.Kmeans import KMeansModelCustom
import service.service as s
from datetime import datetime
import numpy as np

app = flask.Flask(__name__)


def activate_job():
    def run_job():
        print("Preprocessing thread started at ", datetime.now())
        start_processing()
        print("Preprocessing thread finished at ", datetime.now())

    thread = threading.Thread(target=run_job)
    thread.start()


@app.route('/api/v1/predict', methods=['POST'])
def predict_xgb():
    req = flask.request
    ltd = req.args.get('driver_lat')
    lng = req.args.get('driver_lng')
    n_clusters = req.args.get('n_clusters')
    # timestamp = req.args.get('timestamp')
    prediction = s.enrich_prediction_request(ltd, lng, n_clusters, datetime.now())
    return jsonify(prediction)


@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    print("Request accepted")
    # xg_model.train()


@app.route('/api/v1/cluster', methods=['POST'])
def predict_cluster():
    req = flask.request
    ltd = req.args.get('driver_lat')
    lng = req.args.get('driver_lng')
    f = np.array([ltd, lng]).reshape((1, -1))
    k_means_model = KMeansModelCustom(use_pretrained=True)
    prediction = k_means_model.predict(f)
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    # activate_job()
    app.run()
