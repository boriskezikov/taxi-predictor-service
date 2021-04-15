import flask
import threading
from flask import jsonify
from preprocessors.initial_preprocess import init_preprocessing
from models.XGBoost import XGBoostModel

app = flask.Flask(__name__)
xg_model = XGBoostModel()


def activate_job():
    def run_job():
        print("Job started")
        init_preprocessing(xg_model)

    thread = threading.Thread(target=run_job)
    thread.start()


@app.route('/api/v1/predict', methods=['POST'])
def predict_xgb():
    req = flask.request
    ltd = req.args.get('driver_lat')
    lng = req.args.get('driver_lng')
    timestamp = req.args.get('timestamp')
    print("Requested prediction for driver at lat: " + ltd + ", lng: " + lng + ", timestamp: " + timestamp)
    return jsonify(points)


@app.route('/api/v2/predict', methods=['GET'])
def test():
    print("Request accepted")
    xg_model.train()


points = [
    {'lat': 73.123451234,
     'lng': 45.001420412,
     },
    {'lat': 73.2221234512,
     'lng': 45.0101420412,
     },
    {'lat': 73.12423451234,
     'lng': 45.12401420412,
     },
    {'lat': 73.123424412,
     'lng': 45.001412356,
     },
    {'lat': 73.124152341,
     'lng': 45.214523412,
     },
    {'lat': 73.125589023,
     'lng': 45.522551463,
     },
    {'lat': 73.155235613,
     'lng': 46.612356125,
     },
    {'lat': 74.001234512,
     'lng': 44.123312223,
     }
]

if __name__ == '__main__':
    activate_job()
    app.run()
