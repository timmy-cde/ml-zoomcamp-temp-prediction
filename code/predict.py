import cloudpickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd

with open('model.bin', 'rb') as f_in:
    model = cloudpickle.load(f_in)

app = Flask('temp_predict')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    data = pd.DataFrame([data])

    data['datetime'] = pd.to_datetime(data['datetime'])
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day_of_week'] = data['datetime'].dt.day_of_week
    data['day_of_year'] = data['datetime'].dt.day_of_year
    data['hour'] = data['datetime'].dt.hour

    y_pred = model.predict(data)

    result = {
        'temperature_2m': y_pred.item(0),
        'apparent_temperature': y_pred.item(1)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)