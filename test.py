import requests
import argparse
import json

url = 'http://localhost:9696/predict'

parser = argparse.ArgumentParser("Test Temperature Prediction Model")
parser.add_argument("-i","--input_data", nargs='?', help="A dictionary containing the input data of the model (use gen.py script to generate data from test data)")
args = parser.parse_args()

input = args.input_data

if input is not None:
    data = json.loads(input)
else:
    data = {
        "datetime": "2023-12-18 07:00:00",
        "relative_humidity_2m": 85.0,
        "dew_point_2m": 22.6,
        "precipitation": 0.0,
        "rain": 0.0,
        "weather_code": 3.0,
        "pressure_msl": 1013.2,
        "surface_pressure": 1012.2,
        "cloud_cover": 100.0,
        "cloud_cover_low": 47.0,
        "cloud_cover_mid": 55.0,
        "cloud_cover_high": 100.0,
        "et0_fao_evapotranspiration": 0.03,
        "vapour_pressure_deficit": 0.5,
        "wind_speed_10m": 6.7,
        "wind_speed_100m": 8.1,
        "wind_direction_10m": 16.0,
        "wind_direction_100m": 13.0,
        "wind_gusts_10m": 12.6,
        "soil_temperature_0_to_7cm": 25.6,
        "soil_temperature_7_to_28cm": 28.0,
        "soil_temperature_28_to_100cm": 29.2,
        "soil_temperature_100_to_255cm": 29.0,
        "soil_moisture_0_to_7cm": 0.339,
        "soil_moisture_7_to_28cm": 0.296,
        "soil_moisture_28_to_100cm": 0.356,
        "soil_moisture_100_to_255cm": 0.426,
        "shortwave_radiation": 26.0,
        "direct_radiation": 10.0,
        "diffuse_radiation": 16.0,
        "direct_normal_irradiance": 86.2,
        "global_tilted_irradiance": 26.0,
        "terrestrial_radiation": 102.8,
        "shortwave_radiation_instant": 56.8,
        "direct_radiation_instant": 21.8,
        "diffuse_radiation_instant": 34.9,
        "direct_normal_irradiance_instant": 137.4,
        "global_tilted_irradiance_instant": 35.6,
        "terrestrial_radiation_instant": 224.6
    }

response = requests.post(url, json=data).json()
print(response)

# actual data of the default input from test data:
# 'temperature_2m': 25.4
# 'apparent_temperature': 29.8