import pandas as pd
import random
import json


df = pd.read_parquet('data/test_data.parquet')
df['datetime'] = df['datetime'].astype(str)

del df['temperature_2m']
del df['apparent_temperature']

number = random.randint(0, len(df))
print(json.dumps(df.iloc[number].to_dict()))