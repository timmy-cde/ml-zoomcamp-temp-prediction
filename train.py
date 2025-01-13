import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer

import cloudpickle

output_file = 'model.bin'

# data preparation--------------------------------------------------------------
df = pd.read_parquet('data/combined_hourly_data_mnl.parquet')

df['year'] = df.datetime.dt.year
df['month'] = df.datetime.dt.month
df['day_of_week'] = df.datetime.dt.day_of_week
df['day_of_year'] = df.datetime.dt.day_of_year
df['hour'] = df.datetime.dt.hour

numericals = df.columns.to_list()

targets = ['temperature_2m', 'apparent_temperature']
categories = ['weather_code', 'year'] 
cyclical_features = ['month', 'day_of_week', 'day_of_year', 'hour']
non_numeric_cols = targets + categories + cyclical_features + ['datetime']

for cols in non_numeric_cols:
    numericals.remove(cols)

total_len = len(df)
test_len= int(len(df) * 0.2)
train_len = total_len - test_len

tscv = TimeSeriesSplit(
    n_splits=4,
    gap=24,
    max_train_size=int(train_len * 0.8),
    test_size=int(train_len * 0.2),
)

df_train = df.iloc[:train_len]
df_test = df.iloc[train_len:]

y = df[targets].values
y_train = y[:train_len]
y_test = y[train_len:]

# training----------------------------------------------------------------------
def sin_transformer(max_val):
    return FunctionTransformer(lambda x: np.sin((x * 2 * np.pi) / max_val), feature_names_out='one-to-one')

def cos_transformer(max_val):
    return FunctionTransformer(lambda x: np.cos((x * 2 * np.pi) / max_val), feature_names_out='one-to-one')

transformations = [
    ('numerical', 'passthrough', numericals),
    ('day_of_year_sin', sin_transformer(365), ['day_of_year']),
    ('day_of_year_cos', cos_transformer(365), ['day_of_year']),
    ('month_sin', sin_transformer(12), ['month']),
    ('month_cos', cos_transformer(12), ['month']),
    ('day_of_week_sin', sin_transformer(7), ['day_of_week']),
    ('day_of_week_cos', cos_transformer(7), ['day_of_week']),
    ('hour_sin', sin_transformer(24), ['hour']),
    ('hour_cos', cos_transformer(24), ['hour']),
    ('category', OneHotEncoder(dtype='int32', handle_unknown='ignore'), categories)
]

transformer = ColumnTransformer(
    transformations,
    remainder='drop'
)

def evaluate(model, X, y, cv, model_prop=None, model_step=None):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )

params = {
    'rd__alpha': [0, 0.01, 0.1, 1, 10, 100]
}

rd_pipeline = Pipeline([
    ('transformer', transformer),
    ('rd', Ridge(random_state=42, max_iter=1000))
])

model_rd = RandomizedSearchCV(
    estimator=rd_pipeline,
    param_distributions=params,
    cv=tscv,
    random_state=42
)

model_rd.fit(df_train, y_train)

print(f'best score: {model_rd.best_score_}')
print(f'best parameters: {model_rd.best_params_}')

# validation--------------------------------------------------------------------
rd_pipeline = Pipeline([
    ('transformer', transformer),
    ('rd', Ridge(alpha=10, random_state=42, max_iter=1000))
])

rd_pipeline.fit(df_train, y_train)

evaluate(rd_pipeline, df_train, y_train, cv=tscv)

# Saving the model--------------------------------------------------------------
with open(output_file, 'wb') as f_out:
    cloudpickle.dump(rd_pipeline, f_out)

print(f'The model is saved to {output_file}')
