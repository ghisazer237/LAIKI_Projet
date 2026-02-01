"""Utilities pour l'entraînement et l'optimisation du Decision Tree sur Diamonds.csv"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(path='data/diamonds.csv'):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        import seaborn as sns
        df = sns.load_dataset('diamonds')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df


def build_preprocessor(df, features=None):
    if features is None:
        features = [c for c in df.columns if c != 'price']
    X = df[features]
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = StandardScaler()
    # Use `sparse_output` for newer scikit-learn versions; fall back to `sparse` for older versions
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    return preprocessor


def train_baseline(df, features, target='price', test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]
    preprocessor = build_preprocessor(df, features)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=random_state))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return pipe, metrics


def tune_model(df, features, target='price', test_size=0.2, random_state=42, n_iter=30, cv=5):
    X = df[features]
    y = df[target]
    preprocessor = build_preprocessor(df, features)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=random_state))])
    param_dist = {
        'model__max_depth': [None] + list(range(3, 30)),
        'model__min_samples_split': list(range(2, 20)),
        'model__min_samples_leaf': list(range(1, 20)),
        'model__max_features': [None, 'auto', 'sqrt', 'log2'],
        'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=cv, n_jobs=-1, scoring='neg_root_mean_squared_error', random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    metrics = {
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return best, search.best_params_, metrics


def save_model(pipe, path='model.joblib'):
    joblib.dump(pipe, path)
    return path
