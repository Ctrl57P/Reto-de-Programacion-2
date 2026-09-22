#######
import pandas as pd
from mintic.ensemble.random_forest import (
    bootstrap_sample,
    build_id3_tree,
    build_random_forest,
    predict_ensemble
)

try:
    df = pd.read_csv('data/car.data', header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_sample, y_sample = bootstrap_sample(X, y, random_state=42)
    forest = build_random_forest(X_sample, y_sample, n_trees=10, random_state=42)
    
    X_test = X[:10]
    predictions = predict_ensemble(forest, X_test)
    
    print("El código corre sin errores")
except Exception as e:
    print(f"Ocurrió un error: {e}")
