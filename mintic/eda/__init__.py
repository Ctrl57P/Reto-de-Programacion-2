import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def impute_missing(data, strategy='mean', columns=None):
    df = data.copy()
    
    if columns is None:
        columns = df.columns
        
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                val = df[col].mean()
                df[col] = df[col].fillna(val)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                val = df[col].median()
                df[col] = df[col].fillna(val)
            elif strategy == 'mode':
                val = df[col].mode()[0]
                df[col] = df[col].fillna(val)
                
    return df


def detect_outliers(data, method='iqr', threshold=1.5):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(False, index=data.index, columns=data.columns)
    
    if method == 'iqr':
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
            
    elif method == 'zscore':
        for col in numeric_cols:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0: 
                z_scores = np.abs((data[col] - mean_val) / std_val)
                outliers[col] = z_scores > threshold
                
    return outliers


def handle_outliers(data, method='iqr', action='trim', threshold=1.5):
    df = data.copy()
    outliers_mask = detect_outliers(df, method, threshold)
    
    if action == 'trim':
        df = df[~outliers_mask.any(axis=1)]
        
    elif action == 'cap':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == 'iqr':
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                df[col] = np.clip(df[col], lower, upper)
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower = mean_val - threshold * std_val
                upper = mean_val + threshold * std_val
                df[col] = np.clip(df[col], lower, upper)
                
    return df


def plot_missing(data):
    missing_counts = data.isnull().sum()
    
    plt.figure(figsize=(10, 6))
    missing_counts.plot(kind='bar', color='skyblue')
    plt.title('Cantidad de valores faltantes por columna')
    plt.ylabel('Valores nulos')
    plt.xlabel('Columnas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Guarda la imagen en la carpeta principal en lugar de intentar abrir una ventana.... ya que no se esta ejecutando jupyter-lab
    plt.savefig('missing_plot.png')
    print("Gráfica guardada exitosamente como 'missing_plot.png'")
