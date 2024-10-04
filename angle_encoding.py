import numpy as np
import pandas as pd

def Rx(theta):
    """Returns the Rx rotation matrix."""
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])

def Ry(theta):
    """Returns the Ry rotation matrix."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ])

def Rz(lmbda):
    """Returns the Rz rotation matrix."""
    return np.array([
        [np.exp(-1j * lmbda / 2), 0],
        [0, np.exp(1j * lmbda / 2)]
    ])

def apply_rx_encoding(df):
    """Apply the Rx gate to numeric columns, excluding ID columns."""
    numeric_columns = df.select_dtypes(include=[float, int])
    
    # Exclude columns named exactly "id" or its variants
    excluded_columns = [col for col in df.columns if col.lower() in ['id']]
    numeric_columns = numeric_columns.drop(columns=excluded_columns, errors='ignore')
    
    for col in numeric_columns.columns:
        df[f'{col}_Rx'] = df[col].apply(lambda theta: Rx(np.radians(theta)))
    return df

def apply_ry_encoding(df):
    """Apply the Ry gate to numeric columns, excluding ID columns."""
    numeric_columns = df.select_dtypes(include=[float, int])
    
    # Exclude columns named exactly "id" or its variants
    excluded_columns = [col for col in df.columns if col.lower() in ['id']]
    numeric_columns = numeric_columns.drop(columns=excluded_columns, errors='ignore')
    
    for col in numeric_columns.columns:
        df[f'{col}_Ry'] = df[col].apply(lambda theta: Ry(np.radians(theta)))
    return df

def apply_rz_encoding(df):
    """Apply the Rz gate to numeric columns, excluding ID columns."""
    numeric_columns = df.select_dtypes(include=[float, int])
    
    # Exclude columns named exactly "id" or its variants
    excluded_columns = [col for col in df.columns if col.lower() in ['id']]
    numeric_columns = numeric_columns.drop(columns=excluded_columns, errors='ignore')
    
    for col in numeric_columns.columns:
        df[f'{col}_Rz'] = df[col].apply(lambda lmbda: Rz(np.radians(lmbda)))
    return df
