"""
Модуль для завантаження та обробки даних DysonianLineCNN
"""

import gdown
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import DATA_URLS, DATA_FILES, TRAINING_CONFIG, NORMALIZATION


def download_data():
    """
    Завантажує дані з Google Drive
    """
    print("Завантаження даних...")
    
    # Завантажуємо файли
    gdown.download(DATA_URLS['X'], DATA_FILES['X'], quiet=False)
    gdown.download(DATA_URLS['y'], DATA_FILES['y'], quiet=False)
    
    # Завантажуємо у змінні
    X = np.load(DATA_FILES['X'])
    y = np.load(DATA_FILES['y'])
    
    print("Форми масивів:")
    print("X.shape =", X.shape)
    print("y.shape =", y.shape)
    
    return X, y


def normalize_data(X, y):
    """
    Нормалізує дані X та y
    """
    print("Нормалізація даних...")
    
    # Нормалізація вхідних даних X (z-score)
    if NORMALIZATION['X_method'] == 'z_score':
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_normalized = (X - X_mean) / X_std
    else:
        X_normalized = X
    
    # Нормалізація вихідних даних y (min-max)
    if NORMALIZATION['y_method'] == 'minmax':
        scaler_y = MinMaxScaler()
        y_normalized = scaler_y.fit_transform(y)
        
        # Збережемо параметри для денормалізації
        y_min = scaler_y.data_min_
        y_max = scaler_y.data_max_
        
        print("Мінімальні значення y:", y_min)
        print("Максимальні значення y:", y_max)
    else:
        y_normalized = y
        y_min = None
        y_max = None
    
    return X_normalized, y_normalized, y_min, y_max


def split_data(X, y):
    """
    Розділяє дані на train/validation/test
    """
    print("Розділення даних на train/val/test...")
    
    # Розділення на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=TRAINING_CONFIG['test_size'], 
        random_state=TRAINING_CONFIG['random_state']
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=TRAINING_CONFIG['val_size'], 
        random_state=TRAINING_CONFIG['random_state']
    )
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_target_dicts(y_train, y_val):
    """
    Підготовка словників цілей для навчання
    """
    y_train_dict = {
        'B0': y_train[:, 0],
        'dB': y_train[:, 1],
        'p':  y_train[:, 2],
        'I':  y_train[:, 3]
    }
    
    y_val_dict = {
        'B0': y_val[:, 0],
        'dB': y_val[:, 1],
        'p':  y_val[:, 2],
        'I':  y_val[:, 3]
    }
    
    return y_train_dict, y_val_dict


def load_and_prepare_data():
    """
    Основний функція для завантаження та підготовки всіх даних
    """
    # Завантаження
    X, y = download_data()
    
    # Нормалізація
    X_normalized, y_normalized, y_min, y_max = normalize_data(X, y)
    
    # Розділення
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_normalized, y_normalized)
    
    # Підготовка словників
    y_train_dict, y_val_dict = prepare_target_dicts(y_train, y_val)
    
    print('Min y: ', np.min(y_train, axis=0))  # має бути ~0
    print('Max y: ', np.max(y_train, axis=0))  # має бути ~1
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_train_dict': y_train_dict,
        'y_val_dict': y_val_dict,
        'y_min': y_min,
        'y_max': y_max
    } 