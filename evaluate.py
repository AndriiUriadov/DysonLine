"""
Модуль для оцінки та візуалізації результатів DysonianLineCNN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from config import LOSS_WEIGHTS


def load_trained_model(model_path, compile_model=True):
    """
    Завантажує навчену модель
    
    Args:
        model_path: шлях до збереженої моделі
        compile_model: чи потрібно перекомпілювати модель
        
    Returns:
        Завантажена модель
    """
    if compile_model:
        # Завантажуємо модель без компіляції
        model = load_model(model_path, compile=False)
        
        # Повторно скомпілюємо модель
        model.compile(
            optimizer='adam',
            loss={'B0': 'mse', 'dB': 'mse', 'p': 'mse', 'I': 'mse'},
            loss_weights=LOSS_WEIGHTS,
            metrics={'B0': 'mae', 'dB': 'mae', 'p': 'mae', 'I': 'mae'}
        )
    else:
        model = load_model(model_path)
    
    return model


def predict_and_denormalize(model, X_test, y_test, y_min, y_max):
    """
    Робить передбачення та денормалізує результати
    
    Args:
        model: навчена модель
        X_test: тестові дані
        y_test: тестові цілі (нормалізовані)
        y_min: мінімальні значення для денормалізації
        y_max: максимальні значення для денормалізації
        
    Returns:
        Денормалізовані передбачення та істинні значення
    """
    # Передбачення
    y_pred_norm = model.predict(X_test[..., np.newaxis])
    y_pred_norm = np.concatenate(y_pred_norm, axis=1)  # форма стане (n_samples, 4)
    
    # Денормалізація
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    y_true = y_test * (y_max - y_min) + y_min
    
    return y_pred, y_true


def calculate_metrics(y_true, y_pred):
    """
    Обчислює метрики для кожного параметра
    
    Args:
        y_true: істинні значення
        y_pred: передбачені значення
        
    Returns:
        Словник з метриками
    """
    param_names = ['B0', 'dB', 'p', 'I']
    metrics = {}
    
    for i, name in enumerate(param_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        
        metrics[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
        print(f'{name}: MAE = {mae:.3f}, MSE = {mse:.4f}, RMSE = {rmse:.3f}')
    
    return metrics


def plot_predictions(y_true, y_pred):
    """
    Візуалізує результати передбачення
    
    Args:
        y_true: істинні значення
        y_pred: передбачені значення
    """
    param_names = ['B0', 'dB', 'p', 'I']
    
    # Побудова графіків "істинне vs передбачене"
    plt.figure(figsize=(12, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        plt.xlabel(f'True {param_names[i]}')
        plt.ylabel(f'Predicted {param_names[i]}')
        plt.title(f'{param_names[i]}: True vs Predicted')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Додаткові графіки розподілу помилок
    plt.figure(figsize=(12, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        errors = y_pred[:, i] - y_true[:, i]
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel(f'Prediction Error ({param_names[i]})')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution for {param_names[i]}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, y_min, y_max):
    """
    Повна оцінка моделі
    
    Args:
        model: навчена модель
        X_test: тестові дані
        y_test: тестові цілі
        y_min: мінімальні значення для денормалізації
        y_max: максимальні значення для денормалізації
        
    Returns:
        Словник з метриками
    """
    print("Оцінка моделі на тестових даних...")
    
    # Передбачення та денормалізація
    y_pred, y_true = predict_and_denormalize(model, X_test, y_test, y_min, y_max)
    
    # Обчислення метрик
    metrics = calculate_metrics(y_true, y_pred)
    
    # Візуалізація
    plot_predictions(y_true, y_pred)
    
    return metrics


def evaluate_from_file(model_path, data_dict):
    """
    Оцінка моделі з файлу
    
    Args:
        model_path: шлях до збереженої моделі
        data_dict: словник з даними
        
    Returns:
        Словник з метриками
    """
    # Завантажуємо модель
    model = load_trained_model(model_path)
    
    # Отримуємо тестові дані
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    y_min = data_dict['y_min']
    y_max = data_dict['y_max']
    
    # Оцінюємо модель
    metrics = evaluate_model(model, X_test, y_test, y_min, y_max)
    
    return metrics 