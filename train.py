"""
Модуль для навчання моделі DysonianLineCNN
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import TRAINING_CONFIG, PATHS
from model import build_multimodal_model


def create_callbacks():
    """
    Створює callbacks для навчання
    """
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=TRAINING_CONFIG['patience'], 
        restore_best_weights=True, 
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        PATHS['model_save'], 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    return [early_stop, checkpoint]


def train_model(X_train, y_train_dict, X_val, y_val_dict, model=None):
    """
    Навчає модель
    
    Args:
        X_train: тренувальні дані
        y_train_dict: словник цілей для тренування
        X_val: валідаційні дані
        y_val_dict: словник цілей для валідації
        model: модель для навчання (якщо None, створюється нова)
        
    Returns:
        Навчена модель та історія навчання
    """
    if model is None:
        model = build_multimodal_model()
    
    # Створюємо callbacks
    callbacks = create_callbacks()
    
    # Підготовка даних (додаємо розмірність для Conv1D)
    X_train_reshaped = X_train[..., np.newaxis]
    X_val_reshaped = X_val[..., np.newaxis]
    
    print("Початок навчання моделі...")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    
    # Навчання
    history = model.fit(
        X_train_reshaped,
        y_train_dict,
        validation_data=(X_val_reshaped, y_val_dict),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def plot_training_history(history):
    """
    Візуалізує історію навчання
    
    Args:
        history: історія навчання з model.fit()
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Зміна Loss під час навчання')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Графіки для окремих параметрів
    param_names = ['B0', 'dB', 'p', 'I']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, param in enumerate(param_names):
        axes[i].plot(history.history[f'{param}_loss'], label=f'{param} Train Loss')
        axes[i].plot(history.history[f'val_{param}_loss'], label=f'{param} Val Loss')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(f'{param} Loss')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()


def train_and_evaluate(data_dict):
    """
    Повний процес навчання з оцінкою
    
    Args:
        data_dict: словник з даними від data_loader.load_and_prepare_data()
        
    Returns:
        Навчена модель та історія навчання
    """
    # Отримуємо дані
    X_train = data_dict['X_train']
    y_train_dict = data_dict['y_train_dict']
    X_val = data_dict['X_val']
    y_val_dict = data_dict['y_val_dict']
    
    # Навчаємо модель
    model, history = train_model(X_train, y_train_dict, X_val, y_val_dict)
    
    # Візуалізуємо результати
    plot_training_history(history)
    
    return model, history 