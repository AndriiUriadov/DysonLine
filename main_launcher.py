#!/usr/bin/env python3
"""
Головний файл для локального запуску DysonianLineCNN
"""

import sys
import os

# Додаємо поточну директорію до шляху
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_prepare_data
from train import train_and_evaluate
from evaluate import evaluate_model
from utils import print_system_info, create_experiment_log
from config import TRAINING_CONFIG, MODEL_CONFIG, LOSS_WEIGHTS


def main():
    """
    Головна функція для запуску повного циклу навчання
    """
    print("🚀 Запуск DysonianLineCNN...")
    
    # Перевіряємо систему
    print("\n=== Перевірка системи ===")
    system_info = print_system_info()
    
    # Завантажуємо та підготовляємо дані
    print("\n=== Завантаження даних ===")
    data_dict = load_and_prepare_data()
    
    # Навчаємо модель
    print("\n=== Навчання моделі ===")
    model, history = train_and_evaluate(data_dict)
    
    # Оцінюємо модель
    print("\n=== Оцінка моделі ===")
    metrics = evaluate_model(
        model, 
        data_dict['X_test'], 
        data_dict['y_test'], 
        data_dict['y_min'], 
        data_dict['y_max']
    )
    
    # Створюємо лог експерименту
    print("\n=== Збереження результатів ===")
    config_dict = {
        'training_config': TRAINING_CONFIG,
        'model_config': MODEL_CONFIG,
        'loss_weights': LOSS_WEIGHTS,
        'system_info': system_info
    }
    
    create_experiment_log(
        experiment_name="dysonian_line_cnn_local",
        config_dict=config_dict,
        metrics_dict=metrics
    )
    
    print("\n✅ Експеримент завершено успішно!")
    print(f"📊 Результати збережено в лог файлі")
    print(f"💾 Модель збережено як: {model.name}")


if __name__ == "__main__":
    main() 