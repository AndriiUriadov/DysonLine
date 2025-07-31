"""
Допоміжні функції для DysonianLineCNN
"""

import os
import numpy as np
from google.colab import drive
from config import PATHS


def check_gpu():
    """
    Перевіряє наявність GPU та виводить інформацію про нього
    """
    try:
        import subprocess
        gpu_info = subprocess.check_output(['nvidia-smi']).decode()
        print("GPU Information:")
        print(gpu_info)
        return True
    except:
        print("Not connected to a GPU")
        return False


def check_ram():
    """
    Перевіряє обсяг оперативної пам'яті
    """
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print(f'Your runtime has {ram_gb:.1f} gigabytes of available RAM')
    
    if ram_gb < 20:
        print('Not using a high-RAM runtime')
        return False
    else:
        print('You are using a high-RAM runtime!')
        return True


def mount_google_drive():
    """
    Підключає Google Drive
    """
    if not os.path.ismount('/content/drive'):
        print("🔄 Google Drive не підключено. Підключаємо...")
        drive.mount('/content/drive')
    else:
        print("✅ Google Drive вже підключено.")


def unmount_google_drive():
    """
    Відключає Google Drive
    """
    if os.path.ismount('/content/drive'):
        print("🔄 Google Drive підключено. Відключаємо...")
        drive.flush_and_unmount()
    else:
        print("Google Drive не підключено, нема чого відключати.")


def save_model_to_drive(model, filename=None):
    """
    Зберігає модель на Google Drive
    
    Args:
        model: модель для збереження
        filename: ім'я файлу (за замовчуванням з config)
    """
    if filename is None:
        filename = PATHS['google_drive_path']
    
    mount_google_drive()
    
    # Створюємо директорію якщо не існує
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Зберігаємо модель
    model.save(filename)
    print(f"Модель збережено: {filename}")


def load_model_from_drive(filename=None):
    """
    Завантажує модель з Google Drive
    
    Args:
        filename: ім'я файлу (за замовчуванням з config)
        
    Returns:
        Завантажена модель
    """
    if filename is None:
        filename = PATHS['google_drive_path']
    
    mount_google_drive()
    
    if os.path.exists(filename):
        from tensorflow.keras.models import load_model
        model = load_model(filename)
        print(f"Модель завантажено: {filename}")
        return model
    else:
        print(f"Файл не знайдено: {filename}")
        return None


def save_normalization_params(y_min, y_max, filename='normalization_params.npz'):
    """
    Зберігає параметри нормалізації
    
    Args:
        y_min: мінімальні значення
        y_max: максимальні значення
        filename: ім'я файлу
    """
    np.savez(filename, y_min=y_min, y_max=y_max)
    print(f"Параметри нормалізації збережено: {filename}")


def load_normalization_params(filename='normalization_params.npz'):
    """
    Завантажує параметри нормалізації
    
    Args:
        filename: ім'я файлу
        
    Returns:
        y_min, y_max
    """
    data = np.load(filename)
    y_min = data['y_min']
    y_max = data['y_max']
    print(f"Параметри нормалізації завантажено: {filename}")
    return y_min, y_max


def create_experiment_log(experiment_name, config_dict, metrics_dict=None):
    """
    Створює лог експерименту
    
    Args:
        experiment_name: назва експерименту
        config_dict: словник з конфігурацією
        metrics_dict: словник з метриками (опціонально)
    """
    import json
    from datetime import datetime
    
    log_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config_dict,
        'metrics': metrics_dict
    }
    
    filename = f"experiment_log_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Лог експерименту збережено: {filename}")


def print_system_info():
    """
    Виводить інформацію про систему
    """
    print("=== System Information ===")
    
    # GPU
    gpu_available = check_gpu()
    
    # RAM
    ram_high = check_ram()
    
    # Python packages
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    print("========================")
    
    return {
        'gpu_available': gpu_available,
        'ram_high': ram_high
    } 