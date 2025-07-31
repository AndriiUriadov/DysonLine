# DysonianLineCNN

Проєкт для навчання CNN моделі для передбачення параметрів лінії Дайсона.

## Структура проєкту

```
DysonLine-2/
├── config.py              # Конфігурація параметрів
├── data_loader.py         # Завантаження та обробка даних
├── model.py              # Архітектура CNN моделі
├── train.py              # Навчання моделі
├── evaluate.py           # Оцінка та візуалізація
├── utils.py              # Допоміжні функції
├── main_launcher.ipynb   # Notebook для запуску в Colab
├── requirements.txt      # Залежності
└── README.md            # Цей файл
```

## Опис модулів

### `config.py`
Містить всі параметри проєкту:
- URL для завантаження даних
- Параметри моделі (кількість фільтрів, розміри ядер тощо)
- Параметри навчання (batch_size, epochs, learning_rate)
- Ваги втрат для різних параметрів
- Шляхи для збереження

### `data_loader.py`
Функції для:
- Завантаження даних з Google Drive
- Нормалізації вхідних та вихідних даних
- Розділення на train/validation/test
- Підготовки словників цілей для навчання

### `model.py`
Архітектура мультимодальної CNN моделі:
- Гілка 1: CNN для передбачення B0, dB, p
- Гілка 2: Autoencoder bottleneck для I
- Компіляція з відповідними втратами та метриками

### `train.py`
Функції для:
- Створення callbacks (EarlyStopping, ModelCheckpoint)
- Навчання моделі
- Візуалізації історії навчання

### `evaluate.py`
Функції для:
- Завантаження навчених моделей
- Передбачення та денормалізації результатів
- Обчислення метрик (MAE, MSE, RMSE)
- Візуалізації результатів

### `utils.py`
Допоміжні функції:
- Перевірка GPU та RAM
- Робота з Google Drive
- Збереження/завантаження моделей
- Логування експериментів

## Використання

### Локальна розробка

1. Встановіть залежності:
```bash
pip install -r requirements.txt
```

2. Імпортуйте модулі та використовуйте функції:
```python
from data_loader import load_and_prepare_data
from train import train_and_evaluate
from evaluate import evaluate_model

# Завантажуємо дані
data_dict = load_and_prepare_data()

# Навчаємо модель
model, history = train_and_evaluate(data_dict)

# Оцінюємо модель
metrics = evaluate_model(model, data_dict['X_test'], data_dict['y_test'], 
                       data_dict['y_min'], data_dict['y_max'])
```

### Google Colab

Використовуйте `main_launcher.ipynb` для запуску в Google Colab:

1. Завантажте ноутбук в Colab
2. Запустіть всі комірки по порядку
3. Модель буде автоматично збережена на Google Drive

## Параметри моделі

Модель передбачає 4 параметри лінії Дайсона:
- **B0**: базовий параметр
- **dB**: дельта параметр
- **p**: параметр p (з посиленою вагою)
- **I**: параметр I (через autoencoder bottleneck)

## Конфігурація

Всі параметри можна змінити в `config.py`:
- Розміри моделі
- Параметри навчання
- Ваги втрат
- Шляхи до файлів

## Результати

Модель показує наступні метрики на тестових даних:
- B0: MAE ≈ 0.5, MSE ≈ 0.56
- dB: MAE ≈ 2.2, MSE ≈ 8.4
- p: MAE ≈ 0.009, MSE ≈ 0.0004
- I: MAE ≈ 0.000, MSE ≈ 0.0000 