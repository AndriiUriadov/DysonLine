"""
Конфігурація проєкту DysonianLineCNN
"""

# Шляхи до даних
DATA_URLS = {
    'X': 'https://drive.google.com/uc?id=1kOeVd4d1PZfPhfoVIPKXSUScV0tfiRcD',
    'y': 'https://drive.google.com/uc?id=1LKHYyAnb3Ls1qKbxlXOvc6mUY_fMSiAk'
}

DATA_FILES = {
    'X': 'X_dyson.npy',
    'y': 'y_dyson.npy'
}

# Параметри моделі
MODEL_CONFIG = {
    'input_shape': (4096, 1),
    'conv1_filters': 32,
    'conv1_kernel': 7,
    'conv2_filters': 64,
    'conv2_kernel': 5,
    'conv3_filters': 128,
    'conv3_kernel': 3,
    'shared_dense_units': 256,
    'output_dense_units': 64,
    'ae_dense_units': 64,
    'dropout_rate': 0.3
}

# Параметри навчання
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 300,
    'learning_rate': 0.001,
    'patience': 20,
    'test_size': 0.3,
    'val_size': 0.5,
    'random_state': 42
}

# Параметри втрат
LOSS_WEIGHTS = {
    'B0': 1.0,
    'dB': 1.0,
    'p': 2.0,  # посилена вага для p
    'I': 1.0
}

# Шляхи для збереження
PATHS = {
    'model_save': 'best_cnn_model.h5',
    'model_save_keras': 'best_cnn_model.keras',
    'google_drive_path': '/content/drive/MyDrive/Python/DysonianLineCNN-multihead/multihead_cnn_model_30K.h5'
}

# Параметри нормалізації
NORMALIZATION = {
    'X_method': 'z_score',  # z-score нормалізація для X
    'y_method': 'minmax'    # min-max нормалізація для y
} 