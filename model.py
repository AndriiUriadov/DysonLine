"""
Модуль з архітектурою моделі DysonianLineCNN
"""

import numpy as np
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import MODEL_CONFIG, LOSS_WEIGHTS


def build_multimodal_model(input_shape=None):
    """
    Створює мультимодальну CNN модель для передбачення параметрів лінії Дайсона
    
    Args:
        input_shape: форма вхідних даних, за замовчуванням з config
        
    Returns:
        Скомпільована модель Keras
    """
    if input_shape is None:
        input_shape = MODEL_CONFIG['input_shape']
    
    inp = Input(shape=input_shape)

    # ─── Гілка 1: CNN для B0, dB, p ───
    x = Conv1D(MODEL_CONFIG['conv1_filters'], MODEL_CONFIG['conv1_kernel'], 
               activation='relu', padding='same')(inp)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(MODEL_CONFIG['conv2_filters'], MODEL_CONFIG['conv2_kernel'], 
               activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(MODEL_CONFIG['conv3_filters'], MODEL_CONFIG['conv3_kernel'], 
               activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    
    x = Flatten()(x)
    shared_dense = Dense(MODEL_CONFIG['shared_dense_units'], activation='relu')(x)

    # Вихідні шари для B0, dB, p
    out_B0 = Dense(MODEL_CONFIG['output_dense_units'], activation='relu')(shared_dense)
    out_B0 = Dense(1, activation='linear', name='B0')(out_B0)

    out_dB = Dense(MODEL_CONFIG['output_dense_units'], activation='relu')(shared_dense)
    out_dB = Dense(1, activation='linear', name='dB')(out_dB)

    out_p = Dense(MODEL_CONFIG['output_dense_units'], activation='relu')(shared_dense)
    out_p = Dense(1, activation='linear', name='p')(out_p)

    # ─── Гілка 2: Autoencoder bottleneck для I ───
    ae = Conv1D(MODEL_CONFIG['conv1_filters'], MODEL_CONFIG['conv1_kernel'], 
                activation='relu', padding='same')(inp)
    ae = MaxPooling1D(2)(ae)
    
    ae = Conv1D(MODEL_CONFIG['conv2_filters'], MODEL_CONFIG['conv2_kernel'], 
                activation='relu', padding='same')(ae)
    ae = MaxPooling1D(2)(ae)
    
    ae = Flatten()(ae)
    ae = Dense(MODEL_CONFIG['ae_dense_units'], activation='relu')(ae)
    ae = Dropout(MODEL_CONFIG['dropout_rate'])(ae)
    out_I = Dense(1, activation='sigmoid', name='I')(ae)

    # Обʼєднана модель
    model = Model(inputs=inp, outputs=[out_B0, out_dB, out_p, out_I])

    # Компіляція моделі
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'B0': 'mse', 'dB': 'mse', 'p': 'mse', 'I': 'mse'},
        loss_weights=LOSS_WEIGHTS,
        metrics={'B0': 'mae', 'dB': 'mae', 'p': 'mae', 'I': 'mae'}
    )

    return model


def get_model_summary():
    """
    Створює модель та виводить її архітектуру
    """
    model = build_multimodal_model()
    model.summary()
    return model 