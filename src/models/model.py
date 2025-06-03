#  Modular function (build_model) – Easy to reuse in train.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation,
    Input
)
from tensorflow.keras.optimizers import RMSprop, AdamW, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model_one(input_shape=(28, 28, 1), num_classes=10, dropout=0.35, dense_units=256):
    """
    Optimized CNN based on Optuna results:
    - Dropout: 0.35 (from 0.3)
    - Dense units: 256 (optimal from trials)
    - Optimizer: RMSprop with lr=0.00066
    - Batch size: 64 (handled in training)
    
    Best validation accuracy: 90.57%
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 1st convolutional block
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.15))

    # 2nd convolutional block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.15))

    # 3rd convolutional block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))  # Added MaxPooling here
    model.add(Dropout(0.2))  # Slightly higher dropout

    # Flatten the feature maps
    model.add(Flatten())

    # Fully connected layer with optimized dropout
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())  # Added BatchNorm here
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Optimized RMSprop configuration
    optimizer = RMSprop(
        learning_rate=0.00066  # Optimal learning rate from Optuna
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_model_two(input_shape=(28, 28, 1), num_classes=10, dropout=0.378, dense_units=128):
    """
    Enhanced CNN with Optuna-optimized hyperparameters:
    - Dropout: 0.378 (optimized from 0.3)
    - Dense units: 128 (reduced from 256 for better generalization)
    - Optimizer: RMSprop with lr=0.000543
    - Batch size: 32 (handled in training)
    
    Best validation accuracy: 91.28%
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 1st convolutional block
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.15))

    # 2nd convolutional block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.15))

    # 3rd convolutional block with additional pooling
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # 4th convolutional block
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Slightly increased dropout

    # Flatten the feature maps
    model.add(Flatten())

    # Fully connected layers with optimized configuration
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Optimized RMSprop configuration
    optimizer = RMSprop(
        learning_rate=0.000543  # Optimal learning rate from Optuna
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_callbacks_model_two():
    """
    Returns the callbacks for model_two:
    - ReduceLROnPlateau for learning rate scheduling
    - EarlyStopping with best weights restoration
    """
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    return callbacks

def get_data_augmentation_model_two():
    """
    Returns the data augmentation configuration for model_two
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )