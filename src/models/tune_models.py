import os
import numpy as np
import datetime
from tensorflow import keras
from keras_tuner import HyperModel, RandomSearch
from src.models.model import build_model_one
from src.data.load_data import load_data_from_csv

# Define a HyperModel for CNN architecture
class CNNHyperModel(HyperModel):
    def __init__(self):
        super().__init__()

    def build(self, hp):
        # Tune dropout and dense units
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=64)
        
        return build_model_one(dropout=dropout, dense_units=dense_units)

# Prepare data
print("📥 Loading training data...")
train_generator, test_generator, y_true = load_data_from_csv(use_augmentation=True)
print("✅ Data loaded and formatted with augmentation!")

print("\nTuning CNN model...")
tuner = RandomSearch(
    CNNHyperModel(),
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='outputs/tuner_logs',
    project_name='cnn_tuning_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)

tuner.search(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]
print(f"Best hyperparameters for CNN: {best_hp.values}")

# Save best model
best_model.save("outputs/cnn_tuned.keras")
print("Best model saved as 'cnn_tuned.keras'") 