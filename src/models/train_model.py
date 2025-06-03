# In this file i will train the model and evaluate it
import os
# Set environment variables to suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from src.models.model import (
    build_model_one, build_model_two,
    get_callbacks_model_two, get_data_augmentation_model_two
)
from src.data.load_data import load_data_from_csv, get_class_weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime

def train_and_evaluate(model_fn, model_name, epochs=50, batch_size=64, use_custom_training=False):
    """
    Trains and evaluates the given model function.
    Args:
        model_fn: Function to build the model
        model_name: Name of the model for logging
        epochs: Number of epochs to train
        batch_size: Batch size for training
        use_custom_training: Whether to use custom training configuration
    """
    # Load data with specialized augmentation for shirts
    print("\n🔄 Loading data with specialized shirt augmentation...")
    train_generator, test_generator, y_true, steps_per_epoch = load_data_from_csv(
        use_augmentation=True,
        batch_size=batch_size
    )
    
    print("\nUsing enhanced training configuration:")
    print("- Specialized augmentation for shirts")
    print("- Weighted sampling (shirts appear twice as often)")
    print(f"- Batch size: {batch_size}")

    # Build the model
    model = model_fn()
    print(f"\nModel Architecture Summary for {model_name}:")
    model.summary()

    # Setup callbacks
    callbacks = []
    
    # TensorBoard callback
    tb_log_dir = os.path.join("outputs", "logs", model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks.append(TensorBoard(log_dir=tb_log_dir, histogram_freq=1))
    
    if use_custom_training:
        # Add custom callbacks for model_two
        callbacks.extend(get_callbacks_model_two())
    else:
        # Standard early stopping for model_one
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ))

    print("\n🚀 Starting training with enhanced shirt classification...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{model_name}.txt")
    with open(log_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("Using specialized shirt augmentation and class weights\n")
        for key in history.history:
            f.write(f"{key}: {history.history[key]}\n")
    print(f"Training log saved to {log_path}")

    # Plot training history
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"training_history_{model_name}.png"))
    plt.show()

    # Evaluate on test set
    print("\n📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    y_pred = model.predict(test_generator)
    
    print(f"\nTest accuracy: {test_acc:.2f}")

    # Save the model
    model_save_path = os.path.join(log_dir, f"{model_name}.keras")
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    # Generate confusion matrix
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(log_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path)
    plt.show()

    # Generate classification report
    report = classification_report(y_true, y_pred_classes)
    print("\nClassification Report:")
    print(report)
    report_path = os.path.join(log_dir, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    # Configuration for model_two with enhanced shirt classification
    model_fn = build_model_two
    model_name = "model_two_shirt_enhanced"
    batch_size = 64  # Optimized for CPU training
    use_custom_training = True  # Use model_two specific callbacks
    
    print("\nTraining model_two with enhanced shirt classification")
    print("Using configuration:")
    print("- Enhanced augmentation for shirt class")
    print("- Weighted sampling (shirts appear more frequently)")
    print("- Batch size: 64 (optimized for CPU)")
    print("- Dropout: 0.378")
    print("- Dense units: 128")
    print("- Optimizer: RMSprop with lr=0.000543")
    
    train_and_evaluate(
        model_fn=model_fn,
        model_name=model_name,
        epochs=50,
        batch_size=batch_size,
        use_custom_training=use_custom_training
    )