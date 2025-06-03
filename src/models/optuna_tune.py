import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import os
import datetime
import json
from src.models.model import build_model_one, build_model_two
from src.data.load_data import load_data_from_csv

def create_optimizer(name, learning_rate):
    """Create optimizer instance based on name and learning rate."""
    if name == "Adam":
        return Adam(learning_rate=learning_rate)
    elif name == "AdamW":
        return AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif name == "RMSprop":
        return RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def objective(trial, model_builder, train_data, train_labels, val_data, val_labels):
    """
    Optuna objective function for optimizing model hyperparameters.
    
    Args:
        trial: Optuna trial object
        model_builder: Function to build the model (build_model_one or build_model_two)
        train_data: Training data
        train_labels: Training labels
        val_data: Validation data
        val_labels: Validation labels
    """
    # Suggest hyperparameters
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Build and compile model
    model = model_builder(dropout=dropout_rate, dense_units=dense_units)
    optimizer = create_optimizer(optimizer_name, learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping callback with shorter patience
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,  # Reduced from 5 to 3
        restore_best_weights=True,
        verbose=0
    )

    # Train the model with fewer epochs
    history = model.fit(
        train_data, train_labels,
        epochs=10,  # Reduced from 20 to 10
        batch_size=batch_size,
        validation_data=(val_data, val_labels),
        callbacks=[early_stopping],
        verbose=0
    )

    # Get best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    
    # Report intermediate values for pruning
    epoch = len(history.history['val_accuracy'])
    trial.report(best_val_acc, epoch)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.TrialPruned()

    return best_val_acc

def tune_model(model_name="model_one", n_trials=100):
    """
    Tune the specified model using Optuna.
    
    Args:
        model_name: Either "model_one" or "model_two"
        n_trials: Number of Optuna trials to run
    """
    print(f"🚀 Starting Optuna optimization for {model_name}")
    print("📥 Loading data...")

    # Load and preprocess data
    data = load_data_from_csv(use_augmentation=False)
    if len(data) == 3:
        train_generator, test_generator, _ = data
        # Convert generators to numpy arrays
        train_data = []
        train_labels = []
        for x_batch, y_batch in train_generator:
            train_data.append(x_batch)
            train_labels.append(y_batch)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
    else:
        train_data, train_labels, _, _ = data

    # Use a smaller subset of data for faster training
    train_size = len(train_data)
    subset_size = min(20000, train_size)  # Use max 20,000 samples for tuning
    indices = np.random.choice(train_size, subset_size, replace=False)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Split training data into train and validation
    val_split = 0.2
    val_size = int(len(train_data) * val_split)
    val_data = train_data[-val_size:]
    val_labels = train_labels[-val_size:]
    train_data = train_data[:-val_size]
    train_labels = train_labels[:-val_size]

    print(f"Using {len(train_data)} samples for training and {len(val_data)} for validation")

    # Select model builder
    if model_name == "model_one":
        model_builder = build_model_one
    else:
        model_builder = build_model_two

    # Create study with pruning
    study_name = f"{model_name}_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=3
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=pruner
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, model_builder, 
            train_data, train_labels,
            val_data, val_labels
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True  # Add garbage collection to free memory
    )

    # Print results
    print("\n🏆 Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    results_dir = os.path.join("outputs", "optuna_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        "study_name": study_name,
        "best_value": trial.value,
        "best_params": trial.params,
        "n_trials": n_trials,
        "datetime": datetime.datetime.now().isoformat()
    }
    
    results_file = os.path.join(results_dir, f"{study_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Results saved to {results_file}")

    # Create and save best model
    print("\n🔨 Creating model with best parameters...")
    best_model = model_builder(
        dropout=trial.params["dropout"],
        dense_units=trial.params["dense_units"]
    )
    optimizer = create_optimizer(
        trial.params["optimizer"],
        trial.params["lr"]
    )
    best_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model_file = os.path.join(results_dir, f"{study_name}_best_model.keras")
    best_model.save(model_file)
    print(f"💾 Best model saved to {model_file}")

if __name__ == "__main__":
    # Let user choose which model to tune
    print("Which model would you like to tune?")
    print("1: model_one (Simple CNN)")
    print("2: model_two (Enhanced CNN)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        model_name = "model_one"
    elif choice == "2":
        model_name = "model_two"
    else:
        print("Invalid choice. Please enter 1 or 2.")
        exit(1)
    
    # Let user specify number of trials
    try:
        n_trials = int(input("Enter number of trials (default: 100): ") or "100")
    except ValueError:
        print("Invalid number of trials. Using default: 100")
        n_trials = 100
    
    # Run tuning
    tune_model(model_name, n_trials) 