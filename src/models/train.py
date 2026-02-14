"""
Training Script for Cats vs Dogs Classifier
With MLflow experiment tracking
"""

import os
import sys
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import mlflow
import mlflow.keras

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import create_data_generators
from models.model import get_model, compile_model


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict) -> str:
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    return config['mlflow']['experiment_name']


def get_callbacks(config: dict) -> list:
    """Create training callbacks."""
    checkpoint_dir = config['training']['model_checkpoint_path']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['training']['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(config: dict, use_mlflow: bool = True) -> dict:
    """
    Train the model with the specified configuration.
    
    Args:
        config: Configuration dictionary
        use_mlflow: Whether to use MLflow for tracking
        
    Returns:
        Dictionary containing training history and metrics
    """
    # Setup directories
    final_model_dir = config['training']['final_model_path']
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Setup MLflow
    if use_mlflow:
        setup_mlflow(config)
    
    # Create data generators
    print("Loading data...")
    train_gen, val_gen, test_gen = create_data_generators(config)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    # Get and compile model
    print("Building model...")
    model = get_model(config)
    model = compile_model(model, config)
    
    print(model.summary())
    
    # Get callbacks
    callbacks = get_callbacks(config)
    
    # Start MLflow run
    if use_mlflow:
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            'model_architecture': config['model']['architecture'],
            'input_shape': str(config['model']['input_shape']),
            'epochs': config['training']['epochs'],
            'learning_rate': config['training']['learning_rate'],
            'optimizer': config['training']['optimizer'],
            'batch_size': config['data']['batch_size'],
            'image_size': config['data']['image_size'],
            'train_samples': train_gen.samples,
            'val_samples': val_gen.samples
        })
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=config['training']['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_gen, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    
    # Helper function to get metric by name patterns
    def get_metric(metrics_dict, *patterns):
        """Get metric value by trying multiple possible key patterns."""
        for pattern in patterns:
            for key in metrics_dict:
                if pattern in key.lower():
                    return metrics_dict[key]
        return 0.0
    
    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Extract metrics with flexible key matching
    test_accuracy = get_metric(test_metrics, 'accuracy', 'acc', 'compile_metrics')
    test_precision = get_metric(test_metrics, 'precision')
    test_recall = get_metric(test_metrics, 'recall')
    test_auc = get_metric(test_metrics, 'auc')
    
    # Log metrics to MLflow
    if use_mlflow:
        # Log final metrics
        mlflow.log_metrics({
            'test_loss': test_metrics.get('loss', 0.0),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'best_val_accuracy': max(history.history.get('val_accuracy', history.history.get('val_compile_metrics', [0]))),
            'final_train_accuracy': history.history.get('accuracy', history.history.get('compile_metrics', [0]))[-1]
        })
        
        # Log model
        mlflow.keras.log_model(model, "model")
        
        mlflow.end_run()
    
    # Save final model
    final_model_path = os.path.join(final_model_dir, 'cats_dogs_model.h5')
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(final_model_dir, 'training_history.json')
    history_dict = {key: [float(v) for v in values] 
                    for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(final_model_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    
    return {
        'history': history.history,
        'test_metrics': test_metrics,
        'model_path': final_model_path
    }


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs Classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting error: {e}")
    
    # Train model
    results = train_model(config, use_mlflow=not args.no_mlflow)
    
    # Helper to get metric from history with fallbacks
    def get_history_metric(history, *keys):
        for key in keys:
            if key in history:
                return history[key]
        return [0.0]
    
    print("\nTraining complete!")
    val_acc = get_history_metric(results['history'], 'val_accuracy', 'val_compile_metrics', 'val_acc')
    print(f"Best validation accuracy: {max(val_acc):.4f}")
    
    # Get test accuracy from metrics
    test_acc = 0.0
    for key in results['test_metrics']:
        if 'accuracy' in key.lower() or 'acc' in key.lower() or 'compile_metrics' in key.lower():
            test_acc = results['test_metrics'][key]
            break
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
