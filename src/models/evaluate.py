"""
Model Evaluation and Performance Analysis
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import create_data_generators


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str) -> tf.keras.Model:
    """Load a trained model from file."""
    return tf.keras.models.load_model(model_path)


def get_predictions(model: tf.keras.Model, generator) -> tuple:
    """
    Get predictions and ground truth labels from a generator.
    
    Returns:
        Tuple of (predictions, true_labels, filenames)
    """
    predictions = model.predict(generator, verbose=1)
    true_labels = generator.classes
    filenames = generator.filenames
    
    return predictions.flatten(), true_labels, filenames


def plot_confusion_matrix(y_true, y_pred, classes, save_path: str = None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_scores, save_path: str = None):
    """Plot and optionally save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, save_path: str = None):
    """Plot and optionally save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.close()
    
    return pr_auc


def plot_training_history(history_path: str, save_path: str = None):
    """Plot training history from saved JSON file."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train')
    axes[0, 0].plot(history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train')
    axes[0, 1].plot(history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Train')
        axes[1, 0].plot(history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Train')
        axes[1, 1].plot(history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def evaluate_model(config: dict, model_path: str = None, 
                   output_dir: str = 'evaluation_results') -> dict:
    """
    Perform comprehensive model evaluation.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model file
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if model_path is None:
        model_path = config['inference']['model_path']
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Create test generator
    print("Loading test data...")
    _, _, test_gen = create_data_generators(config)
    
    # Get predictions
    print("Generating predictions...")
    y_scores, y_true, filenames = get_predictions(model, test_gen)
    y_pred = (y_scores > config['inference']['threshold']).astype(int)
    
    # Class names
    classes = ['cats', 'dogs']
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(
        y_true, y_pred, classes,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(
        y_true, y_scores,
        save_path=os.path.join(output_dir, 'roc_curve.png')
    )
    
    # Plot Precision-Recall curve
    pr_auc = plot_precision_recall_curve(
        y_true, y_scores,
        save_path=os.path.join(output_dir, 'pr_curve.png')
    )
    
    # Plot training history if available
    history_path = os.path.join(config['training']['final_model_path'], 'training_history.json')
    if os.path.exists(history_path):
        plot_training_history(
            history_path,
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    
    # Compile results
    results = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'total_samples': len(y_true),
        'accuracy': report['accuracy']
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cats vs Dogs Model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory for results")
    args = parser.parse_args()
    
    config = load_config(args.config)
    results = evaluate_model(config, args.model, args.output)
    
    print("\nEvaluation complete!")
    print(f"Overall accuracy: {results['accuracy']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"PR AUC: {results['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
