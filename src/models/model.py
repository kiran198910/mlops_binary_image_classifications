"""
Model Architectures for Cats vs Dogs Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_custom_cnn(input_shape: tuple, num_classes: int = 2) -> Model:
    """
    Build a custom CNN architecture for image classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fifth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def build_mobilenet_model(input_shape: tuple, num_classes: int = 2) -> Model:
    """
    Build a MobileNetV2-based model with transfer learning.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def build_resnet_model(input_shape: tuple, num_classes: int = 2) -> Model:
    """
    Build a ResNet50V2-based model with transfer learning.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50V2
    base_model = ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def get_model(config: dict) -> Model:
    """
    Get the model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Keras model
    """
    architecture = config['model']['architecture']
    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    
    if architecture == 'custom_cnn':
        model = build_custom_cnn(input_shape, num_classes)
    elif architecture == 'mobilenet':
        model = build_mobilenet_model(input_shape, num_classes)
    elif architecture == 'resnet50':
        model = build_resnet_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def compile_model(model: Model, config: dict) -> Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model
        config: Configuration dictionary
        
    Returns:
        Compiled model
    """
    learning_rate = config['training']['learning_rate']
    optimizer_name = config['training']['optimizer']
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def unfreeze_model(model: Model, num_layers_to_unfreeze: int = 20) -> Model:
    """
    Unfreeze the last N layers of a model for fine-tuning.
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
        
    Returns:
        Model with unfrozen layers
    """
    # Unfreeze the base model
    if hasattr(model.layers[0], 'trainable'):
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the last N
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
    
    return model


if __name__ == "__main__":
    # Test model building
    config = load_config()
    
    model = get_model(config)
    model = compile_model(model, config)
    
    print(model.summary())
