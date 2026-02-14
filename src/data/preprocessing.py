"""
Data Preprocessing Module for Cats vs Dogs Dataset
"""

import os
import yaml
import numpy as np
from typing import Tuple, Generator
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_image(image_path: str) -> bool:
    """Verify that an image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clean_corrupted_images(data_dir: str) -> int:
    """Remove corrupted image files from the dataset."""
    removed_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                if not verify_image(file_path):
                    print(f"Removing corrupted image: {file_path}")
                    os.remove(file_path)
                    removed_count += 1
    
    print(f"Removed {removed_count} corrupted images")
    return removed_count


def create_data_generators(config: dict, augment_train: bool = True) -> Tuple:
    """
    Create data generators for training, validation, and test sets.
    
    Args:
        config: Configuration dictionary
        augment_train: Whether to apply data augmentation to training data
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    
    # Training data generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Validation and test generators (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config['data']['train_path'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        config['data']['validation_path'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        config['data']['test_path'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def create_tf_dataset(data_dir: str, config: dict, 
                       is_training: bool = False) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset for efficient data loading.
    
    Args:
        data_dir: Directory containing the data
        config: Configuration dictionary
        is_training: Whether this is training data (for augmentation)
        
    Returns:
        tf.data.Dataset object
    """
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    
    def parse_image(file_path):
        """Parse a single image file."""
        # Read file
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = img / 255.0
        
        # Get label from path
        parts = tf.strings.split(file_path, os.sep)
        label = tf.cast(parts[-2] == 'dogs', tf.float32)
        
        return img, label
    
    def augment(image, label):
        """Apply data augmentation."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label
    
    # Get all image paths
    dataset = tf.data.Dataset.list_files(f"{data_dir}/*/*")
    
    # Parse images
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation for training
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def preprocess_single_image(image_path: str, config: dict) -> np.ndarray:
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        config: Configuration dictionary
        
    Returns:
        Preprocessed image array
    """
    image_size = config['data']['image_size']
    
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_image_bytes(image_bytes: bytes, config: dict) -> np.ndarray:
    """
    Preprocess image from bytes for API inference.
    
    Args:
        image_bytes: Raw image bytes
        config: Configuration dictionary
        
    Returns:
        Preprocessed image array
    """
    import io
    
    image_size = config['data']['image_size']
    
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


if __name__ == "__main__":
    # Test preprocessing
    config = load_config()
    
    # Clean corrupted images
    clean_corrupted_images(config['data']['processed_data_path'])
    
    # Create generators
    train_gen, val_gen, test_gen = create_data_generators(config)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")
