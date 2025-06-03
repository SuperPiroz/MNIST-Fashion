# src/visualize.py

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import array_to_img
from PIL import Image
from src.data.load_data import load_data_from_csv

def create_sprite_image(images, path="outputs/sprite.png"):
    """
    Creates a sprite image from a set of images.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Calculate the square grid size
    n = int(np.ceil(np.sqrt(images.shape[0])))
    
    # Create a blank canvas for the sprite image
    sprite = Image.new(
        mode='L',
        size=(28*n, 28*n),
        color='white'
    )
    
    # Populate the sprite image with individual images
    for index, image in enumerate(images):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Ensure shape (28,28,1)
        img = array_to_img(image)
        sprite.paste(img, (28 * (index % n), 28 * (index // n)))
    
    sprite.save(path)
    return path

def create_metadata_file(labels, class_names, path="outputs/metadata.tsv"):
    """
    Creates a metadata file for TensorBoard projector.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        f.write("Index\tLabel\tClass\n")
        for index, label in enumerate(labels):
            f.write(f"{index}\t{label}\t{class_names[label]}\n")
    return path

def setup_tensorboard_projector(images, labels, log_dir="outputs/logs"):
    """
    Sets up TensorBoard projector with embeddings visualization.
    """
    # Fashion MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Create necessary directories
    os.makedirs(log_dir, exist_ok=True)
    
    # Create sprite image and metadata file
    sprite_path = create_sprite_image(images, path=os.path.join(log_dir, "embedding", "sprite.png"))
    metadata_path = create_metadata_file(labels, class_names, path=os.path.join(log_dir, "embedding", "metadata.tsv"))
    
    # Set up a logs directory
    embedding_dir = os.path.join(log_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Create embeddings (flatten images)
    embeddings = images.reshape((images.shape[0], -1))
    
    # Save the embeddings as a TensorFlow checkpoint
    save_embeddings(embeddings, embedding_dir)
    
    print(f"\nAll files for TensorBoard Projector are now in: {embedding_dir}")
    print("Make sure you have a projector_config.pbtxt in the same directory with correct tensor_name, metadata_path, and sprite.image_path.")

def save_embeddings(embeddings, log_dir):
    print(f"Saving embeddings to {log_dir}")
    embedding_var = tf.Variable(embeddings, name='embedding')
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

def main():
    """
    Main function to set up TensorBoard visualization.
    """
    print("Loading data...")
    # Load data without augmentation for visualization
    x_train, y_train, x_test, y_test = load_data_from_csv(use_augmentation=False)
    
    print("Setting up TensorBoard projector...")
    setup_tensorboard_projector(x_train[:5000], y_train[:5000])  # Using first 5000 images
    
    print("""
    Setup complete! To view the visualizations:
    1. Open a terminal
    2. Navigate to your project directory
    3. Run: tensorboard --logdir outputs/logs
    4. Open a web browser and go to: http://localhost:6006
    5. Click on the 'Projector' tab
    """)

if __name__ == "__main__":
    main()