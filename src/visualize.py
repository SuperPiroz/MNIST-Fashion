import tensorflow as tf
import numpy as np
import os
import imageio

from src.data.load_data import load_data_from_csv

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding.
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / (max + 1e-8)).transpose(3,0,1,2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def create_metadata_file(labels, class_names, path):
    """
    Creates a metadata file for TensorBoard projector.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("Index\tLabel\tClass\n")
        for index, label in enumerate(labels):
            f.write(f"{index}\t{label}\t{class_names[label]}\n")
    return path

def save_embeddings(embeddings, log_dir):
    print(f"Saving embeddings to {log_dir}")
    embedding_var = tf.Variable(embeddings, name='embedding')
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

def setup_tensorboard_projector(images, labels, log_dir="outputs/logs"):
    """
    Sets up TensorBoard projector with embeddings visualization.
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    embedding_dir = os.path.join(log_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Sprite image
    print("Creating sprite image...")
    sprite = images_to_sprite(images)
    sprite_path = os.path.join(embedding_dir, "sprite.png")
    imageio.imwrite(sprite_path, sprite)

    # Metadata
    print("Creating metadata file...")
    metadata_path = os.path.join(embedding_dir, "metadata.tsv")
    create_metadata_file(labels, class_names, metadata_path)
    
    # Embeddings (flatten images)
    print("Creating and saving embeddings...")
    embeddings = images.reshape((images.shape[0], -1))
    save_embeddings(embeddings, embedding_dir)

    print(f"\nAll files for TensorBoard Projector are now in: {embedding_dir}")
    print("Make sure you have a projector_config.pbtxt in the same directory with correct tensor_name, metadata_path, and sprite.image_path.")

def main():
    """
    Main function to set up TensorBoard visualization.
    """
    print("Loading data...")
    # Load data without augmentation for visualization
    x_train, y_train, x_test, y_test = load_data_from_csv(use_augmentation=False)
    
    # Convert grayscale to RGB if needed
    if x_train.shape[-1] == 1:
        x_train = np.repeat(x_train, 3, axis=-1)

    # Use the whole dataset (60,000 images)
    print("Setting up TensorBoard projector for the full dataset (60,000 images)...")
    setup_tensorboard_projector(x_train, y_train)
    
    print("""
    Setup complete! To view the visualizations:
    1. Open a terminal
    2. Navigate to your project directory
    3. Run: tensorboard --logdir outputs/logs/embedding
    4. Open a web browser and go to: http://localhost:6006
    5. Click on the 'Projector' tab
    """)

if __name__ == "__main__":
    main()