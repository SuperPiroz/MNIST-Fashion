# Använd plot_model() för att exportera modellen visuellt


import os
from tensorflow.keras.utils import plot_model

def save_model_architecture(model, filename='model_architecture.png', output_dir='outputs'):
    """
    Saves the architecture of a Keras model as a PNG image.

    Args:
        model (tf.keras.Model): The Keras model to visualize.
        filename (str): The name of the PNG file to save.
        output_dir (str): The directory where the image will be saved.

    Returns:
        str: The path to the saved image file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Full path for the output image
    file_path = os.path.join(output_dir, filename)

    # Save the model architecture as a PNG image
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved to: {file_path}")

    return file_path

# Example usage:
if __name__ == "__main__":
    from model import build_model  # Adjust the import path as needed
    model = build_model()
    save_model_architecture(model)