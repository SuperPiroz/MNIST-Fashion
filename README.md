# 👗 Fashion-MNIST Neural Network Classifier

A modular deep learning project for classifying clothing images from the Fashion-MNIST dataset using TensorFlow/Keras.  
Developed as part of my AI and machine learning studies.

---

## 📦 Project Overview

This project demonstrates a professional workflow for image classification:
- **Data loading & preprocessing** (with augmentation)
- **Model building** (custom CNN, VGG16, ResNet50)
- **Training & evaluation** (with early stopping, logs, and visualizations)
- **Result analysis** (confusion matrix, classification report, Jupyter notebook)
- **Reproducibility** (modular code, config files, version control)

---

## 🗂️ Project Structure

```
project-root/
│
├── data/           # Raw and processed Fashion-MNIST data (CSV, IDX)
├── notebooks/      # Jupyter notebooks for EDA and result analysis
├── outputs/        # Saved models, logs, plots, evaluation reports
├── src/
│   ├── data/       # Data loading and augmentation scripts
│   ├── models/     # Model definitions, training, tuning
│   ├── evaluation/ # Evaluation and visualization utilities
│   ├── features/   # (Reserved for feature engineering)
│   └── utils.py    # Utility functions
├── requirements.txt
└── README.md
```

---

## 🧠 Dataset

- **Fashion-MNIST** ([Zalando Research](https://github.com/zalandoresearch/fashion-mnist))
- 28x28 grayscale images, 10 clothing classes
- Data files: CSV and IDX formats in `data/`

---

## 🏗️ Models

- **Custom CNNs:**  
  - `build_model_one`: Simple CNN with dropout after pooling layers and dense layer.
  - `build_model_two`: (If present) Variant with different dropout/architecture.
- **VGG16 & ResNet50:**  
  - Adapted for grayscale (images converted to RGB, resized to 32x32).
  - Optionally use ImageNet weights.
- All models defined in `src/models/model.py`.

---

## 🚀 Training & Evaluation

- **Script:** `src/models/train_model.py`
- **Features:**
  - Data augmentation (rotation, shift, flip, zoom)
  - Early stopping (patience=5)
  - TensorBoard logging
  - Saves model, training log, confusion matrix, classification report
  - Plots training/validation accuracy and loss
- **How to use:**  
  Edit the `__main__` block in `train_model.py` to select the model, then run:
  ```bash
  python src/models/train_model.py
  ```

---

## 📊 Results & Analysis

- **Outputs:**  
  - Saved in `outputs/` (models, logs, confusion matrices, reports)
- **Notebook:**  
  - Use `notebooks/eda.ipynb` to explore data and visualize results.
  - Function `show_model_results(model_name)` helps display logs, confusion matrix, and classification report for each model.

---

## ⚙️ Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

**Main packages:**  
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`, `notebook`, `imageio`

---

## 👤 Author

Piroz Kianersi
This project is part of my learning journey in AI and deep learning.  
Feel free to explore, learn, and suggest improvements!

---

## 📜 License

Open-source under the MIT License.

---

**Tip:**  
For faster training, use a machine with an NVIDIA GPU and install the appropriate CUDA-enabled versions of TensorFlow or PyTorch.

---

Let me know if you want to add usage examples, more details about each model, or anything else!






# Fashion MNIST Project

## Note about missing files
Some files and directories are not included in this repository due to their large size (>100MB). These include:
- Data files in the `data/` directory
- Model output files in the `outputs/` directory
- Other large binary files

To run this project, you'll need to download the Fashion MNIST dataset separately.

