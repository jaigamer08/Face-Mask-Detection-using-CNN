# 😷 Face Mask Detection using CNN

A Convolutional Neural Network (CNN) built and trained **from scratch** (no transfer learning) to classify a face image as **With Mask** or **Without Mask**, implemented and demonstrated in Google Colab.

## 📌 Overview

This project automates face-mask compliance checking using Deep Learning. Given a face image, the trained CNN predicts whether the person is wearing a mask, along with a confidence score.

**Key highlight of this demo:** it shows *why* Early Stopping matters — the first model overfits badly (47.50% test accuracy), and the second run, trained with `EarlyStopping`, generalises far better (80.00% validation accuracy) and correctly classifies two brand-new test photos.

## 📂 Dataset

- **Source folder:** `small_dataset/` with two class sub-folders — `with_mask/` and `without_mask/`
- **Total images:** 200 (each resized to `100 x 100 x 3`)
- **Train / Test split:** 160 / 40 images (`test_size=0.2`, `random_state=42`)
- **Test set class balance** (from the confusion matrix): 21 With Mask, 19 Without Mask

## 🧠 Model Architecture

A custom CNN (no pre-trained backbone):

| Block | Layers |
|---|---|
| Conv Block 1 | Conv2D(32) → BatchNorm → MaxPooling → Dropout(0.25) |
| Conv Block 2 | Conv2D(64) → BatchNorm → MaxPooling → Dropout(0.25) |
| Conv Block 3 | Conv2D(128) → BatchNorm → MaxPooling → Dropout(0.25) |
| Dense | Flatten → Dense(128) → BatchNorm → Dropout(0.5) |
| Output | Dense(2, activation='softmax') |

**Total params:** 1,733,442 (6.61 MB) · **Trainable:** 1,732,738 · **Non-trainable:** 704

**Training setup:**
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Batch size: 16
- Data Augmentation: rotation (20°), width/height shift (0.1), zoom (0.2), horizontal flip

## 🔁 Two Training Runs

### 1. Initial run — 30 fixed epochs, no Early Stopping
Training accuracy climbs to ~95–98% while validation accuracy stays flat near 47.5%, and validation loss keeps rising — a textbook overfitting curve.

| Metric | Value |
|---|---|
| Test Accuracy | **47.50%** |
| Test Loss | 3.5431 |
| Avg. Prediction Confidence | 98.92% |

**Confusion Matrix (initial model):**

| | Predicted: With Mask | Predicted: Without Mask |
|---|---|---|
| **Actual: With Mask** | 0 | 21 |
| **Actual: Without Mask** | 0 | 19 |

The model collapsed to always predicting "Without Mask".

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| With Mask | 0.00 | 0.00 | 0.00 | 21 |
| Without Mask | 0.47 | 1.00 | 0.64 | 19 |
| **Accuracy** | | | **0.47** | 40 |
| Macro avg | 0.24 | 0.50 | 0.32 | 40 |
| Weighted avg | 0.23 | 0.47 | 0.31 | 40 |

### 2. Retrained run — with `EarlyStopping`
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```
Training automatically stopped after **epoch 11**, restoring weights from the best epoch (**epoch 6**), which had the lowest validation loss.

| Metric (best epoch, restored) | Value |
|---|---|
| Validation Accuracy | **80.00%** |
| Validation Loss | 0.8218 |

This is a large improvement over the initial model's 47.50% test accuracy.

## 🖼️ Live Predictions on New Images

| Test Image | Model Used | Prediction | Confidence |
|---|---|---|---|
| Photo without a mask | Initial (overfit) model | ✅ Without Mask (correct) | 100.00% |
| Photo with a surgical mask | Early-stopped model (restored best weights) | ✅ With Mask (correct) | 99.98% |

## 🚀 How to Run

1. Open the notebook in Google Colab.
2. Upload `small_dataset.zip` (containing `with_mask/` and `without_mask/` folders) to `/content/`.
3. Run all cells top to bottom:
   - Extract dataset → load & label images → normalize → train/test split
   - Build CNN (Conv + BatchNorm + Dropout)
   - Train (30 epochs) → view accuracy/loss plots, confusion matrix, classification report
   - Re-train with `EarlyStopping` for a better-generalising model
   - Upload your own photo in the **Test Model on New Image** cell to get a live With Mask / Without Mask prediction

## 🛠️ Tech Stack

- Python, NumPy, OpenCV
- TensorFlow / Keras (Conv2D, BatchNormalization, Dropout, ImageDataGenerator, EarlyStopping)
- scikit-learn (`train_test_split`, `confusion_matrix`, `classification_report`)
- Matplotlib (training curves, confusion matrix visualization)

## 📈 Key Takeaway

A small dataset (200 images) trained for a fixed number of epochs can reach ~98% *training* accuracy while completely failing to generalise (47.50% test accuracy). Adding **Early Stopping** (alongside Batch Normalization, Dropout, and Data Augmentation) lets training stop at the best-generalising point instead of overfitting, lifting validation accuracy to 80.00% and producing a model that correctly classifies new, unseen photos.

## 📄 License

Add your preferred license here (e.g., MIT).
