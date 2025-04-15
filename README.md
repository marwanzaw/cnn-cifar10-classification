# 🧠 CIFAR-10 Image Classification using CNN

This project demonstrates how to build a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images from the popular **CIFAR-10** dataset.

---

## 📦 Dataset
CIFAR-10 consists of 60,000 32x32 color images across 10 classes:
- Airplane ✈️
- Automobile 🚗
- Bird 🐦
- Cat 🐱
- Deer 🦌
- Dog 🐶
- Frog 🐸
- Horse 🐴
- Ship 🚢
- Truck 🚚

---

## 🧰 Model Architecture

```text
Conv2D (32 filters, 3x3) + ReLU
→ MaxPooling2D
→ Conv2D (64 filters, 3x3) + ReLU
→ MaxPooling2D
→ Conv2D (64 filters, 3x3) + ReLU
→ Flatten
→ Dense (64) + ReLU
→ Dropout (0.5)
→ Dense (10) + Softmax
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python cifar10_cnn.py
```

---

## 📊 Sample Results

Training and validation accuracy/loss over epochs:

| Accuracy | Loss |
|----------|------|
| ![Accuracy](results/accuracy_plot.png) | ![Loss](results/loss_plot.png) |

The trained model is saved as `cnn_cifar10_model.h5`.

---

## 📁 File Structure

```
cnn-cifar10-classification/
├── cifar10_cnn.py
├── requirements.txt
├── README.md
├── results/
│   ├── accuracy_plot.png
│   └── loss_plot.png
```

---

## 💬 Author

Created by **Marwan Zaw**  
📧 [alshabah_marwan@yahoo.com]  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/marwan-al-zawahra-67521b70/)

---

## ⭐ If you like this project
Give it a ⭐ and consider following the repo for more ML/AI projects coming soon!
