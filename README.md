
# 🌿 Plant Leaf Disease Detection Web App

This is a lightweight HTML + Flask web application for detecting plant leaf diseases using a trained CNN model.

## ✅ Features
- Upload a plant leaf image
- Predict disease class using a TensorFlow `.h5` model
- Simple HTML interface

## 🛠 Setup Instructions

1. **Install Dependencies**:
```bash
pip install flask tensorflow
```

2. **Add Dataset Folder**:
Place the `plantvillage_dataset/color/` in the project root.

3. **Train or Place Model**:
Ensure you have `disease_prediction_model.h5` in the project folder.

4. **Run the App**:
```bash
python app.py
```

5. **Open Browser**:
Visit `http://localhost:5000`

## 🧠 To Train Model (Optional)

If needed, use a separate script to train a CNN using the dataset.

---

Enjoy building smarter farms! 🌾
