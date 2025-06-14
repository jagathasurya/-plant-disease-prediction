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
Enjoy building smarter farms! 🌾

output:

![Screenshot 2025-06-14 132344](https://github.com/user-attachments/assets/960d2f4c-3ede-453c-bf05-d1294dbbb390)

The web page of the plant disease detector

![Screenshot 2025-06-14 132404](https://github.com/user-attachments/assets/5e6ec61f-1a9d-48b4-abb7-7f091bdf5073)

Browsing the file

![Screenshot 2025-06-14 132418](https://github.com/user-attachments/assets/cb6d6e91-8b4f-4724-8930-cacf23f73dca)

prediction of the pepper bell bacterial spot

![Screenshot 2025-06-14 132502](https://github.com/user-attachments/assets/ec4b80e9-65cc-4f70-8a7c-8df0ffec3e7c)

Browsing the file

![Screenshot 2025-06-14 132517](https://github.com/user-attachments/assets/5d1bdbc4-ab42-4036-b8b9-2f0608e55907)

prediction of the tomato septorial leaf spot
