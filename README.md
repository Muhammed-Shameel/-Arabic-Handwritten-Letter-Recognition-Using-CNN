# 🇴🇲 Arabic Handwritten Letter Recognition 🧠✍️  

An intelligent deep learning project that recognizes **Arabic handwritten letters** from images and converts them into **machine-readable text**.  
This model uses a **Convolutional Neural Network (CNN)** trained on thousands of handwritten Arabic characters to accurately classify each letter — even from varied handwriting styles.

---

## 🌟 Project Highlights  

🚀 **Deep Learning Model:** Built with TensorFlow/Keras (CNN-based)  
🧾 **Dataset:** Arabic Handwritten Character Dataset (AHCD)  
🖋️ **Purpose:** Convert handwritten Arabic letters to digital text  
📊 **Accuracy:** High test accuracy achieved after training & tuning  
💡 **Applications:** OCR, education tools, document digitization  

---

## 📂 Repository Structure  

Arabic-Letter-Recognition/
│
├── 📓 Arabic_Letter_Recognition.ipynb # Main notebook (training + testing)
│
├── 🧠 model/
│ └── LastmodelV2.h5 # Final trained CNN model
│
│
└── 📄 README.md # Project documentation


---

## 🧰 Tech Stack  

| Category | Tools Used |
|-----------|------------|
| **Language** | Python 3.x |
| **Frameworks** | TensorFlow, Keras |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **Visualization** | Matplotlib |
| **Environment** | Jupyter Notebook / VS Code |

---

## 📊 Dataset Information  

📦 **Name:** Arabic Handwritten Characters Dataset (AHCD)  
🔤 **Classes:** 28 Arabic letters  
🖼️ **Image Size:** 32 × 32 pixels (Grayscale)  
⚙️ **Split:** 80% Training | 20% Testing  
    **Link**: https://www.kaggle.com/datasets/mloey1/ahcd1?select=Arabic+Handwritten+Characters+Dataset+CSV
➡️ *Each folder in the dataset contains images labeled by their respective Arabic letter.*

---

## 🧠 Model Architecture  

> A simple yet powerful **Convolutional Neural Network (CNN)**  

Input (32x32 grayscale)
│
├── Conv2D(32 filters, 3x3) + ReLU + MaxPooling
├── Conv2D(64 filters, 3x3) + ReLU + MaxPooling
│
├── Flatten
├── Dense(128) + Dropout(0.5)
└── Dense(28, activation='softmax')


🧮 **Loss:** Categorical Crossentropy  
⚙️ **Optimizer:** Adam  
📈 **Metric:** Accuracy  

---

## 🧪 Training the Model  

Run the notebook in VS Code or Jupyter:

jupyter notebook Arabic_Letter_Recognition.ipynb
The notebook includes:

Data preprocessing

Model creation

Training and validation

Accuracy/loss visualization

Saving the trained model

After training, the model is saved at:

model/arabic_letter_recognition_final.h5
🖼️ Making Predictions
Load the trained model and test it on a handwritten Arabic letter:


from tensorflow.keras.models import load_model
import cv2, numpy as np

# Load the trained model
model = load_model('model/arabic_letter_recognition_final.h5')

# Load & preprocess image
img = cv2.imread('dataset/test/ain/sample.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))
img = img.reshape(1, 32, 32, 1) / 255.0

# Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
print("Predicted Letter Index:", predicted_class)
💡 You can map the index to the actual Arabic letter using your label dictionary.

🔐 Data Privacy
All image data is processed locally — no external API calls.

Ideal for offline or educational use.

Can be safely deployed in secure environments.

🌍 Future Enhancements
✅ Extend to full-word recognition (Arabic OCR)
✅ Add Tashkeel (diacritic) support
✅ Deploy with Streamlit/Flask for live demo
✅ Use data augmentation to improve robustness

🤝 Contributing
Contributions are welcome!
If you’d like to improve the model, dataset, or interface:

Fork this repo 🍴

Create a new branch:


git checkout -b feature-improvement
Commit your changes and open a PR ✨

🧑‍💻 Author
Muhammed Shameel
Machine Learning & Deep Learning Engineer
📍 Kerala, India

📧 muhammedshameel3009@gmail.com
