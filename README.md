# 🧠 Credit Score Predictor

A Flask web application that predicts a customer's credit score — **Good**, **Standard**, or **Bad** — based on financial and behavioral data using a machine learning model.

---

## 🚀 Features

- Predicts credit score using a trained Random Forest Classifier
- Accepts inputs via form or raw JSON
- Preprocessing includes scaling and label encoding
- Outputs prediction with explanation and confidence score
- Clean web UI using Flask + HTML + Jinja2

---

## 📊 Dataset Source

This dataset is obtained from [https://www.kaggle.com/datasets/parisrohan/credit-score-classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

---

## 📥 Model Files

Download the necessary model files from the link below and place them in your project root:

- `credit_score_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`

🔗 [Download Model Files](https://your-download-link.com)

---

## 🏗️ Tech Stack

- Python 3.11
- Flask
- Scikit-learn
- Pandas, NumPy
- Joblib (model persistence)
- HTML (Jinja2 templating)

---

## 📦 Project Structure

├── app.py # Flask backend
├── templates/
│ └── index.html # HTML UI
├── credit_score_model.pkl # Trained ML model
├── scaler.pkl # Scaler for numeric features
├── label_encoders.pkl # Encoders for categorical features
├── requirements.txt # Python dependencies
└── README.md


---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Score-Predictor.git
cd Credit-Score-Predictor

### 2. Create virtual environment

python -m venv venv
source venv/bin/activate        # For Linux/macOS
venv\Scripts\activate           # For Windows

### 2. Install dependencies
pip install -r requirements.txt

### 2.  Run the Flask app
python app.py
Visit: http://127.0.0.1:5000

📥 Download the `scaler.pkl`, `credit_score_model.pkl`, and `label_encoders.pkl` files from this link: [MODEL DOWNLOAD LINK](https://drive.google.com/drive/folders/1PIqq1qhPNSAO-xJcPHa2iOWdJhmA1iHK?usp=sharing)



