# 15_Classifying_Benign_and_Malignant_Tumors_from_Tabular_Data_using_SVM
Machine Learning project for Build for Bharat Fellowship 2026
# 🧠 Classifying Benign and Malignant Tumors from Tabular Data using SVM

### 🩺 Overview  
This project is a complete **Machine Learning + Flask Web App** that classifies tumors as **benign (non-cancerous)** or **malignant (cancerous)** based on tabular data.  
It uses **Support Vector Machine (SVM)** and **Random Forest** algorithms trained on synthetic tumor data to make predictions.

The main goal is to demonstrate how machine learning can help in early tumor classification using structured numerical data.  
This project was created as part of my **Data Science learning journey** and submitted for the **Build for Bharat Fellowship 2026**.

---

### 🎯 Objective  
The objective of this project is to build and deploy a model that can predict the type of tumor based on patient data.  
The features used are:
- Age  
- Tumor size  
- Texture  
- Smoothness  
- Compactness  

---

### ⚙️ Tech Stack  

**Programming Language:**  
- Python  

**Frameworks & Tools:**  
- Flask (for web app)  
- SQLite (for local database)  

**Libraries Used:**  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Plotly  

---

### 🧩 Features  
- Generates synthetic tumor dataset using NumPy  
- Stores data locally in an SQLite database (`tumors.db`)  
- Trains two ML models: **SVM** and **Random Forest**  
- Displays accuracy and classification metrics  
- Flask-based interactive dashboard  
- Allows uploading custom CSV datasets  
- Offers API for real-time predictions in JSON format  

---

### 🔬 Machine Learning Workflow  
1. **Data Generation:** Creates a synthetic dataset of 2000 records using NumPy.  
2. **Database Creation:** Stores the dataset in `tumors.db`.  
3. **Feature Scaling:** Normalizes data using `StandardScaler`.  
4. **Model Training:** Trains SVM and Random Forest models.  
5. **Evaluation:** Computes accuracy, classification report, and confusion matrix.  
6. **Model Saving:** Saves models and scaler using Joblib for reuse.  

---

### 📈 Model Performance  
Both algorithms perform well on the dataset with consistent accuracy.  
Typical results:  
- **SVM Accuracy:** ~90–92%  
- **Random Forest Accuracy:** ~93–95%  

---

### 💻 How to Run the Project  

#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/15_Classifying_Benign_and_Malignant_Tumors_from_Tabular_Data_using_SVM.git
cd 15_Classifying_Benign_and_Malignant_Tumors_from_Tabular_Data_using_SVM
