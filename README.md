# 🔍 Intelligent Hidden Compartment Detection System

## 📌 Overview
This project presents a real-time embedded system for detecting hidden compartments using **Ultrasonic sensors and mmWave Radar**. 

By combining **sensor fusion** with **machine learning models**, the system accurately classifies objects as **Normal** or **Suspicious**, making it suitable for logistics, warehouse inspection, and security screening.

---

## 🎯 Objectives
- Detect hidden compartments in real time  
- Improve accuracy using multi-sensor fusion  
- Reduce manual inspection effort  
- Enable low-cost embedded deployment using Raspberry Pi  

---

## 🏗️ System Architecture
![Architecture](images/workflow.png)

---

## 🔄 Workflow
1. Sensors collect depth and reflectivity data  
2. Data is preprocessed (scaling & normalization)  
3. Features are passed to ML models  
4. Predictions are generated  
5. Final classification: **Normal / Suspicious**  

---

## 🧠 Machine Learning Models
- **Random Forest**
  - Handles noisy data effectively  
- **XGBoost**
  - Captures complex feature interactions  
- **Ensemble Model (Weighted)**
  - Combines both models for improved accuracy  

---

## 📊 Features Used
- Ultrasonic Depth Measurements  
- mmWave Radar Reflectivity (NEAR, MID values)  
- Multi-angle sensor readings  

---

## ⚙️ Tech Stack
- **Hardware:** Raspberry Pi 4, HC-SR04, LD2410 Radar  
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost  
- **Visualization:** Matplotlib / Seaborn  

---

## 📈 Results
- **Random Forest Accuracy:** 94.63%  
- **XGBoost Accuracy:** 96.25%  
- **Ensemble Accuracy:** 96.25%  
- **AUC Score:** ~0.99  

---

## 📸 Outputs

### 🔹 System Setup
![Setup](images/output1.png)

### 🔹 Normal Detection
![Normal](images/output2.png)

### 🔹 Suspicious Detection
![Suspicious](images/output3.png)

### 🔹 Additional Output
![Output](images/output4.png)

---

## 🔬 Key Insights
- Sensor fusion improves detection reliability  
- Ensemble model reduces false positives  
- System performs well in real-time conditions  

---

## 🚀 Applications
- Warehouse inspection  
- Security screening  
- Logistics monitoring  
- Detection of concealed items  

---

## 🔮 Future Scope
- Integration with thermal imaging  
- Deployment on edge devices (Jetson Nano)  
- Advanced models (3D CNN, Transformers)  
- Real-time alert system  

---

## 👩‍💻 Authors
- Prathibha KS  
- Kushi Budale  
- Lavanya Balasubramanyam  
- Rachana Muthukumar  

---

## ⭐ Support
If you found this project useful, consider giving it a ⭐ on GitHub!
