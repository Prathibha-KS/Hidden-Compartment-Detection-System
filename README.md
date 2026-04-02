# 🔍 Intelligent Hidden Compartment Detection System

## Overview
This project presents a real-time embedded system for detecting hidden compartments using **Ultrasonic sensors and mmWave Radar**. 

By combining **sensor fusion** with **machine learning models**, the system accurately classifies objects as **Normal** or **Suspicious**, making it suitable for logistics, warehouse inspection, and security screening.

---

## 🔄 Workflow
1. Sensors collect depth and reflectivity data  
2. Data is preprocessed (scaling & normalization)  
3. Features are passed to ML models  
4. Predictions are generated  
5. Final classification: **Normal / Suspicious**  

---

## Machine Learning Models
- **Random Forest**
  - Handles noisy data effectively  
- **XGBoost**
  - Captures complex feature interactions  
- **Ensemble Model (Weighted)**
  - Combines both models for improved accuracy  

---

## Results
- **Random Forest Accuracy:** 94.63%  
- **XGBoost Accuracy:** 96.25%  
- **Ensemble Accuracy:** 96.25%  
- **AUC Score:** ~0.99  

---

## System Setup
![](images/components)
![](images/component_setup)

---

## 📸 Outputs

### 🔹 Normal Detection
![Normal](images/normal)

### 🔹 Suspicious Detection
![Suspicious](images/suspicious)


