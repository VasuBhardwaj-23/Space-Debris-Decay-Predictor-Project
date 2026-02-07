# 🛰️ Space Debris Decay Predictor

A machine learning project that predicts the **orbital decay time of space debris** and classifies the associated **re-entry risk** using an interactive Streamlit application.

This project focuses on **real-world relevance, interpretability, and clean ML design**, rather than just chasing accuracy numbers.

---

## 🌍 Problem Overview

Thousands of non-functional objects (space debris) orbit Earth.  
Over time, atmospheric drag causes these objects to lose altitude and eventually re-enter the atmosphere.

Understanding **when** this will happen is critical for:
- Space traffic management  
- Collision avoidance  
- Risk assessment  

This project aims to make that prediction **simple, explainable, and usable**.

---

## 🚀 What This Project Does

- Predicts **orbital decay time (in days)** using machine learning  
- Converts predictions into **risk levels** (High / Medium / Low)  
- Provides a **clean, professional Streamlit dashboard**  
- Separates **ML logic** from **business/risk logic**

---

## 🧠 Approach & Methodology

### 📊 Dataset
- Physics-inspired synthetic dataset  
- ~2200 samples  
- Realistic orbital parameter ranges  

**Key features:**
- Orbital altitude  
- Mass  
- Cross-sectional area  
- Drag coefficient  
- Mean motion  
- Solar activity  

**Target variable:**  
- `decay_time_days`

---

### 🧹 Data Preprocessing
- No missing values  
- Categorical features encoded  
- Train–test split (70/30)  
- Scaling applied where required  

---

### 🤖 Models Used

| Model | Purpose |
|-----|--------|
| Linear Regression | Baseline benchmark |
| Random Forest Regressor | Final deployed model |

The Random Forest model significantly outperformed the baseline by capturing **non-linear orbital decay patterns**.

---

### 🔍 Model Interpretability

Feature importance analysis revealed that:
- **Mass**
- **Cross-sectional area**
- **Orbital altitude**

are the most influential factors affecting decay time — aligning well with real orbital physics.

---

## ⚠️ Risk Classification Logic

The ML model predicts decay time as a **continuous value**.  
Risk categories are applied **at the application level** for clarity and flexibility.

| Predicted Decay Time | Risk Level |
|---------------------|----------|
| ≤ 180 days | 🔴 High Risk |
| 181–730 days | 🟠 Medium Risk |
| > 730 days | 🟢 Low Risk |

This design keeps the model clean while allowing domain-informed decision rules.

---

## 🖥️ Streamlit Web Application

The deployed app includes:
- Sidebar-based input controls  
- Real-time decay predictions  
- Clear visual risk indicators  
- Professional dark-themed UI  

---

## 📂 Project Structure

