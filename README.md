#  Credit Card Fraud Detection System

##  Overview

Fraudulent credit card transactions pose a significant threat to financial institutions and customers worldwide. This project presents a **robust machine learning solution** designed to detect fraudulent transactions with high accuracy and reliability.

The system leverages historical transaction data and applies advanced classification algorithms to distinguish between **legitimate** and **fraudulent** activities in real-time.

---

##  Problem Statement

Develop a machine learning model capable of identifying fraudulent credit card transactions using transaction-level data.

The model should:

* Accurately classify transactions as **Fraudulent (1)** or **Legitimate (0)**
* Handle highly **imbalanced datasets**
* Minimize false positives and false negatives

---

##  Solution Approach

This project follows a **structured ML pipeline**:

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Deployment (Optional Web App)

---

##  Project Structure

```id="r9n8v1"
credit_card_fraud_detection/
│
├── data/
│   └── transactions.csv
│
├── preprocessing.py      # Data cleaning & preparation
├── eda.py                # Data analysis & visualization
├── train_model.py        # Model training
├── predict.py            # Prediction logic
├── main.py               # Entry point
│
├── model/
│   └── fraud_model.pkl   # Saved trained model
│
└── app.py                # Streamlit web app (optional)
```

---

##  Dataset Description

The dataset contains anonymized credit card transaction data with features such as:

* Transaction amount
* Time of transaction
* PCA-transformed features (V1, V2, ..., V28)
* Target variable:

  * `0` → Legitimate transaction
  * `1` → Fraudulent transaction

---

##  Data Preprocessing

* Handled **class imbalance** using:

  * Undersampling / Oversampling (SMOTE)
* Normalized numerical features
* Removed noise and irrelevant data
* Ensured dataset consistency

---

##  Exploratory Data Analysis (EDA)

* Distribution of fraud vs non-fraud transactions
* Correlation heatmaps
* Transaction amount analysis
* Fraud patterns over time

---

##  Machine Learning Models

The following algorithms were implemented and compared:

###  Logistic Regression

* Baseline model
* Fast and interpretable

###  Decision Tree

* Handles non-linear relationships
* Easy to visualize

###  Random Forest 

* Ensemble learning method
* High accuracy and robustness
* Best performing model in this project

---

##  Model Performance

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

>  Special focus on **Recall**, since detecting fraud is critical.

---

##  How to Run

###  Install Dependencies

```id="k2bz7p"
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

###  Train the Model

```id="0j3g2s"
python main.py
```

###  Run Web Application

```id="y5c7k1"
streamlit run app.py
```

---

## Web Application

A simple and interactive web app built using **Streamlit** allows users to:

* Input transaction details
* Get instant fraud predictions
* Visualize results easily

---

##  Key Features

✔ Handles highly imbalanced datasets
✔ Multiple ML models comparison
✔ Clean and modular code structure
✔ Model persistence using Pickle
✔ Real-time prediction capability
✔ Scalable and production-ready design

---

##  Future Enhancements

* Implement advanced models (XGBoost, Neural Networks)
* Real-time fraud detection API
* Deploy on cloud platforms (AWS / Azure / GCP)
* Integrate with banking systems
* Add anomaly detection techniques

---

##  Use Cases

* Banking and financial institutions
* Payment gateways
* E-commerce platforms
* Fraud monitoring systems

---

## Challenges Addressed

* Extreme class imbalance
* High dimensionality
* Minimizing false negatives (critical in fraud detection)

---

##  Contribution

Contributions are welcome! Feel free to:

* Fork the repository
* Open issues
* Submit pull requests

---


## 🙌 Acknowledgment

This project is inspired by real-world financial fraud detection challenges and aims to demonstrate practical machine learning applications in cybersecurity and fintech.

---

⭐ If you find this project useful, don’t forget to **star the repository!**
# credit-card-fraud-detection
