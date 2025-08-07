# 🔐 Network Threat Detection using Machine Learning (CIC-IDS 2017)

This project focuses on building and evaluating multiple machine learning models for **intrusion detection** using the **CIC-IDS-2017** dataset. The system is capable of classifying network traffic into normal and various types of attacks.

---

## 📊 Dataset: CIC-IDS-2017

The **CICIDS 2017** dataset includes realistic network traffic captured in a controlled environment and labeled with various attack types such as:

- DoS
- DDoS
- PortScan
- BruteForce
- BENIGN (Normal traffic)

📁 Files are available here: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## 🎯 Project Goals

- Load and preprocess raw CSV traffic data
- Handle missing values, infinite values, and label encoding
- Address class imbalance using **RandomUnderSampler**
- Train and compare 4 machine learning models:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Support Vector Machine (SVM)
- Evaluate performance using F1 Score, Accuracy, Precision, and Recall
- Hyperparameter tuning with GridSearchCV
- Identify the best-performing model

---

## 🧠 AI Models Used

| Model                | Description |
|---------------------|-------------|
| **Random Forest**    | Ensemble of decision trees, great for tabular data |
| **XGBoost**          | Gradient boosting technique, efficient and accurate |
| **Logistic Regression** | Simple linear model for classification |
| **SVM**              | Finds optimal hyperplane for classification |

---

## ⚙️ Setup Instructions (Google Colab Compatible)

### 1. Install Required Libraries

```bash
!pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost
```

### 2. Upload and Extract Dataset

- Upload your dataset to Colab or mount Google Drive
- Unzip files to `/content/MachineLearningCSV/MachineLearningCVE/`

### 3. Run the Script

Paste the entire script into a Colab notebook and run sequentially.

---

## 🧹 Data Preprocessing Summary

- Handled missing and infinite values
- Encoded categorical labels using `LabelEncoder`
- Scaled features using `StandardScaler`
- Balanced the dataset using `RandomUnderSampler` (optional: SMOTE can be used)
- Split into train/test sets using stratified sampling

---

## 📈 Model Evaluation Metrics

- Confusion Matrix
- Accuracy, Precision, Recall
- Weighted F1-Score
- Per-Class F1 Scores
- Cross-validation with 5-fold CV
- Feature importance plots for tree-based models

---

## 🏆 Final Results

- ✅ **Best Model:** Random Forest / XGBoost (based on F1 Score)
- 🎯 **F1 Score:** ~0.97 (varies by run)
- 📊 Balanced performance across multiple attack classes
- 🚀 Supports hyperparameter tuning for best performance

---

## 📌 Recommendations for Deployment

1. Use real-time traffic feature extraction tools (e.g., Wireshark + parser)
2. Continuously retrain the model with fresh labeled data
3. Integrate the model with a SIEM or firewall for live threat blocking
4. Use ensemble models for higher robustness
5. Apply logging and monitoring for production use

---

## 📂 Folder Structure

```
project-root/
│
├── /notebooks              # Jupyter/Colab notebooks
├── /data                   # CIC-IDS-2017 dataset CSV files
├── /model_results/         # Saved plots and evaluation results
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies (optional)
```

---

## 🤝 Acknowledgements

- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- Network attack detection community for datasets and techniques

---

## 💡 License

This project is released under the [MIT License](LICENSE).
