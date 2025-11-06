# Bank Reconciliation using Logistic Regression

## 📋 Project Overview

This project implements an **automated bank reconciliation system** using **Machine Learning (Logistic Regression)** to match transactions between API source data and bank settlement records. The model achieves **100% test accuracy** and **1.0 AUC score**, making it production-ready for deployment.

## 🎯 Objective

Automate the tedious manual process of bank reconciliation by building a binary classification model that predicts whether transactions from two different sources (API and Bank) are matching or not.

## 🔍 Problem Statement

Banks and financial institutions need to reconcile transactions daily between:
- **API Source**: Internal transaction records from the payment API
- **Bank Settlement**: Transaction records from the bank

Manual reconciliation is:
- Time-consuming
- Error-prone
- Difficult to scale

**Solution**: Use Machine Learning to automate matching and flag discrepancies.

## 📊 Dataset

### Input Files:
1. **`api_source.csv`** - Transactions from API system
2. **`bank_settlement.csv`** - Transactions from bank records

### Key Fields:
- `utr` - Unique Transaction Reference
- `amount` - Transaction amount
- `status` - Transaction status (success/failed)
- `txn_date` - Transaction date
- `settlement_date` - Bank settlement date

### Dataset Statistics:
- Total Transactions: 82
- Matched Records: 78 (95.12%)
- Unmatched Records: 4

## 🛠️ Technology Stack

- **Python 3.11+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine Learning (Logistic Regression)
- **Matplotlib** - Data visualization
- **Joblib** - Model serialization

## 🧪 Methodology

### 1. Data Loading & Preprocessing
- Load API and Bank CSV files
- Clean column names and data (lowercase, strip whitespace)
- Parse dates properly

### 2. Data Merging
- Outer join on `utr` and `txn_date`
- Identify matched vs unmatched records

### 3. Feature Engineering
Created 4 key features:
- **`amount_diff`** - Absolute difference between API and Bank amounts
- **`status_match`** - Binary indicator if status matches (1=Yes, 0=No)
- **`utr_match`** - Binary indicator if UTR is present
- **`date_diff`** - Days between transaction and settlement date

### 4. Model Training
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 70-30
- **Class Weighting**: Balanced (handles imbalanced data)
- **Max Iterations**: 1000

### 5. Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- ROC Curve & AUC Score
- 5-Fold Cross-Validation
- Threshold Analysis

## 📈 Results

### Model Performance:

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 100.0% |
| **AUC Score** | 1.000 |
| **Cross-Validation F1** | 97.4% (±1.3%) |
| **Precision** | 100% |
| **Recall** | 100% |

### Feature Importance:

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| **status_match** | +3.3711 | ⬆️ Most Important |
| **amount_diff** | +0.2866 | ⬆️ Moderate |
| **date_diff** | -0.2684 | ⬇️ Moderate |
| **utr_match** | +0.0014 | ⬆️ Minimal |

**Key Insight**: Status matching between API and Bank is the strongest predictor of a true match.

## 💼 Business Insights

### Reconciliation Categories:
- **High Confidence Matches (>90% probability)**: 6 records
  - ✅ Can be auto-approved
  
- **Manual Review Required (40-60% probability)**: 0 records
  - ⚠️ Requires human verification
  
- **Likely Mismatches (<30% probability)**: 8 records
  - ❌ Flag for investigation

### Recommendations:
1. **Auto-approve** transactions with match probability > 90%
2. **Manual review** for probabilities between 40-60%
3. **Flag/investigate** transactions with probability < 30%
4. Monitor high amount discrepancies (>100)

## 📁 Project Structure

```
Bancapp_Task/
│
├── main.ipynb                          # Main Jupyter notebook with full analysis
├── README.md                           # Project documentation (this file)
│
├── api_source.csv                      # Input: API transaction data
├── bank_settlement.csv                 # Input: Bank settlement data
│
├── reconciliation_predictions.csv      # Output: All predictions with probabilities
├── logreg_recon.pkl                    # Output: Trained model (serialized)
└── feature_names.csv                   # Output: Feature configuration
```

## 🚀 How to Run

### Prerequisites:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### Steps:
1. **Clone/Download** this repository
2. **Open** `main.ipynb` in Jupyter Notebook or VS Code
3. **Run all cells** sequentially
4. **Review** outputs and predictions in `reconciliation_predictions.csv`

### Using the Trained Model:
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logreg_recon.pkl')

# Prepare your data with the same 4 features
new_data = pd.DataFrame({
    'amount_diff': [0, 150, 5],
    'status_match': [1, 0, 1],
    'utr_match': [1, 1, 1],
    'date_diff': [1, 3, 0]
})

# Get predictions
probabilities = model.predict_proba(new_data)[:, 1]
predictions = model.predict(new_data)

print("Match Probabilities:", probabilities)
print("Predictions:", predictions)
```

## 📊 Outputs

### 1. `reconciliation_predictions.csv`
Contains all transactions with:
- Original API and Bank fields
- Engineered features
- Match probability (`pred_prob_match`)
- Binary prediction (`pred_match`)

### 2. `logreg_recon.pkl`
Serialized Logistic Regression model ready for deployment

### 3. `feature_names.csv`
List of features used by the model (for consistency)

## 🔮 Future Improvements

1. **Add More Features**:
   - Merchant name similarity
   - Account number matching
   - Transaction type alignment

2. **Try Other Algorithms**:
   - Random Forest
   - XGBoost
   - Neural Networks

3. **Real-time Deployment**:
   - Create REST API endpoint
   - Real-time reconciliation service
   - Dashboard for monitoring

4. **Alerting System**:
   - Email alerts for high-value mismatches
   - Daily reconciliation reports
   - Anomaly detection

5. **Explainability**:
   - SHAP values for individual predictions
   - Better feature importance visualization

## 📝 Key Learnings

1. **Feature Engineering is Critical**: The quality of features directly impacts model performance
2. **Class Imbalance**: Using `class_weight='balanced'` helped handle imbalanced data
3. **Cross-Validation**: Essential for ensuring model reliability (97.4% CV F1 score)
4. **Threshold Tuning**: Different thresholds serve different business needs
5. **Business Context**: Understanding reconciliation requirements shaped the solution

## 👤 Author

**Internship Task Submission**  
Bank Reconciliation ML Project

## 📄 License

This project is created for educational/interview purposes.

## 🙏 Acknowledgments

- Scikit-learn documentation
- Pandas community
- Bank reconciliation domain knowledge

---

**Note**: This project demonstrates end-to-end ML workflow including data preprocessing, feature engineering, model training, evaluation, and deployment readiness.

### ⭐ Model Status: **Production Ready**
