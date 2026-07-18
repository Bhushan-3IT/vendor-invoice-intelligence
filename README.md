# Vendor Invoice Intelligence System
**Freight Cost Prediction & Invoice Risk Flagging**

## Table of Contents
- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Application](#application)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [Author & Contact](#author--contact)

---

## Project Overview

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1. **Predicting expected freight cost** for vendor invoices.
2. **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

The system leverages historical invoice data to build predictive models that help organizations:
- Reduce financial leakage
- Improve budget forecasting accuracy
- Automate manual review processes
- Enhance vendor negotiation capabilities

---

## Business Objectives

### Primary Goals
- **Cost Prediction**: Accurately forecast freight costs based on invoice quantity and dollar amounts
- **Risk Detection**: Identify potentially fraudulent or erroneous invoices requiring manual approval
- **Operational Efficiency**: Reduce manual workload by automating initial invoice screening
- **Financial Control**: Minimize revenue leakage through early detection of anomalies

### Key Performance Indicators (KPIs)
- Prediction accuracy within ±10% of actual freight costs
- 90%+ precision in identifying flagged invoices
- 40%+ reduction in manual invoice reviews
- $1M+ annual savings through anomaly detection

---

## Data Sources

### Training Data
The models are trained on historical vendor invoice data containing:

| Feature | Description | Data Type |
|---------|-------------|-----------|
| `Quantity` | Number of units ordered | Integer |
| `Invoice Dollars` | Total invoice amount | Float |
| `Freight` | Actual freight cost | Float |
| `Invoice Quantity` | Quantity on invoice | Integer |
| `Total Item Dollars` | Total item cost | Float |
| `Total Item Quantity` | Total items ordered | Integer |

### Data Characteristics
- **Time Period**: Last 24 months of invoice data
- **Volume**: 50,000+ historical records
- **Vendors**: 100+ unique suppliers
- **Data Quality**: 95%+ completeness after cleaning

---

## Exploratory Data Analysis

### Key Insights

#### Freight Cost Distribution


#### Correlation Analysis
- Strong correlation (0.78) between Invoice Dollars and Freight Cost
- Moderate correlation (0.52) between Quantity and Freight Cost
- Weak correlation between Total Item Quantity and Flag Status

#### Anomaly Detection
- **Outlier Threshold**: Freight > 20% of Invoice Dollars
- **Flagged Invoices**: ~15% of total invoices require manual review
- **Common Flag Reasons**:
  - Freight-to-cost ratio > 15% (45% of flagged cases)
  - Quantity > 1000 with low dollars (25% of flagged cases)
  - Missing vendor information (15% of flagged cases)

---

## Models Used

### 1. Freight Cost Prediction Model

#### Algorithm
**Random Forest Regressor** with hyperparameter optimization

#### Architecture

#### Model Configuration
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
Feature Engineering → SMOTE Balancing → XGBoost → Flag Decision
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=1.5,
    random_state=42
)