# Random Forest Classification

Predict whether a user will purchase a product based on age and estimated salary using a Random Forest ensemble classifier.

## Overview

This project trains a **Random Forest Classifier** on the Social Network Ads dataset to predict purchase behavior. The model builds multiple decision trees and aggregates their votes for robust classification. Both Python and R implementations are provided, along with decision-boundary visualizations.

## Dataset

**Social_Network_Ads.csv** — 400 records with the following features:

| Column           | Description                        |
|------------------|------------------------------------|
| User ID          | Unique identifier (not used)       |
| Gender           | Male / Female (not used)           |
| Age              | User age                           |
| EstimatedSalary  | Annual estimated salary             |
| Purchased        | Target — 0 (No) or 1 (Yes)         |

## Methodology

1. Load dataset and select features (Age, EstimatedSalary)
2. Split into 75% training / 25% test
3. Apply standard scaling
4. Train a Random Forest with 10 trees (entropy criterion)
5. Evaluate with confusion matrix and classification report
6. Visualize decision boundaries for training and test sets

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Primary implementation |
| 📊 R | Alternative implementation |
| 🔬 scikit-learn | Random Forest, preprocessing, metrics |
| 📈 matplotlib | Decision boundary visualization |
| 🐼 pandas / NumPy | Data handling |
| 🌲 randomForest (R) | R-based ensemble model |

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

```bash
python random_forest_classification.py
```

Or in R:

```r
source("random_forest_classification.R")
```

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which has been archived from CRAN. Use `install.packages("ElemStatLearn", repos = "https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/")` or an alternative dataset package.
- Visualizations use a fine mesh grid (`step=0.01`) which can be slow on large feature ranges.

## License

See [LICENSE](LICENSE) for details.
