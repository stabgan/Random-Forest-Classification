# 🌲 Random Forest Classification

A machine learning project implementing **Random Forest Classification** in both Python and R to predict whether a user will purchase a product based on age and estimated salary, using the Social Network Ads dataset.

## 📖 Description

This project demonstrates a supervised classification approach using the Random Forest ensemble method. The model builds multiple decision trees during training and outputs the class that is the mode of the individual trees' predictions — reducing overfitting and improving generalization compared to a single decision tree.

## 🔬 Methodology

1. **Data Loading** — Import the Social Network Ads dataset (400 records with Age, Estimated Salary, and Purchase outcome)
2. **Feature Selection** — Extract Age and Estimated Salary as independent variables
3. **Train/Test Split** — 75% training, 25% test (stratified via `random_state=0`)
4. **Feature Scaling** — Standardize features using `StandardScaler`
5. **Model Training** — Fit a Random Forest Classifier with entropy criterion
6. **Evaluation** — Confusion matrix and classification report (precision, recall, F1-score)
7. **Visualization** — Decision boundary plots for both training and test sets

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Primary implementation |
| 📊 R | Alternative implementation |
| 🧠 scikit-learn | ML model, preprocessing, evaluation |
| 🔢 NumPy | Numerical operations |
| 🐼 pandas | Data loading and manipulation |
| 📈 matplotlib | Decision boundary visualization |

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

For the R version:

```r
install.packages(c("caTools", "randomForest"))
```

## 🚀 How to Run

### Python

```bash
python random_forest_classification.py
```

### R

```r
source("random_forest_classification.R")
```

The script will:
- Train the model and print evaluation metrics (confusion matrix + classification report)
- Display decision boundary plots for training and test sets

## ⚠️ Known Issues

- **R `ElemStatLearn` package** — The `ElemStatLearn` library used in the R script has been archived from CRAN. The visualization section in the R version may fail on newer R installations. Consider using `ggplot2` as an alternative.
- **Visualization performance** — The decision boundary mesh grid (`step=0.01`) can be slow on large feature ranges. Increase the step size if plotting takes too long.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
