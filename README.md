# UCI Bone Marrow Transplant Study

**Dataset Source**: [UCI Bone Marrow Transplant Children](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children)

---

## 1. Overview

We analyzed the **UCI Bone Marrow** dataset to predict two key outcomes:

- **Survival Status** (categorical)  
- **Survival Time** (continuous)

Our main objective was to determine which variables best predict these outcomes and to compare different supervised learning models in terms of their predictive performance.

---

## 2. Methodology

### 2.1 Exploratory Analysis
- Inspected missing data using visualizations (`vis_miss`, `gg_miss_var`).
- Examined correlations (`corrplot`) and outliers using the IQR rule.
- Explored distributions via histograms, density plots, and scatterplot matrices.

### 2.2 Modeling Approaches

1. **Survival Status (Classification)**
   - Logistic Regression
   - Random Forest

2. **Survival Time (Regression)**
   - Linear Regression
   - Lasso (L1 Regularization)
   - Random Forest

### 2.3 Variable Selection
We applied three main strategies to identify important features:
1. **Stepwise Selection** (using AIC-based forward/backward selection)  
2. **Lasso Regularization** (to shrink less important coefficients to zero)  
3. **Random Forest Feature Importance** (ranking variables by mean decrease in node purity)

---

## 3. Key Results

### 3.1 Survival Status

- **Most Important Predictors** (overlap of stepwise, Lasso, and Random Forest):
  1. *Relapse*
  2. *extcGvHD*
  3. *Survival Time*
  4. *Txpostrelapse*

- **Model Comparison**  
  - **Logistic Regression**: ~94.44% accuracy  
  - **Random Forest**: ~94.44% accuracy (before and after tuning)

- **Best Model**  
  - **Logistic Regression** remained the best choice in our comparison, even after Random Forest tuning, due to consistent predictive performance and model interpretability.

### 3.2 Survival Time

- **Features Identified by Each Method**:

  1. **Stepwise Selection**  
     *Stemcellsource, RecipientABO, Disease, Txpostrelapse, extcGvHD, Recipientage, Rbodymass, survival_status, DosageGroup*  

  2. **Lasso**  
     *Donorage, CD34kgx10d6, CD3dCD34, CD3dkgx10d8, Rbodymass, ANCrecovery, PLTrecovery, time_to_aGvHD_III_IV, survival_status*  

  3. **Random Forest**  
     *survival_status, extcGvHD, CD3dCD34, PLTrecovery, CD3dkgx10d8, Donorage, CD34kgx10d6, CMVstatus, Rbodymass, HLAgrI*

- **Model Comparison**  
  - **Stepwise Linear Model**  
    - R-squared: 0.654  
    - RMSE: 494.12  
    - AIC: 2814.30  

  - **Lasso Model**  
    - R-squared: 0.612  
    - RMSE: 523.05  
    - AIC: 2817.01  

  - **Random Forest Model**  
    - R-squared: 0.656  
    - RMSE: 492.40  
    - AIC: 2815.03  

- **Best Model**  
  - **Random Forest** outperformed other models with the highest R-squared and lowest RMSE. It indicates that Random Forest is the most robust regressor for predicting survival time.

---

## 4. Interpretation and Conclusions

1. **Survival Status** depends primarily on:
   - *Relapse*, *extcGvHD*, *Survival Time*, and *Txpostrelapse*
   - *CD34+ dosage* did not appear as a crucial determinant for survival status in the final models.
   - **Logistic Regression** proved the most reliable for classification.

2. **Survival Time** is strongly influenced by:
   - *Survival Status*, *extcGvHD*, *CD3dCD34*, *PLTrecovery*, *CD3dkgx10d8*, *Donorage*, *CD34kgx10d6*, *CMVstatus*, *Rbodymass*, and *HLAgrI*
   - **CD34+ dosage** surfaced as a significant predictor of survival time but does not alone guarantee survival.

3. **Interaction Between Outcomes**  
   - *Survival Status* and *Survival Time* are interdependent (each appeared in the other’s final model).
   - Only *extcGvHD* was shared as a top predictor across both final models.

4. **Potential Insight**  
   - While higher **CD34+ dosage** may prolong survival time, it does not unequivocally ensure survival status.

---

## 5. Final Takeaways

- For **categorical** survival status predictions, **Logistic Regression** is recommended.
- For **continuous** survival time predictions, **Random Forest** is most effective.
- Future research can explore additional clinical covariates and alternative machine learning algorithms (e.g., gradient boosting, neural networks) for further improvements.
- The hypothesis that higher CD34+ cell dosage extends survival time is partly supported by the results, though not conclusively linked to improved survival status.
