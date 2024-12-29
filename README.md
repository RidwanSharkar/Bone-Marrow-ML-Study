# UC-Irvine Bone Marrow Transplant Study
“This data set describes pediatric patients with several hematolic disease, who were subject to the unmanipulated allogeneic unrelated donor hematopoietic stem cell.” (UC Irvine). 

## 1. Overview
CD34+ cells, also known as hematopoietic stem cells (HSCs), are a type of cell that have the CD34 protein on their surface. CD34 is a transmembrane phosphoglycoprotein protein encoded by the CD34 gene in humans. The function of CD34+ cells are to self-renew and produce mature blood cells, such as erythrocytes, leukocytes, platelets, and lymphocytes. CD34+ cells compose roughly 1-5% of total T cells found in bone marrow as well as near endothelial & vascular tissue, and is generally used in cell-based therapies, oncology research, transplant & regenerative medicine studies. 

**Dataset Source**: [UCI Bone Marrow Transplant Children](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children) (187 observations x 36 features)

<img width="997" alt="Screenshot 2024-12-25 at 1 00 11 AM" src="https://github.com/user-attachments/assets/9a499ecc-f586-4137-b188-0916f022d3e2" />

UCI Bone Marrow dataset was analyzed to predict two key outcomes:

- **Survival Status** (categorical)  
- **Survival Time** (continuous)

Our main objective was to determine which variables best predict these outcomes and to compare different supervised learning models in terms of their predictive performance.

---

## 2. Methods

### 2.1 Exploratory Analysis
- Inspected missing data using visualizations (`vis_miss`, `gg_miss_var`).
- Examined correlations (`corrplot`) and outliers using the IQR rule.
- Explored distributions via histograms, density plots, and scatterplot matrices:

**Correlation Matrix:** <br>
<img width="505" alt="Screenshot 2024-12-25 at 12 53 39 AM" src="https://github.com/user-attachments/assets/db075630-5f9c-42d8-bcfe-b5872b980fc0" />

**Scatterplot Matrix:** <br> 
<img width="991" alt="Screenshot 2024-12-25 at 12 55 17 AM" src="https://github.com/user-attachments/assets/ab731b3c-68ff-4c03-86fa-2bdc3b46df35" />

### 2.2 Modeling Approaches

1. **Survival Status (Classification)**
   - Logistic Regression
   - Random Forest

2. **Survival Time (Regression)**
   - Linear Regression
   - Lasso (L1 Regularization)
   - Random Forest

### 2.3 Variable Selection
Three main strategies were used to identify important features:
1. **Stepwise Selection** (using AIC-based forward/backward selection)  
2. **Lasso Regularization** (to shrink less important coefficients to zero)  
3. **Random Forest Feature Importance** (ranking variables by mean decrease in node purity)


---

## 3. Results

### 3.1 Survival Status

- **Most Important Predictors** (overlap of stepwise, Lasso, and Random Forest):
  1. *Relapse*
  2. *extcGvHD*
  3. *Survival Time*
  4. *Txpostrelapse*

- **Model Comparison**  
  - **Logistic Regression**: ~94.44% accuracy  
  - **Random Forest**: ~94.44% accuracy (rounded before and after tuning)
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
  - **Random Forest** outperformed other models with the highest R-squared and lowest RMSE, indicating that Random Forest is the most robust regressor for predicting survival time.

---

## 4. Analysis 

1. **Survival Status** depends primarily on:
   - *Relapse*, *extcGvHD*, *Survival Time*, and *Txpostrelapse*
   - *CD34+ dosage* did not appear as a crucial determinant for survival status in the final models.
   - **Logistic Regression** proved the most reliable for classification.

2. **Survival Time** is strongly influenced by:
   - *Survival Status*, *extcGvHD*, *CD3dCD34*, *PLTrecovery*, *CD3dkgx10d8*, *Donorage*, *CD34kgx10d6*, *CMVstatus*, *Rbodymass*, and *HLAgrI*
   - **CD34+ dosage** surfaced as a significant predictor of survival time but does not alone guarantee survival.

3. **Interaction Between Outcomes**  
   - *Survival Status* and *Survival Time* are interdependent.
   - Only *extcGvHD* was shared as a top predictor across both final models.


---

## 5. Conclusion
- While higher **CD34+ dosage** may prolong survival time, it does not unequivocally ensure survival status.
- For categorical survival status predictions, Logistic Regression is recommended, while for continuous survival time predictions, Random Forest is most effective.
- The hypothesis that higher CD34+ cell dosage extends survival time is partially supported by the results, though not conclusively linked to improved survival status.
