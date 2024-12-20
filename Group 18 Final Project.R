#preliminary anaylsis

#Load the dataset
library(readr)
data <- read_csv("~/Downloads/Statistical Learning /final project/bone-marrow.csv")
View(bone_marrow)

#View the first few rows of the dataset
head(data)

####Converting file
install.packages("RWeka")
library(RWeka)
dataset <- read.arff("~/Downloads/Statistical Learning /final project/bone-marrow.arff")
write.csv(dataset, "bone-marrow.2.csv", row.names = FALSE)
data <- read.csv("bone-marrow.csv")
str(data)
summary(data)

##
install.packages("ggplot2")
library(ggplot2)

install.packages("naniar")
library(naniar)

#Visualize missing data
vis_miss(data)
gg_miss_var(data)

install.packages("corrplot")
library(corrplot)

#Correlation matrix
numeric_data <- data[sapply(data, is.numeric)] # Select numeric columns
cor_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(cor_matrix, method = "color")

#Create a boxplot for all numeric columns
numeric_cols <- data[sapply(data, is.numeric)]  # Select numeric columns

install.packages("reshape2")
library(reshape2)
numeric_melt <- melt(numeric_cols)

#Boxplot for all numeric variables
ggplot(numeric_melt, aes(x = variable, y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 2) +
  labs(title = "Boxplots of Numeric Variables", x = "Variables", y = "Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#Apply IQR rule to detect outliers for all numeric columns
detect_outliers_iqr <- function(df) {
  numeric_cols <- sapply(df, is.numeric)  
  outliers <- list()  
  
  for (col in names(df)[numeric_cols]) {
    Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    
    #Identify outliers
    outlier_values <- df[[col]][df[[col]] < lower_bound | df[[col]] > upper_bound]
    outliers[[col]] <- outlier_values  #Store outliers for this column
  }
  
  return(outliers)
}

#Detect outliers
outliers_iqr <- detect_outliers_iqr(data)
print(outliers_iqr)

#Scatterplot matrix to visualize relationships between numeric columns
install.packages("GGally")
library(GGally)

ggpairs(numeric_cols, 
        lower = list(continuous = "points"),
        diag = list(continuous = "densityDiag"),
        upper = list(continuous = "cor"))

#Loop through each numeric column and create a density plot
numeric_cols <- sapply(data, is.numeric)  # Identify numeric columns

#Loop over numeric columns and plot density for each
for (col in colnames(data)[numeric_cols]) {
  ggplot(data, aes(x = .data[[col]])) +
    geom_density(fill = "blue", alpha = 0.5) +
    labs(title = paste("Density Plot of", col), x = col, y = "Density") +
    theme_minimal()
}

#Histogram for all numeric variables
for (col in colnames(numeric_cols)) {
  ggplot(data, aes(x = .data[[col]])) +
    geom_histogram(binwidth = 10, fill = "blue", color = "black", alpha = 0.7) +
    labs(title = paste("Histogram of", col), x = col, y = "Frequency") +
    theme_minimal()
}

#Scatter plot with regression line
ggplot(data, aes(x = CD34kgx10d6, y = survival_time)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "CD34+ Cell Dosage vs Survival Time",
       x = "CD34+ Dosage (cells/kg)",
       y = "Survival Time (days)") +
  theme_minimal()

#Categorize CD34+ dosage into groups (example: low, medium, high)
data$DosageGroup <- cut(data$CD34kgx10d6, 
                        breaks = c(0, 2, 4, 6, Inf), 
                        labels = c("Low", "Medium", "High", "Very High"))

#Boxplot to show Survival Time across different Dosage groups
ggplot(data, aes(x = DosageGroup, y = survival_time)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Survival Time across Different CD34+ Dosage Levels",
       x = "CD34+ Dosage Group",
       y = "Survival Time (days)") +
  theme_minimal()

##model building for surivial_status
#exclude na-values
data <- na.omit(data)

library(dplyr)

#Convert binary and categorical features into factors
data <- data %>%
  mutate(across(c(Recipientgender, Stemcellsource, Donorage35, IIIV, Gendermatch, DonorABO, 
                  RecipientABO, RecipientRh, ABOmatch, CMVstatus, DonorCMV, RecipientCMV, 
                  Riskgroup, Txpostrelapse, Diseasegroup, HLAmatch, HLAmismatch, Antigen, Alel, 
                  HLAgrI, Recipientage10, Recipientageint, Relapse, aGvHDIIIIV, extcGvHD),
                ~ as.factor(.)))

#Confirm data types
str(data)

#Fit initial full model (example: predicting survival_status)
library(MASS)
full_model <- glm(survival_status ~ ., data = data, family = binomial)

#Perform stepwise selection
step_model <- stepAIC(full_model, direction = "both")
#View selected variables
summary(step_model)

install.packages("glmnet")
library(glmnet)

#Prepare the data for glmnet (convert factors to dummy variables)
x <- model.matrix(survival_status ~ ., data = data)[, -1] # Exclude intercept
y <- data$survival_status

#Fit Lasso model
lasso_model <- cv.glmnet(x, y, family = "binomial", alpha = 1)

#Optimal lambda
best_lambda <- lasso_model$lambda.min

#Coefficients of the selected model
selected_variables <- coef(lasso_model, s = best_lambda)
print(selected_variables)

library(randomForest)

#Fit Random Forest model
rf_model <- randomForest(survival_status ~ ., data = data, importance = TRUE)

#View and plot variable importance
importance <- importance(rf_model)
print(importance)
varImpPlot(rf_model)

#RANDOM FOREST MODEL
# Load necessary libraries
install.packages("caret")
library(randomForest)
library(caret)

#Prepare data (ensure that the response variable is factorized for classification)
data$survival_status <- as.factor(data$survival_status)

#Define the selected variables based on feature importance
selected_vars <- c("Relapse", "extcGvHD", "survival_time", "Txpostrelapse")

#Subset the data with selected variables
rf_data <- data[, c(selected_vars, "survival_status")]

#Split data into training and testing sets
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(rf_data$survival_status, p = 0.7, list = FALSE)
train_data <- rf_data[trainIndex, ]
test_data <- rf_data[-trainIndex, ]

#Train the Random Forest model
rf_model <- randomForest(survival_status ~ ., data = train_data, importance = TRUE)

#model summary
print(rf_model)

#Make predictions
rf_predictions <- predict(rf_model, test_data)

#Evaluate the model performance
confusionMatrix(rf_predictions, test_data$survival_status)

#Plot the feature importance
importance(rf_model)
varImpPlot(rf_model)

##LOGISTIC REGRESSION MODEL
#Prepare data
data$survival_status <- as.factor(data$survival_status)

#Define the selected variables based on feature importance
selected_vars <- c("Relapse", "extcGvHD", "survival_time", "Txpostrelapse")

#Subset the data with selected variables
logistic_data <- data[, c(selected_vars, "survival_status")]

#Split data into training and testing sets
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(logistic_data$survival_status, p = 0.7, list = FALSE)
train_data <- logistic_data[trainIndex, ]
test_data <- logistic_data[-trainIndex, ]

#Train the Logistic Regression model
logit_model <- glm(survival_status ~ ., family = binomial, data = train_data)

summary(logit_model)

#Make predictions
logit_predictions <- predict(logit_model, test_data, type = "response")

#Convert probabilities to binary predictions
logit_predictions_bin <- ifelse(logit_predictions > 0.5, 1, 0)

#Evaluate the model performance
confusionMatrix(as.factor(logit_predictions_bin), test_data$survival_status)

##MODEL COMPARISION
#Compare model performances (accuracy, confusion matrix)
rf_accuracy <- confusionMatrix(rf_predictions, test_data$survival_status)$overall["Accuracy"]
logit_accuracy <- confusionMatrix(as.factor(logit_predictions_bin), test_data$survival_status)$overall["Accuracy"]

print(paste("Random Forest Accuracy: ", rf_accuracy))
print(paste("Logistic Regression Accuracy: ", logit_accuracy))

#Based on accuracy, you can choose the best model
if(rf_accuracy > logit_accuracy) {
  print("Random Forest is the best model.")
} else {
  print("Logistic Regression is the best model.")
}

##Tunning
#Tune the Random Forest model
tune_rf <- train(survival_status ~ ., data = train_data, method = "rf", 
                 trControl = trainControl(method = "cv", number = 5), 
                 tuneGrid = expand.grid(mtry = c(1, 2, 3)))

#Print the tuned model
print(tune_rf)

#Make predictions
tune_rf_predictions <- predict(tune_rf, test_data)

#Evaluate the model performance
confusionMatrix(tune_rf_predictions, test_data$survival_status)

##comparision with tunned model
tune_rf_accuracy <- confusionMatrix(tune_rf_predictions, test_data$survival_status)$overall["Accuracy"]
logit_accuracy <- confusionMatrix(as.factor(logit_predictions_bin), test_data$survival_status)$overall["Accuracy"]

print(paste("Random Forest Accuracy: ", tune_rf_accuracy))
print(paste("Logistic Regression Accuracy: ", logit_accuracy))

#Based on accuracy, you can choose the best model
if(tune_rf_accuracy > logit_accuracy) {
  print("Random Forest is the best model.")
} else {
  print("Logistic Regression is the best model.")
}

##model building for surivial_time##
#Define the selected variables (initial set with all predictors)
library(MASS)
initial_model <- lm(survival_time ~ ., data = data)

#Perform stepwise selection (both directions: forward and backward)
stepwise_model <- stepAIC(initial_model, direction = "both", trace = TRUE)

summary(stepwise_model)

##
#survival_time is excluded from predictors
#predictors <- data[, !names(data) %in% c("survival_time")]
#outcome <- data$survival_time

#Fit the Lasso model
library(glmnet)
#lasso_model <- cv.glmnet(as.matrix(predictors), outcome, alpha = 1)

#Check which variables are selected by Lasso
#selected_vars <- coef(lasso_model, s = "lambda.min")[-1]  # Exclude intercept
#selected_vars <- selected_vars[selected_vars != 0]  # Only non-zero coefficients
#print(selected_vars)

#Define the predictors (excluding the outcome)
X <- data[, setdiff(names(data), "survival_time")]
y <- data$survival_time

#Fit a Lasso regression model
lasso_model <- glmnet(as.matrix(X), y, alpha = 1)

#Cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(as.matrix(X), y, alpha = 1)

#Plot the cross-validation results
plot(cv_lasso)

#coeff the model with the best lambda (best model)
best_lambda <- cv_lasso$lambda.min
best_lambda
lasso_coefs <- coef(cv_lasso, s = "lambda.min")
lasso_coefs_m <- as.matrix(lasso_coefs)

#Print the selected variables (non-zero coefficients)
#selected_vars <- rownames(lasso_coefs)[lasso_coefs != 0]
selected_vars <- rownames(lasso_coefs_m)[lasso_coefs_m != 0 & rownames(lasso_coefs_m) != "(Intercept)"]

print("Selected variables from Lasso regression:")
print(selected_vars)

##

library(randomForest)

#the predictors (excluding the outcome)
X_rf <- data[, setdiff(names(data), "survival_time")]
y_rf <- data$survival_time

#Fit the Random Forest regression model
rf_model <- randomForest(x = X_rf, y = y_rf, importance = TRUE)

#feature importance
importance(rf_model)

#Plot the feature importance
varImpPlot(rf_model)

#Print the most important variables based on mean decrease in accuracy
rf_importance <- importance(rf_model)
important_vars <- rownames(rf_importance)[order(rf_importance[, "IncNodePurity"], decreasing = TRUE)]

print("Most important variables from Random Forest:")
print(important_vars)

#models for survivalTime
##MODEL FOR LINEAR REGIONS USING STEPWISE FEATURES
# Use the features selected by stepwise regression (e.g., from stepwise_model)
final_model_stepwise <- lm(survival_time ~ Stemcellsource + RecipientABO + Disease + Txpostrelapse + extcGvHD + Recipientage + Rbodymass + survival_status + DosageGroup, data = data)

#Print the summary of the final model
summary(final_model_stepwise)

##LINEAR REGRESSION USING LASSO FEATURES
# Extract selected variables from the Lasso model (non-zero coefficients)
selected_vars
selected_lasso_vars <- selected_vars

#Create the formula for the final model
formula_lasso <- as.formula(paste("survival_time ~", paste(selected_lasso_vars, collapse = " + ")))

#Fit the final model using the selected features
final_model_lasso <- lm(formula_lasso, data = data)

#Print the summary of the final model
summary(final_model_lasso)

##
#Extract the top important variables based on the random forest results
selected_rf_vars <- important_vars[1:10]  # Assuming the top 10 features are selected
selected_rf_vars
#Create the formula for the final model
formula_rf <- as.formula(paste("survival_time ~", paste(selected_rf_vars, collapse = " + ")))

#Fit the final model using the selected features
final_model_rf <- lm(formula_rf, data = data)

summary(final_model_rf)

##model compariison
#Evaluate Stepwise Model
pred_stepwise <- predict(final_model_stepwise, data)
rmse_stepwise <- sqrt(mean((pred_stepwise - data$survival_time)^2))
rsq_stepwise <- summary(final_model_stepwise)$r.squared
aic_stepwise <- AIC(final_model_stepwise)

#Evaluate Lasso Model
pred_lasso <- predict(final_model_lasso, data)
rmse_lasso <- sqrt(mean((pred_lasso - data$survival_time)^2))
rsq_lasso <- summary(final_model_lasso)$r.squared
aic_lasso <- AIC(final_model_lasso)

#Evaluate Random Forest Model
pred_rf <- predict(final_model_rf, data)
rmse_rf <- sqrt(mean((pred_rf - data$survival_time)^2))
rsq_rf <- summary(final_model_rf)$r.squared
aic_rf <- AIC(final_model_rf)

#Print the evaluation metrics
cat("Stepwise Model:\n")
cat("R-squared:", rsq_stepwise, "RMSE:", rmse_stepwise, "AIC:", aic_stepwise, "\n")

cat("Lasso Model:\n")
cat("R-squared:", rsq_lasso, "RMSE:", rmse_lasso, "AIC:", aic_lasso, "\n")

cat("Random Forest Model:\n")
cat("R-squared:", rsq_rf, "RMSE:", rmse_rf, "AIC:", aic_rf, "\n")

