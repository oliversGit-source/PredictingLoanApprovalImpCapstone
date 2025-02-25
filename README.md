# Predicting Loan Approval - Binary Classification of Financial Risk Dataset
# Imperial Business School - Professional Certificate in Data Analytics Capstone Project - Oliver Butterworth-Bakhshi
This repository contains an analysis and predictive modelling project evaluating ML classification algorithms on a Financial Risk dataset. The objective of the study is to improve the binary classification model of loan approval in order to reduce the costs of customers defaulting on their loans to lenders.
## Project Structure
Jupyter Notebook: FinancialRiskCapstone3.ipynb contains all the code, analysis, and results. Jupyter notebook link: 

Data: Includes preprocessed and cleaned data with detailed steps for imputation, outlier handling, and feature transformations. The raw data can be found in the data folder.

README: Provides an overview of the project, methods, and key findings.
## Summary of Findings
* **Objective:** Use ML models to predict loan classification from a Financial Risk dataset. 
* **Use case:** ML model can reduce the number of false positives (based on specificity), saving the bank millions of pounds in customer mortgage defaults that would otherwise be avoided. 
* **Data Summary:**
  (Before processing): 20000 rows, 36 columns. (After cleaning and pre-processing): 5452 rows, 42 columns.
  * Target variable: y, Loan Approval, 1 for approved, 0 for rejected.
  * Key preprocessing steps: Removal of outliers, Drop irrelevant columns, Train-test split, Transform and scale categorical columns using Transformer, get.dummies() and StandardScaler.
* **Modelling and results:**
  * Models used: kNN, SVC, Decision Tree, Bagging, Pasting, Random Forest, XGBoost, lightgbm.
  * Evaluation metrics: Precision, Accuracy, Recall, Specificity, with a focus on precision and specificity to maximise business benefit financially.
  * Findings: Best performing model overall = SVC. All models (apart from kNN) had over 0.96 in all metrics scored, very good fitting of data to model.
  * Key Insights: EDA highlighted that higher educated applicants were more likely to be successful. Feature importance and SHAP analysis highlighted key predictors of loan approval as RiskScore, Annual Income and Total DTI.
* **Recommendations and further study:**
  * Recommendations: 
    * Use best predictive SVC model to implement classification of loan applications. 
  * Further study: 
    * Use local SHAP analysis of best performing model to further validate conclusions.
    * Either improve the performance of feature-compatible models, or find a way to show feature analysis for the best performing SVC model. 
    * Estimate business impact for a single bank: request and use a company dataset in a similar manner to this study.
