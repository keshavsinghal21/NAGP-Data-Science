# Customer Churn Prediction

## Git repo link
https://github.com/keshavsinghal21/NAGP-Data-Science

## Dataset Used
- data/TelcoCustomerChurn.csv
- data/TelcoCustomerChurn - Data Dictionary.csv

## Project Structure

customer_churn_project/
- data/
  - TelcoCustomerChurn.csv
  - TelcoCustomerChurn - Data Dictionary.csv
- notebook/
  - churn_analysis.ipynb
- model/
  - churn_model.pkl
  - input_schema.json
- app.py
- requirements.txt
- sample_request.json
- README.md

## Steps

### 1. Data Preparation
- Loaded the dataset and checked shape, columns, and data types.
- Checked missing values and duplicates.
- Found blank values in 'TotalCharges' and handled them during preprocessing.
- Used a 70:30 train-test split ('random_state=42') and created a validation split from training data.
- Kept the workflow leakage safe by fitting preprocessing only on training data.

### 2. EDA
created multiple visualizations to understand churn behavior. Main observations:
- Churn class is imbalanced.
- Month-to-month customers churn much more than longer contract customers.
- Lower-tenure customers churn more frequently.
- Higher monthly charges are linked with higher churn risk.

### 3. Feature Engineering
added these features:
- NumServices
- AvgChargesPerMonth
- IsNewCustomer

Features details:

1. NumServices
- How I created it: counted how many services are active ('Yes') across service columns like PhoneService, OnlineSecurity, StreamingTV, etc.
- Why it helps: this gives a single engagement score. Customers using fewer services often have lower switching cost and can be more likely to churn.

2. AvgChargesPerMonth
- How I created it: divided 'TotalCharges' by 'tenure' (with safe handling for zero tenure).
- Why it helps: this normalizes billing by customer age and gives a better view of effective monthly burden.

3. IsNewCustomer
- How I created it: set value to 1 if tenure is 12 months or less, else 0.
- Why it helps: early lifecycle customers usually churn more, so this feature captures that behavior directly.

### 4. Model Building
- Trained Two Decision Tree models(max_depth=4 and max_depth=8) with different parameter settings.
- Compared them using validation metrics.
- ran GridSearch tuning.
- Selected final model based on churn-focused performance (especially recall/F1).

### 5. Final Test Performance
- Accuracy: 0.7539
- Precision: 0.5255
- Recall: 0.7540
- F1 Score: 0.6193

Confusion matrix values:
- TN = 1170
- FP = 382
- FN = 138
- TP = 423

The model catches most churners (good recall), which is useful for retention.

Precision vs Recall decision:
- For this telecom churn case, prioritized 'Recall' over Precision.
- Reason: missing a true churner (false negative) can directly lead to customer loss.
- A few extra false positives are still manageable because retention teams can contact those customers which shouldn't have any issue just few cost.

### 6. Model Interpretation
Top 5 features influencing predictions:
1. Contract_Month-to-month
2. TechSupport_No
3. tenure
4. TotalCharges
5. MonthlyCharges

month-to-month customers without tech support are high-risk.

### 7. Model Saving + API
- Saved model pipeline in Model folder.
- Built Flask POST API with '/predict'.
- API returns both prediction and churn probability.

Note:
- kept feature creation logic inside the API code for this assignment so the flow stays simple and easy to explain.
- The same feature logic is applied at prediction time, so the output remains consistent.

## Steps to Run This

Python version used in this project: 3.13

1. Install dependencies:

pip install -r requirements.txt


2. Start API server:

python app.py

Server runs at: 'http://127.0.0.1:5000'

## API Test

Run below curl command:

curl.exe -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" --data "@sample_request.json"

Sample json response:

{   
  "churn_probability": 0.7471,
  "prediction": "Yes"
}

## Final Suggestion
For retention action, first target:
- Month-to-month customers
- Customers without tech support
- Lower-tenure customers with higher monthly spend

These groups consistently appeared as high-risk in both EDA and model interpretation.
