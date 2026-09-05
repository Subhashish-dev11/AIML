# CreditWise Frontend

A responsive HTML/CSS/JavaScript frontend for the CreditWise Loan Approval Prediction System.

## Folder Structure

```text
CreditWiseFrontend/
├── index.html
├── css/
│   └── style.css
├── js/
│   └── app.js
└── README.md
```

## How the frontend works

1. The user fills in the loan application.
2. JavaScript validates the input.
3. JavaScript creates a JSON object using the same feature names as the ML dataset.
4. The frontend sends a POST request to:

```text
http://127.0.0.1:5000/predict
```

5. Flask receives the data.
6. Flask creates the engineered features required by the trained pipeline.
7. Flask loads `creditwise_loan_approval_pipeline.pkl`.
8. Flask returns the prediction and approval probability.
9. JavaScript displays the result.

## Expected Backend Response

The frontend can work with a response such as:

```json
{
    "approved": true,
    "probability": 0.93,
    "model": "Gradient Boosting"
}
```

or:

```json
{
    "prediction": 1,
    "approval_probability": 93.0,
    "model": "Gradient Boosting"
}
```

## Run

You can open `index.html` while the Flask API is running.

For local development, the backend should allow CORS requests from the frontend.

## Important

The frontend sends the original applicant features. The backend should calculate these engineered features before calling the model:

- Total_Income
- Loan_to_Income
- Collateral_to_Loan
- DTI_Squared
- Credit_Score_Squared

Do not calculate them differently in the frontend and backend, because the prediction pipeline expects the same feature definitions used during training.
