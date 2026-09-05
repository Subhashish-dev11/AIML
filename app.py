# ============================================================
# CreditWise - Flask Backend
# Loan Approval Prediction System
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib


# ------------------------------------------------------------
# 1. Create the Flask application
# ------------------------------------------------------------

app = Flask(__name__)

# Allow our HTML/CSS/JavaScript frontend to communicate
# with this Flask backend during local development.
CORS(app)


# ------------------------------------------------------------
# 2. Load the trained machine learning pipeline
# ------------------------------------------------------------

# The .pkl file contains:
# - Missing value handling
# - Categorical encoding
# - Feature scaling
# - Trained Gradient Boosting model
#
# Therefore, we do NOT need to repeat those steps here.

MODEL_PATH = "creditwise_loan_approval_pipeline.pkl"

model = joblib.load(MODEL_PATH)

print("CreditWise ML model loaded successfully!")


# ------------------------------------------------------------
# 3. Home route
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """
    Simple route used to check whether the Flask server
    is running correctly.
    """

    return jsonify({
        "message": "CreditWise Loan Approval API is running!",
        "status": "success"
    })


# ------------------------------------------------------------
# 4. Prediction route
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives applicant information from the frontend,
    creates the engineered features used during training,
    and returns the loan approval prediction.
    """

    try:

        # ----------------------------------------------------
        # Get JSON data sent by JavaScript
        # ----------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No applicant data received."
            }), 400


        # ----------------------------------------------------
        # Convert the received data into a DataFrame
        # ----------------------------------------------------

        # A DataFrame is required because our trained
        # scikit-learn pipeline expects named columns.

        df = pd.DataFrame([data])


        # ----------------------------------------------------
        # Convert numerical columns to numbers
        # ----------------------------------------------------

        numeric_columns = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Age",
            "Dependents",
            "Credit_Score",
            "Existing_Loans",
            "DTI_Ratio",
            "Savings",
            "Collateral_Value",
            "Loan_Amount",
            "Loan_Term"
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        # ----------------------------------------------------
        # Validate important numerical values
        # ----------------------------------------------------

        if df[numeric_columns].isnull().any().any():

            return jsonify({
                "error": "Please provide valid numerical values."
            }), 400


        if not 300 <= df["Credit_Score"].iloc[0] <= 900:

            return jsonify({
                "error": "Credit Score must be between 300 and 900."
            }), 400


        if not 0 <= df["DTI_Ratio"].iloc[0] <= 1:

            return jsonify({
                "error": "DTI Ratio must be between 0 and 1."
            }), 400


        # ----------------------------------------------------
        # 5. Feature Engineering
        # ----------------------------------------------------

        # These features were created during model training.
        # We MUST recreate them in exactly the same way
        # before sending the data to the trained pipeline.

        # Total income of applicant + coapplicant
        df["Total_Income"] = (
            df["Applicant_Income"]
            + df["Coapplicant_Income"]
        )

        # Loan amount compared with total income
        df["Loan_to_Income"] = (
            df["Loan_Amount"]
            / (df["Total_Income"] + 1)
        )

        # Value of collateral compared with loan amount
        df["Collateral_to_Loan"] = (
            df["Collateral_Value"]
            / (df["Loan_Amount"] + 1)
        )

        # Squared DTI ratio captures nonlinear relationships
        df["DTI_Squared"] = df["DTI_Ratio"] ** 2

        # Squared credit score captures nonlinear relationships
        df["Credit_Score_Squared"] = df["Credit_Score"] ** 2


        # ----------------------------------------------------
        # 6. Make the prediction
        # ----------------------------------------------------

        prediction = model.predict(df)[0]


        # ----------------------------------------------------
        # 7. Calculate approval probability
        # ----------------------------------------------------

        # Gradient Boosting supports predict_proba().
        probabilities = model.predict_proba(df)[0]

        # Probability of class 1 (Approved)
        approval_probability = float(probabilities[1])


        # ----------------------------------------------------
        # 8. Convert prediction into readable result
        # ----------------------------------------------------

        approved = bool(prediction == 1)

        decision = "Approved" if approved else "Rejected"


        # ----------------------------------------------------
        # 9. Send result back to JavaScript
        # ----------------------------------------------------

        return jsonify({

            "approved": approved,

            "prediction": int(prediction),

            "decision": decision,

            "probability": approval_probability,

            "approval_probability":
                round(approval_probability * 100, 2),

            "model": "Gradient Boosting"

        })


    # --------------------------------------------------------
    # 10. Handle unexpected errors
    # --------------------------------------------------------

    except Exception as e:

        print("Prediction Error:", str(e))

        return jsonify({
            "error": "An error occurred while processing the application.",
            "details": str(e)
        }), 500


# ------------------------------------------------------------
# 11. Start Flask server
# ------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )