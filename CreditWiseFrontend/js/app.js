// CreditWise Frontend
// This file collects form data, validates it and sends the application
// to the Python backend. The backend is responsible for running the ML model.

const API_URL = "http://127.0.0.1:5000/predict";

const form = document.getElementById("loanForm");
const submitBtn = document.getElementById("submitBtn");
const errorBox = document.getElementById("errorBox");
const progressBar = document.getElementById("progressBar");

const placeholder = document.getElementById("placeholder");
const resultContent = document.getElementById("resultContent");
const decisionBadge = document.getElementById("decisionBadge");
const decisionTitle = document.getElementById("decisionTitle");
const decisionMessage = document.getElementById("decisionMessage");
const probabilityText = document.getElementById("probabilityText");
const probabilityBar = document.getElementById("probabilityBar");
const decisionSmall = document.getElementById("decisionSmall");
const resetBtn = document.getElementById("resetBtn");

// Update the small progress indicator as the user fills the form.
function updateProgress() {
    const fields = [...form.querySelectorAll("input, select")];
    const completed = fields.filter(field => field.value.trim() !== "").length;
    const percentage = (completed / fields.length) * 100;
    progressBar.style.width = `${percentage}%`;
}

form.addEventListener("input", updateProgress);
form.addEventListener("change", updateProgress);

// Convert HTML form fields into the exact feature names used by the ML dataset.
function getFormData() {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric strings to JavaScript numbers before sending JSON.
    const numericFields = [
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
    ];

    numericFields.forEach(field => {
        data[field] = Number(data[field]);
    });

    return data;
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
}

function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
}

function showResult(result) {
    // Accept common backend naming styles to make the frontend easy to integrate.
    const approved =
        result.approved ??
        result.loan_approved ??
        result.prediction === 1 ??
        result.prediction === "Yes";

    const probabilityRaw =
        result.probability ??
        result.approval_probability ??
        result.confidence ??
        0;

    // The backend can return either 0.95 or 95.
    const probability = probabilityRaw > 1
        ? probabilityRaw
        : probabilityRaw * 100;

    const safeProbability = Math.max(0, Math.min(100, probability));

    placeholder.hidden = true;
    resultContent.hidden = false;

    decisionBadge.className = `decision-badge ${approved ? "approved" : "rejected"}`;
    decisionBadge.textContent = approved ? "APPROVAL LIKELY" : "REVIEW REQUIRED";

    decisionTitle.textContent = approved ? "Loan Approved" : "Loan Rejected";
    decisionMessage.textContent = approved
        ? "The machine-learning model predicts that this application is likely to be approved."
        : "The machine-learning model predicts that this application is unlikely to be approved.";

    probabilityText.textContent = `${safeProbability.toFixed(1)}%`;
    probabilityBar.style.width = `${safeProbability}%`;
    decisionSmall.textContent = approved ? "Approved" : "Rejected";

    document.getElementById("modelName").textContent =
        result.model || "Gradient Boosting";

    document.getElementById("resultCard").scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideError();

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const data = getFormData();

    // Basic domain validation before making a network request.
    if (data.Credit_Score < 300 || data.Credit_Score > 900) {
        showError("Credit Score must be between 300 and 900.");
        return;
    }

    if (data.DTI_Ratio < 0 || data.DTI_Ratio > 1) {
        showError("DTI Ratio must be between 0 and 1.");
        return;
    }

    submitBtn.disabled = true;
    submitBtn.querySelector("span").textContent = "Assessing Application...";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const result = await response.json();
        showResult(result);

    } catch (error) {
        console.error(error);

        showError(
            "Could not connect to the CreditWise backend. " +
            "Make sure Flask is running at http://127.0.0.1:5000."
        );
    } finally {
        submitBtn.disabled = false;
        submitBtn.querySelector("span").textContent = "Assess Loan Application";
    }
});

resetBtn.addEventListener("click", () => {
    form.reset();
    updateProgress();
    hideError();

    placeholder.hidden = false;
    resultContent.hidden = true;
    probabilityBar.style.width = "0%";
});

updateProgress();
