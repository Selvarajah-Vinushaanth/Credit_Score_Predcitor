from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

# Load model, scaler, and label encoders
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# List of all required features (must match training column order)
features = [
    'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
    'Num_of_Loan', 'Type_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Credit_Utilization_Ratio', 'Credit_History_Age',
    'Payment_of_Min_Amount', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance'
]

categorical_cols = list(label_encoders.keys())

# Define valid options for categorical fields
VALID_OPTIONS = {
    'Occupation': ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
                  'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 
                  'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'],
    'Type_of_Loan': ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan', 
                     'Home Equity Loan', 'Payday Loan', 'Student Loan', 
                     'Mortgage Loan', 'Not Specified'],
    'Credit_Mix': ['Good', 'Standard', 'Bad'],
    'Payment_Behaviour': ['High_spent_Small_value_payments', 
                         'Low_spent_Large_value_payments',
                         'Low_spent_Medium_value_payments',
                         'Low_spent_Small_value_payments',
                         'High_spent_Medium_value_payments',
                         'High_spent_Large_value_payments'],
    'Payment_of_Min_Amount': ['No', 'NM', 'Yes']
}

# Optional: default input values
default_inputs = {
    'Age': 30,
    'Occupation': 'Engineer',
    'Annual_Income': 50000,
    'Monthly_Inhand_Salary': 3000,
    'Num_Bank_Accounts': 2,
    'Num_Credit_Card': 1,
    'Interest_Rate': 10.0,
    'Num_of_Loan': 1,
    'Type_of_Loan': 'Personal Loan',
    'Delay_from_due_date': 0,
    'Num_of_Delayed_Payment': 0,
    'Changed_Credit_Limit': 0,
    'Num_Credit_Inquiries': 1,
    'Credit_Mix': 'Good',
    'Outstanding_Debt': 1000,
    'Credit_Utilization_Ratio': 0.3,
    'Credit_History_Age': 24,
    'Payment_of_Min_Amount': 'Yes',
    'Total_EMI_per_month': 300,
    'Amount_invested_monthly': 200,
    'Payment_Behaviour': 'High_spent_Small_value_payments',
    'Monthly_Balance': 1000
}

# Example input for "Bad" credit score
example_bad_input = {
    'Age': 30,
    'Occupation': 'Engineer',
    'Annual_Income': 50000,
    'Monthly_Inhand_Salary': 3000,
    'Num_Bank_Accounts': 2,
    'Num_Credit_Card': 1,
    'Interest_Rate': 10.0,
    'Num_of_Loan': 1,
    'Type_of_Loan': 'Personal Loan',
    'Delay_from_due_date': 0,
    'Num_of_Delayed_Payment': 0,
    'Changed_Credit_Limit': 0,
    'Num_Credit_Inquiries': 1,
    'Credit_Mix': 'Good',
    'Outstanding_Debt': 1000,
    'Credit_Utilization_Ratio': 0.3,
    'Credit_History_Age': 24,
    'Payment_of_Min_Amount': 'Yes',
    'Total_EMI_per_month': 300,
    'Amount_invested_monthly': 200,
    'Payment_Behaviour': 'High_spent_Small_value_payments',
    'Monthly_Balance': 1000
}

@app.route('/')
def home():
    return render_template('index.html', defaults=default_inputs, options=VALID_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = dict(request.form)
        input_mode = form_data.get('input_mode', 'manual')
        input_source = {}

        if input_mode == 'json':
            json_input = form_data.get('json_input', '')
            # Only try to parse if not empty and not just whitespace
            if json_input and json_input.strip():
                json_input_stripped = json_input.strip()
                # If user pasted "example_bad_input = {...}", strip the assignment part
                if json_input_stripped.startswith("example_bad_input"):
                    eq_idx = json_input_stripped.find("=")
                    if eq_idx != -1:
                        json_input_stripped = json_input_stripped[eq_idx+1:].strip()
                # If user pasted a Python dict, convert single quotes to double quotes for a best-effort parse
                if json_input_stripped.startswith("{'") or json_input_stripped.startswith("{\"") or json_input_stripped.startswith("{"):
                    # Try to replace single quotes with double quotes if needed
                    if "'" in json_input_stripped and '"' not in json_input_stripped:
                        json_input_stripped = json_input_stripped.replace("'", '"')
                try:
                    json_data = json.loads(json_input_stripped)
                    for k in features:
                        if k in json_data:
                            input_source[k] = json_data[k]
                except Exception as je:
                    tip = ""
                    if json_input.strip().startswith("{'") or json_input.strip().startswith("example_bad_input") or "'" in json_input:
                        tip = "<br>Tip: Paste valid JSON, not Python dict. Use double quotes for keys and values.<br>Example:<br><pre>{<br>  \"Age\": 22,<br>  ...<br>}</pre>"
                    merged_defaults = {**default_inputs, **form_data}
                    return render_template(
                        'index.html',
                        prediction_text=f"<span style='color:red;'>Invalid JSON input: {je}{tip}</span>",
                        defaults=merged_defaults
                    )
            merged_defaults = {**default_inputs, **form_data, **input_source}
        else:
            merged_defaults = {**default_inputs, **form_data}

        input_data = {}
        for feature in features:
            if input_mode == 'json':
                val = input_source.get(feature, default_inputs[feature])
            else:
                val = form_data.get(feature, default_inputs[feature])
            
            if feature in categorical_cols:
                val = str(val).strip()
                le = label_encoders[feature]
                # Check if value is in valid options for the feature
                if feature in VALID_OPTIONS:
                    if val not in VALID_OPTIONS[feature]:
                        val = VALID_OPTIONS[feature][0]  # Use first valid option as fallback
                if val in le.classes_:
                    input_data[feature] = le.transform([val])[0]
                else:
                    # Use first known class as fallback for unknown values
                    input_data[feature] = le.transform([le.classes_[0]])[0]
            else:
                try:
                    fval = float(val)
                except Exception:
                    fval = float(default_inputs[feature])
                if feature != "Changed_Credit_Limit" and fval < 0:
                    fval = 0
                input_data[feature] = fval

        # Print encoded input values for debugging
        print("\n=== Encoded Input Values ===")
        print(input_data)
        print("===========================\n")
        
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        
        print("\n=== Scaled Input Values ===")
        scaled_df = pd.DataFrame(scaled_input, columns=features)
        print(scaled_df.to_string())
        print("========================\n")
        
        prediction = model.predict(scaled_input)
        pred_raw = prediction[0]

        if isinstance(pred_raw, str):
            predicted_label = pred_raw
        else:
            score_map = {0: 'Bad', 1: 'Standard', 2: 'Good'}
            predicted_label = score_map.get(int(pred_raw), str(pred_raw))

        explanations = {
            'Good': "Excellent credit profile. Likely to get best rates.",
            'Standard': "Average credit profile. May get standard rates.",
            'Bad': "Poor credit profile. May face difficulty getting credit."
        }
        explanation = explanations.get(predicted_label, "")

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_input)
            if hasattr(model, "classes_"):
                if predicted_label in model.classes_:
                    idx = list(model.classes_).index(predicted_label)
                    confidence = float(proba[0][idx]) * 100
                else:
                    confidence = float(np.max(proba)) * 100
            else:
                confidence = float(np.max(proba)) * 100

        details = f"Predicted Credit Score: <b>{predicted_label}</b> (model output: {pred_raw})<br>"
        if confidence is not None:
            details += f"Confidence: <b>{confidence:.1f}%</b><br>"
        details += f"<i>{explanation}</i>"

        return render_template('index.html', prediction_text=details, defaults=merged_defaults)
    except Exception as e:
        form_data = dict(request.form)
        merged_defaults = {**default_inputs, **form_data}
        return render_template('index.html', prediction_text=f"<span style='color:red;'>Error: {str(e)}</span>", defaults=merged_defaults)

if __name__ == '__main__':
    app.run(debug=True)
