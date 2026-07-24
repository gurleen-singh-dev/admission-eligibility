import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Admission_Predict.csv")

FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]
TARGET = "Chance of Admit"

TREE_MAX_DEPTH = 4  # simple, fixed depth -- easy to reason about


# Step 1: Load data
def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Could not find dataset at: {DATA_PATH}")
        print("Make sure Admission_Predict.csv is inside the 'data/' folder.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]  # clean up stray spaces
    return df


# Step 2: Train the model
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    model = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=42)
    model.fit(X_train, y_train)

    # accuracy check
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


# Step 3: accept details
def _ask_number(prompt_text, min_val, max_val, is_int=False):
    while True:
        raw = input(prompt_text).strip()
        try:
            val = int(raw) if is_int else float(raw)
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if not (min_val <= val <= max_val):
            print(f"  Please enter a value between {min_val} and {max_val}.")
            continue
        return val


def get_applicant_details():
    print("\nEnter applicant details:")
    gre = _ask_number("  GRE Score (260-340): ", 260, 340, is_int=True)
    toefl = _ask_number("  TOEFL Score (0-120): ", 0, 120, is_int=True)
    rating = _ask_number("  University Rating (1-5): ", 1, 5, is_int=True)
    sop = _ask_number("  SOP strength (1.0-5.0): ", 1.0, 5.0)
    lor = _ask_number("  LOR strength (1.0-5.0): ", 1.0, 5.0)
    cgpa = _ask_number("  CGPA (0.0-10.0): ", 0.0, 10.0)
    research = _ask_number("  Research experience (0 = No, 1 = Yes): ",
                           0,
                           1,
                           is_int=True)

    return pd.DataFrame([[gre, toefl, rating, sop, lor, cgpa, research]],
                        columns=FEATURES)


# Step 4: Predict
def predict_eligibility(model, applicant_df):
    prediction = model.predict(applicant_df)[0]
    probabilities = model.predict_proba(applicant_df)[0]

    eligible_index = list(model.classes_).index(1)
    confidence = probabilities[eligible_index]

    label = "Eligible" if prediction == 1 else "Not Eligible"
    return label, confidence


# Main function
def main():
    print(" \nAdmission Eligibility Predictor (Decision Tree)\n")

    df = load_data()
    model, accuracy = train_model(df)
    print(f"Model trained. Test accuracy: {accuracy:.2%}")

    applicant_df = get_applicant_details()
    label, confidence = predict_eligibility(model, applicant_df)

    print("\n--- Result ---")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.1%} chance of being eligible")
    print("-" * 20)

if __name__ == "__main__":
    main()
