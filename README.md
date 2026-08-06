# Admission Eligibility Predictor

Predicts whether a student is likely to be **eligible** for admission,
based on GRE, TOEFL, CGPA, and other stats. Built with a Decision Tree
Classifier (scikit-learn), with both a terminal tool and a Tkinter GUI.

## Project Structure

```
admission-eligibility-predictor/
├── data/
│   └── Admission_Predict.csv
├── backend/
│   └── admission_predictor.py
├── frontend/
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Terminal:**
```bash
python backend/admission_predictor.py
```

**GUI:**
```bash
python frontend/app.py
```

Fill in the applicant's stats (or click **Load Sample**) and click
**Predict** to see the result with a confidence score.

## Model

- Decision Tree Classifier, `max_depth=4`
- Test accuracy: ~87.5%