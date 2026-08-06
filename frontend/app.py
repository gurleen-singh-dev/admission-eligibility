import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.admission_predictor import (
    FEATURES,
    load_data,
    train_model,
    predict_eligibility,
)

FIELD_CONFIG = {
    "GRE Score": ("GRE Score (260-340)", 260, 340, True),
    "TOEFL Score": ("TOEFL Score (0-120)", 0, 120, True),
    "University Rating": ("University Rating (1-5)", 1, 5, True),
    "SOP": ("SOP strength (1.0-5.0)", 1.0, 5.0, False),
    "LOR": ("LOR strength (1.0-5.0)", 1.0, 5.0, False),
    "CGPA": ("CGPA (0.0-10.0)", 0.0, 10.0, False),
    "Research": ("Research (0 = No, 1 = Yes)", 0, 1, True),
}


class AdmissionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Admission Eligibility Predictor")
        self.resizable(False, False)

        self.df = load_data()
        self.model, self.accuracy = train_model(self.df)

        self.entries = {}
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        header = tk.Label(self,
                          text="Admission Eligibility Predictor",
                          font=("Segoe UI", 14, "bold"))
        header.grid(row=0, column=0, columnspan=2, pady=(15, 0))

        subheader = tk.Label(
            self,
            text=f"Decision Tree | Test accuracy: {self.accuracy:.2%}",
            font=("Segoe UI", 9),
            fg="gray")
        subheader.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        row = 2
        for feature in FEATURES:
            label_text, min_v, max_v, _ = FIELD_CONFIG[feature]
            tk.Label(self, text=label_text).grid(row=row,
                                                 column=0,
                                                 sticky="w",
                                                 **pad)
            entry = tk.Entry(self, width=15)
            entry.grid(row=row, column=1, **pad)
            self.entries[feature] = entry
            row += 1

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="Load Sample",
                  command=self.load_sample).pack(side="left", padx=5)
        tk.Button(btn_frame,
                  text="Predict",
                  command=self.predict,
                  bg="#2563eb",
                  fg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Clear",
                  command=self.clear_fields).pack(side="left", padx=5)
        row += 1

        self.result_label = tk.Label(self,
                                     text="",
                                     font=("Segoe UI", 12, "bold"))
        self.result_label.grid(row=row, column=0, columnspan=2, pady=(5, 5))
        row += 1

        self.confidence_label = tk.Label(self, text="", font=("Segoe UI", 10))
        self.confidence_label.grid(row=row,
                                   column=0,
                                   columnspan=2,
                                   pady=(0, 15))

    def load_sample(self):
        sample = self.df.sample(1).iloc[0]
        for feature in FEATURES:
            self.entries[feature].delete(0, tk.END)
            value = sample[feature]
            if FIELD_CONFIG[feature][3]:
                value = int(value)
            self.entries[feature].insert(0, str(value))
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    def _validate_inputs(self):
        values = {}
        for feature in FEATURES:
            label_text, min_v, max_v, is_int = FIELD_CONFIG[feature]
            raw = self.entries[feature].get().strip()
            if raw == "":
                messagebox.showerror("Missing value",
                                     f"Please enter a value for {feature}.")
                return None
            try:
                val = int(raw) if is_int else float(raw)
            except ValueError:
                messagebox.showerror("Invalid value",
                                     f"{feature} must be a number.")
                return None
            if not (min_v <= val <= max_v):
                messagebox.showerror(
                    "Out of range",
                    f"{feature} must be between {min_v} and {max_v}.")
                return None
            values[feature] = val
        return values

    def predict(self):
        values = self._validate_inputs()
        if values is None:
            return

        import pandas as pd
        applicant_df = pd.DataFrame([[values[f] for f in FEATURES]],
                                    columns=FEATURES)
        label, confidence = predict_eligibility(self.model, applicant_df)

        color = "#16a34a" if label == "Eligible" else "#dc2626"
        self.result_label.config(text=label, fg=color)
        self.confidence_label.config(
            text=f"{confidence:.1%} chance of being eligible")


if __name__ == "__main__":
    app = AdmissionApp()
    app.mainloop()
