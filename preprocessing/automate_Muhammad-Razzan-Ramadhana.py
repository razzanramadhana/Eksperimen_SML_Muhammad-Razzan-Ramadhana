"""
automate_Muhammad-Razzan-Ramadhana.py

Preprocessing otomatis untuk dataset Loan Prediction.
Struktur ini adalah konversi dari eksperimen di notebook:
- Drop Loan_ID
- Imputasi missing value (categorical: most_frequent, LoanAmount: KNNImputer n_neighbors=5)
- Konversi satuan: ApplicantIncome & CoapplicantIncome dikali 16000,
  LoanAmount dikali 1000*16000
- Transformasi log untuk LoanAmount, ApplicantIncome, CoapplicantIncome
- Encoding manual kategori -> numerik (mapping sesuai eksperimen)
- Buat Loan_Amount_Term_Code (mapping sesuai eksperimen)
- Feature engineering: Total_Income, Income_Per_Person, Loan_Income_Ratio,
  Educated_SelfEmployed, Is_Single

Output: data numerik siap untuk dilatih (X, y) atau untuk inference (X).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


# -----------------------------
# Helper: mapping yang konsisten
# -----------------------------
_TERM_MAPPING = {
    12.0: 1,
    36.0: 2,
    60.0: 3,
    84.0: 4,
    120.0: 5,
    180.0: 6,
    240.0: 7,
    300.0: 8,
    360.0: 9,
    480.0: 10,
}

_GENDER_MAP = {"Male": 1, "Female": 0}
_MARRIED_MAP = {"Yes": 1, "No": 0}
_EDU_MAP = {"Graduate": 1, "Not Graduate": 0}  # sesuai dataset umum & eksperimen
_SELF_EMP_MAP = {"Yes": 1, "No": 0}
# Property_Area di dataset sering: Urban, Semiurban, Rural (kadang "Semi Urban")
_PROPERTY_MAP = {"Urban": 2, "Semiurban": 1, "Semi Urban": 1, "Rural": 0}
_LOAN_STATUS_MAP = {"Y": 1, "N": 0}


@dataclass
class LoanPreprocessor:
    """
    Preprocessor yang bisa di-fit lalu dipakai transform ulang (mis. train/test split).
    """
    categorical_imputer: Optional[SimpleImputer] = None
    loanamount_imputer: Optional[KNNImputer] = None
    fitted_: bool = False

    # Kolom imputasi mengikuti eksperimen
    cat_cols_to_impute = ["Gender", "Married", "Dependents", "Self_Employed",
                          "Credit_History", "Loan_Amount_Term"]
    num_cols_to_impute = ["LoanAmount"]

    def fit(self, df: pd.DataFrame) -> "LoanPreprocessor":
        df_ = df.copy()

        # Drop kolom ID jika ada
        if "Loan_ID" in df_.columns:
            df_ = df_.drop(columns=["Loan_ID"])

        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.loanamount_imputer = KNNImputer(n_neighbors=5)

        # Fit imputers (gunakan kolom yang tersedia saja agar robust)
        cat_cols = [c for c in self.cat_cols_to_impute if c in df_.columns]
        num_cols = [c for c in self.num_cols_to_impute if c in df_.columns]

        if cat_cols:
            self.categorical_imputer.fit(df_[cat_cols])
        if num_cols:
            self.loanamount_imputer.fit(df_[num_cols])

        self.fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "Loan_Status",
        return_y: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        if not self.fitted_:
            raise RuntimeError("Preprocessor belum di-fit. Panggil .fit(df_train) dulu.")

        df_ = df.copy()

        # -----------------------------
        # 1) Drop ID
        # -----------------------------
        if "Loan_ID" in df_.columns:
            df_ = df_.drop(columns=["Loan_ID"])

        # -----------------------------
        # 2) Imputasi missing values (sesuai eksperimen)
        # -----------------------------
        cat_cols = [c for c in self.cat_cols_to_impute if c in df_.columns]
        num_cols = [c for c in self.num_cols_to_impute if c in df_.columns]

        if cat_cols:
            df_[cat_cols] = self.categorical_imputer.transform(df_[cat_cols])
        if num_cols:
            df_[num_cols] = self.loanamount_imputer.transform(df_[num_cols])

        # Pastikan Loan_Amount_Term numerik float (sering terbaca float)
        if "Loan_Amount_Term" in df_.columns:
            df_["Loan_Amount_Term"] = pd.to_numeric(df_["Loan_Amount_Term"], errors="coerce")

        # -----------------------------
        # 3) Konversi satuan (sesuai eksperimen)
        # -----------------------------
        if "ApplicantIncome" in df_.columns:
            df_["ApplicantIncome"] = pd.to_numeric(df_["ApplicantIncome"], errors="coerce") * 16000
        if "CoapplicantIncome" in df_.columns:
            df_["CoapplicantIncome"] = pd.to_numeric(df_["CoapplicantIncome"], errors="coerce") * 16000
        if "LoanAmount" in df_.columns:
            df_["LoanAmount"] = pd.to_numeric(df_["LoanAmount"], errors="coerce") * 1000 * 16000

        # -----------------------------
        # 4) Log transform 3 kolom numerik (sesuai eksperimen)
        # -----------------------------
        for col in ["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]:
            if col not in df_.columns:
                continue
            # handle zeros sesuai eksperimen (log1p kalau ada 0)
            if (df_[col] == 0).any():
                df_[col] = np.log1p(df_[col])
            else:
                df_[col] = np.log(df_[col])

        # -----------------------------
        # 5) Encoding manual (sesuai eksperimen)
        # -----------------------------
        if "Gender" in df_.columns:
            df_["Gender"] = df_["Gender"].map(_GENDER_MAP)
        if "Married" in df_.columns:
            df_["Married"] = df_["Married"].map(_MARRIED_MAP)
        if "Dependents" in df_.columns:
            # "3+" -> 3 lalu float
            df_["Dependents"] = df_["Dependents"].replace("3+", 3)
            df_["Dependents"] = pd.to_numeric(df_["Dependents"], errors="coerce").astype(float)
        if "Education" in df_.columns:
            df_["Education"] = df_["Education"].map(_EDU_MAP)
        if "Self_Employed" in df_.columns:
            df_["Self_Employed"] = df_["Self_Employed"].map(_SELF_EMP_MAP)
        if "Property_Area" in df_.columns:
            df_["Property_Area"] = df_["Property_Area"].map(_PROPERTY_MAP)
        if "Credit_History" in df_.columns:
            df_["Credit_History"] = pd.to_numeric(df_["Credit_History"], errors="coerce").astype("Int64").astype(int)

        # Target (jika ada)
        y = None
        if return_y and target_col in df_.columns:
            df_[target_col] = df_[target_col].map(_LOAN_STATUS_MAP)
            y = df_[target_col].astype(int)

        # -----------------------------
        # 6) Loan_Amount_Term_Code (sesuai eksperimen)
        # -----------------------------
        if "Loan_Amount_Term" in df_.columns:
            df_["Loan_Amount_Term_Code"] = df_["Loan_Amount_Term"].map(_TERM_MAPPING)

        # -----------------------------
        # 7) Feature engineering (sesuai eksperimen)
        # -----------------------------
        # 1. Total Income
        if {"ApplicantIncome", "CoapplicantIncome"}.issubset(df_.columns):
            df_["Total_Income"] = df_["ApplicantIncome"] + df_["CoapplicantIncome"]

        # 2. Income per Person
        if {"Total_Income", "Dependents"}.issubset(df_.columns):
            df_["Income_Per_Person"] = df_["Total_Income"] / (df_["Dependents"] + 1)

        # 3. Loan to Income Ratio
        if {"LoanAmount", "Total_Income"}.issubset(df_.columns):
            df_["Loan_Income_Ratio"] = df_["LoanAmount"] / df_["Total_Income"]

        # 7. Interaction: Education × Self_Employed
        if {"Education", "Self_Employed"}.issubset(df_.columns):
            df_["Educated_SelfEmployed"] = df_["Education"] * df_["Self_Employed"]

        # 8. Is_Single
        if {"Married", "Dependents"}.issubset(df_.columns):
            df_["is_Single"] = ((df_["Married"] == 0) & (df_["Dependents"] == 0)).astype(int)

        # -----------------------------
        # 8) Final: siapkan X (numerik)
        # -----------------------------
        if return_y and target_col in df_.columns:
            X = df_.drop(columns=[target_col])
        else:
            X = df_

        # Pastikan tidak ada object/category tersisa
        non_numeric = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Masih ada kolom non-numerik setelah preprocessing: {non_numeric}. "
                "Pastikan mapping/encoding mencakup kolom tersebut."
            )

        if y is None:
            return X
        return X, y


def preprocess_ready_to_train(
    df: pd.DataFrame,
    *,
    target_col: str = "Loan_Status",
) -> Tuple[pd.DataFrame, pd.Series, LoanPreprocessor]:
    """
    Shortcut untuk sekali jalan:
    - fit preprocessor pada df (umumnya data train)
    - transform -> menghasilkan X, y
    - return juga objek preprocessor (agar bisa dipakai transform data baru / test set)
    """
    prep = LoanPreprocessor().fit(df)
    X, y = prep.transform(df, target_col=target_col, return_y=True)
    return X, y, prep


if __name__ == "__main__":
    import pandas as pd
    import os

    # Path sesuai struktur tugas
    raw_path = "preprocessing/loan-dataset_raw/loan_raw.csv"
    output_dir = "preprocessing/loan-dataset_preprocessing"
    output_path = f"{output_dir}/loan_processed.csv"

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    preprocessor = LoanPreprocessor().fit(df)

    # hasil transform mengembalikan (X, y)
    X, y = preprocessor.transform(df, target_col="Loan_Status", return_y=True)

    # satukan lagi jadi satu dataset siap latih (fitur + target)
    df_processed = X.copy()
    df_processed["Loan_Status"] = y

    df_processed.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai. Output: {output_path}")

