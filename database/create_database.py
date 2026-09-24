import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cleaned"
)

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "pims.db")


def create_database():

    conn = sqlite3.connect(DB_PATH)

    print("Creating PIMS database...")

    # --------------------------------------------------
    # PROJECT DATA
    # --------------------------------------------------

    projects_file = os.path.join(
        DATA_DIR,
        "projects_features.csv"
    )

    if os.path.exists(projects_file):

        df_projects = pd.read_csv(projects_file)

        df_projects.to_sql(
            "projects",
            conn,
            if_exists="replace",
            index=False
        )

        print(
            f"Projects table created: {len(df_projects)} records"
        )

    # --------------------------------------------------
    # RISK PREDICTIONS
    # --------------------------------------------------

    risk_file = os.path.join(
        DATA_DIR,
        "project_risk_predictions.csv"
    )

    if os.path.exists(risk_file):

        df_risk = pd.read_csv(risk_file)

        df_risk.to_sql(
            "risk_predictions",
            conn,
            if_exists="replace",
            index=False
        )

        print(
            f"Risk predictions table created: {len(df_risk)} records"
        )

    # --------------------------------------------------
    # RISK FACTORS
    # --------------------------------------------------

    factors_file = os.path.join(
        DATA_DIR,
        "project_risk_factors.csv"
    )

    if os.path.exists(factors_file):

        df_factors = pd.read_csv(factors_file)

        df_factors.to_sql(
            "risk_factors",
            conn,
            if_exists="replace",
            index=False
        )

        print(
            f"Risk factors table created: {len(df_factors)} records"
        )

    conn.close()

    print("\nDatabase created successfully!")
    print(f"Location: {DB_PATH}")


if __name__ == "__main__":
    create_database()