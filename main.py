from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_cleaning import (
    DATA_PATH,
    OUTPUT_DIR,
    TARGET,
    build_modeling_data,
    coerce_raw_values_to_numeric,
    keep_county_rows,
    load_chr_data,
    save_clean_data,
    select_raw_value_columns,
)
from src.explain import coefficient_table, shap_table_and_plots
from src.model import RANDOM_STATE, TEST_SIZE, regression_metrics, split_data, train_models


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_chr_data(DATA_PATH)
    df = keep_county_rows(df)
    raw_df, id_cols, raw_cols = select_raw_value_columns(df)
    raw_df = coerce_raw_values_to_numeric(raw_df, raw_cols)
    X, y, feature_cols = build_modeling_data(raw_df, raw_cols)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    elasticnet, random_forest = train_models(X_train, y_train)

    en_pred = elasticnet.predict(X_test)
    rf_pred = random_forest.predict(X_test)

    metrics = pd.DataFrame(
        [
            {"model": "ElasticNet", **regression_metrics(y_test, en_pred)},
            {"model": "RandomForest", **regression_metrics(y_test, rf_pred)},
        ]
    )

    en_model = elasticnet.named_steps["model"]
    metrics.loc[metrics["model"] == "ElasticNet", "alpha"] = float(en_model.alpha_)
    metrics.loc[metrics["model"] == "ElasticNet", "l1_ratio"] = float(en_model.l1_ratio_)
    metrics.loc[metrics["model"] == "ElasticNet", "nonzero_coefficients"] = int(np.sum(en_model.coef_ != 0))

    coef_df = coefficient_table(elasticnet, feature_cols)
    shap_df = shap_table_and_plots(random_forest, X_test, feature_cols, OUTPUT_DIR)

    top_coef = coef_df.head(15).sort_values("coefficient")
    plt.figure(figsize=(8, 6))
    plt.barh(top_coef["feature"], top_coef["coefficient"])
    plt.xlabel("Standardized coefficient")
    plt.ylabel("")
    plt.title("Top 15 ElasticNet coefficients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elasticnet_top15_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close()

    save_clean_data(raw_df.loc[X.index], id_cols, X, y, OUTPUT_DIR)

    coef_df.to_csv(OUTPUT_DIR / "elasticnet_coefficients.csv", index=False)
    shap_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    summary = {
        "target": TARGET,
        "n_rows": int(len(X)),
        "n_features": int(len(feature_cols)),
        "top_shap_feature": str(shap_df.iloc[0]["feature"]),
        "top_coefficient_feature": str(coef_df.iloc[0]["feature"]),
    }
    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Pipeline complete.")
    print(metrics.to_string(index=False))
    print("Top SHAP feature:", shap_df.iloc[0]["feature"])


if __name__ == "__main__":
    main()
