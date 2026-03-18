from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def coefficient_table(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    coefs = model.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coefficient", ascending=False)


def shap_table_and_plots(model: Pipeline, X_test: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    rf_model = model.named_steps["model"]

    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)
    X_shap = X_test_imp.sample(n=min(250, len(X_test_imp)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    top_shap = shap_df.head(15).sort_values("mean_abs_shap")
    plt.figure(figsize=(8, 6))
    plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    plt.title("Top 15 features by SHAP importance")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_top15_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    return shap_df
