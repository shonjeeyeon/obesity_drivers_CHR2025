from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/analytic_data2025_v3.csv")
OUTPUT_DIR = Path("outputs")
TARGET = "Adult Obesity raw value"
MAX_MISSING_RATE = 0.25


def load_chr_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.iloc[1:].copy()
    df.columns = df.columns.str.strip()
    return df


def keep_county_rows(df: pd.DataFrame) -> pd.DataFrame:
    county_col = "County Clustered (Yes=1/No=0)"
    if county_col not in df.columns:
        raise KeyError(f"Missing expected county indicator column: {county_col}")
    return df[df[county_col].astype(str).eq("1")].copy()


def select_raw_value_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    id_cols = [c for c in ["State Abbreviation", "Name", "5-digit FIPS Code"] if c in df.columns]
    raw_cols = [c for c in df.columns if c.endswith("raw value")]
    if TARGET not in raw_cols:
        raise KeyError(f"Target column not found: {TARGET}")
    return df[id_cols + raw_cols].copy(), id_cols, raw_cols


def coerce_raw_values_to_numeric(df: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in raw_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_modeling_data(df: pd.DataFrame, raw_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df[df[TARGET].notna()].copy()
    missing_rate = df[raw_cols].isna().mean()
    feature_cols = [c for c in raw_cols if c != TARGET and missing_rate[c] <= MAX_MISSING_RATE]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y, feature_cols


def save_clean_data(original_df: pd.DataFrame, id_cols: list[str], X: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    clean_df = pd.concat(
        [original_df[id_cols].reset_index(drop=True), y.reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )
    clean_df.to_csv(output_dir / "chr2025_obesity_model_data.csv", index=False)
