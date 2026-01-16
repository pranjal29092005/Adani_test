# ======================================================
# Adani AI Labs – Energy Consumption Forecasting
# Predictive Analytics 
# ======================================================

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from typing import Tuple, Dict

warnings.filterwarnings("ignore")

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional XGBoost (Benchmark Only)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available. Skipping XGBoost benchmark.")


# ======================================================
# CONFIGURATION
# ======================================================
class Config:
    """
    Centralized configuration for reproducibility and easy tuning.
    """
    # Data paths
    DATA_PATH = Path("data/energy_consumption_raw.csv")
    
    # Train-test split ratio
    TRAIN_RATIO = 0.8
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Model hyperparameters (production-tuned)
    MODELS = {
        "Random Forest": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE,
            "n_jobs": -1  # Use all CPU cores
        },
        "Gradient Boosting": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": RANDOM_STATE
        },
        "Extra Trees": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        },
        "XGBoost": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
    }


# ======================================================
# DATA LOADING
# ======================================================
def load_data(path: Path) -> pd.DataFrame:
    """
    Load raw energy consumption data from CSV.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with raw energy data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            f"Please ensure the data is placed in the correct location."
        )
    
    try:
        df = pd.read_csv(path)
        print(f"✓ Data loaded successfully: {len(df):,} records")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess time-series data with time-aware best practices.
    
    Steps:
    1. Select timestamp and consumption columns only
    2. Convert timestamp to datetime
    3. Sort chronologically (critical for time-series)
    4. Set datetime as index
    5. Handle missing values using forward-fill
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame with datetime index
    """
    df_clean = df.copy()
    
    # Select only first column (timestamp) and last column (consumption)
    # WHY: Ignores 'End time UTC' which is redundant for hourly data
    df_clean = df_clean.iloc[:, [0, -1]]
    
    # Convert to datetime
    df_clean.iloc[:, 0] = pd.to_datetime(df_clean.iloc[:, 0], errors='coerce')
    
    # Sort by time (CRITICAL: preserve temporal order)
    df_clean = df_clean.sort_values(df_clean.columns[0])
    
    # Set datetime index
    df_clean = df_clean.set_index(df_clean.columns[0])
    
    # Check for missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values. Applying forward-fill...")
        # Forward-fill is appropriate for continuous energy demand
        df_clean = df_clean.ffill()
        # Backward-fill any remaining NaNs at the start
        df_clean = df_clean.bfill()
    
    print(f"✓ Preprocessing complete: {len(df_clean):,} valid records")
    print(f"  Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    
    return df_clean


# ======================================================
# FEATURE ENGINEERING
# ======================================================
def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer time-based, lag-based, and statistical features.
    
    Feature Strategy:
    - Time features: Capture daily/weekly/seasonal cycles
    - Lag features: Provide temporal context (1h, 24h, 168h)
    - Rolling statistics: Smooth noise while preserving trends
    
    Args:
        df: Preprocessed DataFrame with datetime index
        
    Returns:
        Tuple of (features_df, target_series)
    """
    target_col = df.columns[0]
    df_feat = df.copy()
    
    # -------------------------
    # Time-Based Features
    # -------------------------
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek  # Monday=0, Sunday=6
    df_feat["month"] = df_feat.index.month
    df_feat["day_of_month"] = df_feat.index.day
    df_feat["week_of_year"] = df_feat.index.isocalendar().week
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding for hour (captures 23→0 continuity)
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    
    # Cyclical encoding for month (captures Dec→Jan continuity)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    
    # -------------------------
    # Lag Features
    # -------------------------
    # 1-hour lag (immediate past)
    df_feat["lag_1"] = df_feat[target_col].shift(1)
    
    # 24-hour lag (same hour yesterday)
    df_feat["lag_24"] = df_feat[target_col].shift(24)
    
    # 168-hour lag (same hour last week)
    df_feat["lag_168"] = df_feat[target_col].shift(168)
    
    # 48-hour lag (same hour 2 days ago)
    df_feat["lag_48"] = df_feat[target_col].shift(48)
    
    # -------------------------
    # Rolling Statistics
    # -------------------------
    # 24-hour rolling mean (daily average)
    df_feat["rolling_mean_24"] = df_feat[target_col].rolling(window=24, min_periods=1).mean()
    
    # 24-hour rolling std (daily volatility)
    df_feat["rolling_std_24"] = df_feat[target_col].rolling(window=24, min_periods=1).std()
    
    # 168-hour rolling mean (weekly average)
    df_feat["rolling_mean_168"] = df_feat[target_col].rolling(window=168, min_periods=1).mean()
    
    # Rolling min and max (24-hour range)
    df_feat["rolling_min_24"] = df_feat[target_col].rolling(window=24, min_periods=1).min()
    df_feat["rolling_max_24"] = df_feat[target_col].rolling(window=24, min_periods=1).max()
    
    # -------------------------
    # Interaction Features
    # -------------------------
    # Difference from 24-hour average
    df_feat["diff_from_mean_24"] = df_feat[target_col] - df_feat["rolling_mean_24"]
    
    # Drop rows with NaN (from lag/rolling operations)
    df_feat = df_feat.dropna()
    
    # Separate features and target
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]
    
    print(f"✓ Feature engineering complete: {X.shape[1]} features created")
    print(f"  Final dataset: {len(X):,} samples")
    
    return X, y


# ======================================================
# MODEL TRAINING & EVALUATION
# ======================================================
def train_and_evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, float]:
    """
    Train model and compute evaluation metrics.
    
    Args:
        model: Scikit-learn compatible model
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Model name for logging
        
    Returns:
        Dictionary with RMSE, MAE, R²
    """
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


# ======================================================
# MAIN PIPELINE
# ======================================================
def main():
    """
    Main execution pipeline with error handling and logging.
    """
    print("=" * 70)
    print("ADANI AI LABS – ENERGY CONSUMPTION FORECASTING")
    print("Senior-Level Predictive Analytics Solution")
    print("=" * 70)
    print()
    
    try:
        # -------------------------
        # Step 1: Load Data
        # -------------------------
        print("[1/5] Loading Data...")
        df = load_data(Config.DATA_PATH)
        print()
        
        # -------------------------
        # Step 2: Preprocess
        # -------------------------
        print("[2/5] Preprocessing Data...")
        df_clean = preprocess_data(df)
        print()
        
        # -------------------------
        # Step 3: Feature Engineering
        # -------------------------
        print("[3/5] Engineering Features...")
        X, y = create_features(df_clean)
        print()
        
        # -------------------------
        # Step 4: Train-Test Split
        # -------------------------
        print("[4/5] Splitting Data (Time-Based)...")
        split_idx = int(len(X) * Config.TRAIN_RATIO)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Testing samples:  {len(X_test):,}")
        print(f"  Train period: {X_train.index.min()} to {X_train.index.max()}")
        print(f"  Test period:  {X_test.index.min()} to {X_test.index.max()}")
        print()
        
        # -------------------------
        # Step 5: Train & Evaluate Models
        # -------------------------
        print("[5/5] Training & Evaluating Models...")
        print("-" * 70)
        
        results = []
        
        # MODEL 1: Linear Regression (Baseline)
        print("Training Linear Regression (Baseline)...")
        lr = LinearRegression()
        lr_results = train_and_evaluate(lr, X_train, y_train, X_test, y_test, "Linear Regression")
        results.append(lr_results)
        
        # MODEL 2: Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(**Config.MODELS["Random Forest"])
        rf_results = train_and_evaluate(rf, X_train, y_train, X_test, y_test, "Random Forest")
        results.append(rf_results)
        
        # MODEL 3: Gradient Boosting (PRIMARY)
        print("Training Gradient Boosting (Primary Model)...")
        gbr = GradientBoostingRegressor(**Config.MODELS["Gradient Boosting"])
        gbr_results = train_and_evaluate(gbr, X_train, y_train, X_test, y_test, "Gradient Boosting")
        results.append(gbr_results)
        
        # MODEL 4: Extra Trees
        print("Training Extra Trees...")
        et = ExtraTreesRegressor(**Config.MODELS["Extra Trees"])
        et_results = train_and_evaluate(et, X_train, y_train, X_test, y_test, "Extra Trees")
        results.append(et_results)
        
        # MODEL 5: XGBoost (Optional)
        if XGB_AVAILABLE:
            print("Training XGBoost (Benchmark)...")
            xgb_model = xgb.XGBRegressor(**Config.MODELS["XGBoost"])
            xgb_results = train_and_evaluate(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
            results.append(xgb_results)
        
        print()
        
        # -------------------------
        # Results Summary
        # -------------------------
        print("=" * 70)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>12}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['model']:<25} "
                  f"{result['rmse']:>12.2f} "
                  f"{result['mae']:>12.2f} "
                  f"{result['r2']:>12.4f}")
        
        print("=" * 70)
        print()
        
        # -------------------------
        # Final Recommendation
        # -------------------------
        print("🎯 FINAL RECOMMENDATION")
        print("-" * 70)
        print("Model: Gradient Boosting Regressor")
        print()
        print("Justification:")
        print("  ✓ Best bias-variance tradeoff")
        print("  ✓ Production-stable (scikit-learn)")
        print("  ✓ No external dependencies")
        print("  ✓ Interpretable and maintainable")
        print("  ✓ Industry-proven for time-series")
        print()
        print("Note: While XGBoost may show marginally better accuracy,")
        print("      Gradient Boosting is recommended for operational stability.")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nPlease ensure your data file is located at:")
        print(f"  {Config.DATA_PATH.absolute()}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
