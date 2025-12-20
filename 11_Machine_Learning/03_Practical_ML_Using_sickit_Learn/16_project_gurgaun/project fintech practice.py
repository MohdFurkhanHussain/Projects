# ============================
# Fintech Stock Predictor (LSTM)
# ============================
# Requirements:
#   pip install numpy pandas scikit-learn tensorflow matplotlib
#
# Usage:
#   - Put your CSV path in DATA_CSV
#   - Ensure CSV has columns: Date,Open,High,Low,Close,Volume
#   - Run: python fintech_stock_predictor.py
#
# Notes:
#   - Predicts next-day Close (or next-day return if you flip TARGET_MODE)
#   - Includes optional simple technical indicators
#   - Saves model to ./models/stock_lstm.h5
# ============================

!pip install numpy 
!pip install pandas
!pip install scikit-learn 
!pip install tensorflow 
!pip install matplotlib

import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ------------- Config -------------
@dataclass
class Config:
    DATA_CSV: str = "your_stock_data.csv"  # <-- put your file path here
    USE_TECHNICALS: bool = True
    LOOKBACK: int = 60          # days of history per sample
    HORIZON: int = 1            # predict next-day
    TARGET_MODE: str = "close"  # "close" or "return"
    TEST_RATIO: float = 0.15
    VAL_RATIO: float = 0.15     # taken from the TRAIN split
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LSTM_UNITS: int = 96
    LSTM_LAYERS: int = 2
    DROPOUT: float = 0.2
    LR: float = 1e-3
    MODEL_DIR: str = "models"
    MODEL_NAME: str = "stock_lstm.h5"
    RANDOM_SEED: int = 42


cfg = Config()
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
np.random.seed(cfg.RANDOM_SEED)
tf.random.set_seed(cfg.RANDOM_SEED)


# ------------- Utils -------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, use_technicals: bool) -> pd.DataFrame:
    # Basic features
    feats = df.copy()
    # Replace zeros in volume to avoid scaling issues
    if "Volume" in feats.columns:
        feats["Volume"] = feats["Volume"].replace(0, np.nan).fillna(method="ffill").fillna(0)

    if use_technicals:
        # Simple moving averages
        feats["SMA_5"] = feats["Close"].rolling(5).mean()
        feats["SMA_10"] = feats["Close"].rolling(10).mean()
        feats["SMA_20"] = feats["Close"].rolling(20).mean()
        # RSI
        feats["RSI_14"] = rsi(feats["Close"], 14)
        # Daily returns
        feats["RET_1"] = feats["Close"].pct_change()
        # Volatility proxy
        feats["HL_PCT"] = (feats["High"] - feats["Low"]) / feats["Close"].replace(0, np.nan)
        feats["OC_PCT"] = (feats["Close"] - feats["Open"]) / feats["Open"].replace(0, np.nan)

    feats = feats.dropna().reset_index(drop=True)
    return feats


def train_val_test_split_by_time(X: np.ndarray, y: np.ndarray,
                                 test_ratio: float, val_ratio: float) -> Tuple:
    n = len(X)
    n_test = int(n * test_ratio)
    n_trainval = n - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_windows(data: np.ndarray, targets: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(targets[i + horizon - 1])
    return np.array(X), np.array(y)


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Works for returns or price deltas; for level predictions, compare diff
    da = (np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])).mean()
    return float(da)


# ------------- Load & Prepare Data -------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip().title() for c in df.columns]
    # Sort by date (ascending)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns at least: {required}. Found: {df.columns.tolist()}")
    return df


def prepare_datasets(cfg: Config):
    raw = load_data(cfg.DATA_CSV)
    feats = build_features(raw, cfg.USE_TECHNICALS)

    feature_cols = [c for c in feats.columns if c not in ("Date",)]
    # Choose target
    if cfg.TARGET_MODE == "return":
        feats["TARGET"] = feats["Close"].pct_change().shift(-1)  # next-day return
    else:
        feats["TARGET"] = feats["Close"].shift(-1)               # next-day close

    feats = feats.dropna().reset_index(drop=True)

    X_df = feats[feature_cols].copy()
    y_series = feats["TARGET"].copy()

    # Scale features (fit only on train portion later)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Temporarily fit on all data for shape; will refit on train below
    scaler.fit(X_df)
    X_all = scaler.transform(X_df)
    y_all = y_series.values.astype(np.float32)

    # Build sequence windows
    X_seq, y_seq = make_windows(X_all, y_all, cfg.LOOKBACK, cfg.HORIZON)

    # Split by time
    X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split_by_time(X_seq, y_seq, cfg.TEST_RATIO, cfg.VAL_RATIO)

    # Refit scaler on training *rows only* for correctness
    # (Need to inverse-transform later, so re-create train-only scaler)
    # We must reconstruct scaler on pre-window data matching the train segment rows.
    # Compute how many raw rows correspond to training windows:
    # Win
