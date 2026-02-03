"""
Machine Learning service for stock prediction.
Enhanced with better labels, more features, ensemble models, and walk-forward validation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
from datetime import datetime

# Try to import XGBoost and LightGBM (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Model cache
_models = {}
_scalers = {}
_feature_names = {}
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_features(df, threshold=2.0, use_3class=False):
    """
    Prepare feature matrix from stock data with indicators.

    Args:
        df: DataFrame with OHLCV and indicators
        threshold: Minimum % change to count as up/down (default 2%)
        use_3class: If True, use 3 classes (down=0, hold=1, up=2)

    Returns:
        X: Feature DataFrame
        y: Labels array
        valid_indices: Indices of valid samples (for walk-forward)
    """
    if df.empty or len(df) < 60:
        return None, None, None

    features = []
    labels = []
    valid_indices = []

    for i in range(60, len(df) - 5):  # Start at 60 for MA60
        row = df.iloc[i]

        # Skip if missing critical data
        if pd.isna(row.get('close')) or row.get('close', 0) <= 0:
            continue

        close = row['close']

        # Technical indicator features
        feat = {
            # Price vs MAs (short to long term)
            'price_vs_ma5': _safe_ratio(close, row.get('ma5'), close) - 1,
            'price_vs_ma10': _safe_ratio(close, row.get('ma10'), close) - 1,
            'price_vs_ma20': _safe_ratio(close, row.get('ma20'), close) - 1,
            'price_vs_ma60': _safe_ratio(close, row.get('ma60'), close) - 1,

            # Long-term MAs (if available)
            'price_vs_ma120': _safe_ratio(close, row.get('ma120'), close) - 1,
            'price_vs_ma250': _safe_ratio(close, row.get('ma250'), close) - 1,

            # MA alignment (trend strength)
            'ma5_vs_ma10': _safe_ratio(row.get('ma5'), row.get('ma10'), 1) - 1,
            'ma5_vs_ma20': _safe_ratio(row.get('ma5'), row.get('ma20'), 1) - 1,
            'ma10_vs_ma20': _safe_ratio(row.get('ma10'), row.get('ma20'), 1) - 1,
            'ma20_vs_ma60': _safe_ratio(row.get('ma20'), row.get('ma60'), 1) - 1,

            # MA slope (trend direction)
            'ma5_slope': _calc_slope(df, i, 'ma5', 5),
            'ma20_slope': _calc_slope(df, i, 'ma20', 5),

            # MACD features
            'macd_dif': _safe_val(row.get('macd_dif'), 0),
            'macd_dea': _safe_val(row.get('macd_dea'), 0),
            'macd_hist': _safe_val(row.get('macd_hist'), 0),
            'macd_hist_slope': _calc_slope(df, i, 'macd_hist', 3),

            # RSI with zones
            'rsi': _safe_val(row.get('rsi'), 50),
            'rsi_oversold': 1 if _safe_val(row.get('rsi'), 50) < 30 else 0,
            'rsi_overbought': 1 if _safe_val(row.get('rsi'), 50) > 70 else 0,
            'rsi_neutral': 1 if 40 <= _safe_val(row.get('rsi'), 50) <= 60 else 0,

            # KDJ
            'kdj_k': _safe_val(row.get('kdj_k'), 50),
            'kdj_d': _safe_val(row.get('kdj_d'), 50),
            'kdj_j': _safe_val(row.get('kdj_j'), 50),
            'kdj_golden_cross': 1 if _safe_val(row.get('kdj_k'), 50) > _safe_val(row.get('kdj_d'), 50) else 0,

            # Bollinger Bands
            'boll_pct_b': _safe_val(row.get('boll_pct_b'), 0.5),
            'boll_width': _safe_val(row.get('boll_width'), 10),
            'boll_squeeze': 1 if _safe_val(row.get('boll_width'), 10) < 5 else 0,

            # Volume features
            'volume_ratio': _safe_val(row.get('volume_ratio'), 1),
            'volume_trend': _calc_volume_trend(df, i, 5),
            'high_volume': 1 if _safe_val(row.get('volume_ratio'), 1) > 2 else 0,

            # ATR (volatility)
            'atr_pct': (_safe_val(row.get('atr'), 0) / close * 100) if close > 0 else 0,
            'volatility_regime': _get_volatility_regime(df, i),

            # Momentum (past returns)
            'ret_1d': _safe_val(row.get('change_pct'), 0),
            'ret_5d': _calc_return(df, i, 5),
            'ret_10d': _calc_return(df, i, 10),
            'ret_20d': _calc_return(df, i, 20),

            # Momentum acceleration
            'momentum_accel': _calc_return(df, i, 5) - _calc_return(df, i-5, 5) if i >= 10 else 0,

            # Price patterns
            'higher_high': 1 if _is_higher_high(df, i) else 0,
            'lower_low': 1 if _is_lower_low(df, i) else 0,

            # Day of week (0=Monday, 4=Friday)
            'day_of_week': _get_day_of_week(row),
            'is_monday': 1 if _get_day_of_week(row) == 0 else 0,
            'is_friday': 1 if _get_day_of_week(row) == 4 else 0,
        }

        features.append(feat)
        valid_indices.append(i)

        # Calculate future return for label
        future_close = df.iloc[i + 5]['close']
        future_ret = (future_close - close) / close * 100

        # Threshold-based labeling
        if use_3class:
            if future_ret > threshold:
                labels.append(2)  # Up
            elif future_ret < -threshold:
                labels.append(0)  # Down
            else:
                labels.append(1)  # Hold
        else:
            # Binary with threshold - skip uncertain samples
            if abs(future_ret) < threshold:
                # Remove last feature since we're skipping this sample
                features.pop()
                valid_indices.pop()
                continue
            labels.append(1 if future_ret > threshold else 0)

    if not features or len(features) < 30:
        return None, None, None

    X = pd.DataFrame(features).fillna(0)
    y = np.array(labels)

    return X, y, valid_indices


def _safe_val(val, default):
    """Safely get value, return default if None or NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def _safe_ratio(numerator, denominator, default):
    """Safely calculate ratio."""
    num = _safe_val(numerator, None)
    den = _safe_val(denominator, None)
    if num is None or den is None or den == 0:
        return default
    return num / den


def _calc_slope(df, i, column, periods):
    """Calculate slope of a column over periods."""
    if i < periods or column not in df.columns:
        return 0
    try:
        current = _safe_val(df.iloc[i].get(column), None)
        past = _safe_val(df.iloc[i - periods].get(column), None)
        if current is None or past is None or past == 0:
            return 0
        return (current - past) / abs(past) * 100
    except:
        return 0


def _calc_return(df, i, periods):
    """Calculate return over periods."""
    if i < periods:
        return 0
    try:
        current = df.iloc[i]['close']
        past = df.iloc[i - periods]['close']
        if past == 0:
            return 0
        return (current - past) / past * 100
    except:
        return 0


def _calc_volume_trend(df, i, periods):
    """Calculate volume trend (current vs average)."""
    if i < periods:
        return 1
    try:
        current_vol = df.iloc[i]['volume']
        avg_vol = df.iloc[i-periods:i]['volume'].mean()
        if avg_vol == 0:
            return 1
        return current_vol / avg_vol
    except:
        return 1


def _get_volatility_regime(df, i, lookback=20):
    """Determine volatility regime: 0=low, 1=normal, 2=high."""
    if i < lookback:
        return 1
    try:
        returns = df.iloc[i-lookback:i]['change_pct'].dropna()
        if len(returns) < 10:
            return 1
        std = returns.std()
        if std < 1.5:
            return 0  # Low volatility
        elif std > 3:
            return 2  # High volatility
        return 1  # Normal
    except:
        return 1


def _is_higher_high(df, i, periods=5):
    """Check if current high is higher than recent highs."""
    if i < periods:
        return False
    try:
        current_high = df.iloc[i]['high']
        past_high = df.iloc[i-periods:i]['high'].max()
        return current_high > past_high
    except:
        return False


def _is_lower_low(df, i, periods=5):
    """Check if current low is lower than recent lows."""
    if i < periods:
        return False
    try:
        current_low = df.iloc[i]['low']
        past_low = df.iloc[i-periods:i]['low'].min()
        return current_low < past_low
    except:
        return False


def _get_day_of_week(row):
    """Get day of week from trade_date."""
    try:
        date_str = str(row.get('trade_date', ''))
        if date_str:
            dt = pd.to_datetime(date_str)
            return dt.dayofweek
    except:
        pass
    return 2  # Default to Wednesday


def walk_forward_validate(model, X, y, n_splits=5):
    """
    Perform walk-forward validation (time-series cross-validation).
    Always trains on past data and tests on future data.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(X_train) < 30 or len(X_test) < 5:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    return scores if scores else [0.5]


def create_ensemble_model():
    """Create an ensemble of multiple models."""
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )),
    ]

    # Add XGBoost if available
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )))

    # Add LightGBM if available
    if HAS_LIGHTGBM:
        estimators.append(('lgb', LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )))

    return VotingClassifier(estimators=estimators, voting='soft')


def train_model(stock_code, df, threshold=2.0, use_ensemble=True):
    """
    Train a model for a specific stock with improved methodology.

    Args:
        stock_code: Stock code
        df: DataFrame with OHLCV and indicators
        threshold: Minimum % change threshold for labels
        use_ensemble: Whether to use ensemble of models
    """
    X, y, indices = prepare_features(df, threshold=threshold)

    if X is None or len(X) < 50:
        return None, None, "Insufficient data for training (need at least 50 samples)"

    # Check class balance
    positive_rate = y.mean()
    if positive_rate < 0.2 or positive_rate > 0.8:
        # Highly imbalanced, might need adjustment
        pass

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Create model
    if use_ensemble:
        model = create_ensemble_model()
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )

    # Walk-forward validation (proper time-series CV)
    wf_scores = walk_forward_validate(model, X_scaled, y, n_splits=5)

    # Train final model on all data
    model.fit(X_scaled, y)

    # Cache model and scaler
    _models[stock_code] = model
    _scalers[stock_code] = scaler
    _feature_names[stock_code] = list(X.columns)

    # Save to disk
    save_model(stock_code, model, scaler, list(X.columns))

    # Get feature importance (for RF-based models)
    importance = _get_feature_importance(model, X.columns)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

    return model, scaler, {
        'cv_accuracy': round(np.mean(wf_scores) * 100, 1),
        'cv_std': round(np.std(wf_scores) * 100, 1),
        'samples': len(X),
        'positive_rate': round(positive_rate * 100, 1),
        'top_features': top_features,
        'threshold': threshold,
        'validation_type': 'walk-forward',
        'ensemble': use_ensemble,
        'models_used': _get_model_names(model)
    }


def _get_feature_importance(model, columns):
    """Extract feature importance from model."""
    importance = {}
    try:
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(columns, model.feature_importances_))
        elif hasattr(model, 'estimators_'):
            # Ensemble model - average importance
            importances = []
            for name, est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                importance = dict(zip(columns, avg_importance))
    except:
        importance = {col: 0 for col in columns}
    return importance


def _get_model_names(model):
    """Get names of models in ensemble."""
    if hasattr(model, 'estimators_'):
        return [name for name, _ in model.estimators_]
    return [type(model).__name__]


def predict(stock_code, df, threshold=2.0):
    """Make prediction for a stock using trained model."""
    # Try to load from disk if not in cache
    if stock_code not in _models:
        loaded = load_model(stock_code)
        if not loaded:
            return None, "Model not trained"

    model = _models[stock_code]
    scaler = _scalers[stock_code]
    feature_names = _feature_names.get(stock_code, None)

    X, _, _ = prepare_features(df, threshold=threshold)
    if X is None or len(X) == 0:
        return None, "Cannot prepare features"

    # Ensure feature alignment
    if feature_names:
        missing_cols = set(feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[feature_names]

    # Use only the latest data point
    X_latest = X.iloc[[-1]]
    X_scaled = scaler.transform(X_latest)

    # Predict probability
    prob = model.predict_proba(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]

    # Handle binary vs multi-class
    if len(prob) == 2:
        return {
            'prediction': 'bullish' if prediction == 1 else 'bearish',
            'confidence': round(max(prob) * 100, 1),
            'prob_up': round(prob[1] * 100, 1),
            'prob_down': round(prob[0] * 100, 1),
        }, None
    else:
        # 3-class: down, hold, up
        pred_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
        return {
            'prediction': pred_map.get(prediction, 'neutral'),
            'confidence': round(max(prob) * 100, 1),
            'prob_up': round(prob[2] * 100, 1) if len(prob) > 2 else 0,
            'prob_down': round(prob[0] * 100, 1),
            'prob_hold': round(prob[1] * 100, 1) if len(prob) > 1 else 0,
        }, None


def save_model(stock_code, model, scaler, feature_names):
    """Save model to disk."""
    try:
        model_path = os.path.join(MODEL_DIR, f'{stock_code}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{stock_code}_scaler.pkl')
        features_path = os.path.join(MODEL_DIR, f'{stock_code}_features.pkl')

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, features_path)
        return True
    except Exception as e:
        print(f"Error saving model for {stock_code}: {e}")
        return False


def load_model(stock_code):
    """Load model from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, f'{stock_code}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{stock_code}_scaler.pkl')
        features_path = os.path.join(MODEL_DIR, f'{stock_code}_features.pkl')

        if not os.path.exists(model_path):
            return False

        _models[stock_code] = joblib.load(model_path)
        _scalers[stock_code] = joblib.load(scaler_path)
        if os.path.exists(features_path):
            _feature_names[stock_code] = joblib.load(features_path)
        return True
    except Exception as e:
        print(f"Error loading model for {stock_code}: {e}")
        return False


def train_general_model(stocks_data, threshold=2.0):
    """Train a general model using data from multiple stocks."""
    all_X = []
    all_y = []

    for stock_code, df in stocks_data.items():
        X, y, _ = prepare_features(df, threshold=threshold)
        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return None, None, "No valid data"

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)

    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Use ensemble model
    model = create_ensemble_model()

    # Walk-forward validation
    wf_scores = walk_forward_validate(model, X_scaled, y, n_splits=5)

    # Train on all data
    model.fit(X_scaled, y)

    # Cache as general model
    _models['_general'] = model
    _scalers['_general'] = scaler
    _feature_names['_general'] = list(X.columns)

    # Save to disk
    save_model('_general', model, scaler, list(X.columns))

    importance = _get_feature_importance(model, X.columns)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

    return model, scaler, {
        'cv_accuracy': round(np.mean(wf_scores) * 100, 1),
        'cv_std': round(np.std(wf_scores) * 100, 1),
        'samples': len(X),
        'stocks_used': len(stocks_data),
        'positive_rate': round(y.mean() * 100, 1),
        'top_features': top_features,
        'threshold': threshold,
        'validation_type': 'walk-forward',
        'models_used': _get_model_names(model)
    }


def predict_with_general_model(df, threshold=2.0):
    """Predict using the general model."""
    if '_general' not in _models:
        loaded = load_model('_general')
        if not loaded:
            return None, "General model not trained"

    return predict('_general', df, threshold=threshold)


def get_model_info():
    """Get information about available models."""
    info = {
        'cached_models': list(_models.keys()),
        'has_xgboost': HAS_XGBOOST,
        'has_lightgbm': HAS_LIGHTGBM,
        'model_dir': MODEL_DIR,
    }

    # Check for saved models
    if os.path.exists(MODEL_DIR):
        saved = [f.replace('_model.pkl', '') for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]
        info['saved_models'] = saved

    return info
