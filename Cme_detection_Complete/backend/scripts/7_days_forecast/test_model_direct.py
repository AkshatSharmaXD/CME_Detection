"""
Direct test script to fetch values from the model
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directories to path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

print("="*70)
print("DIRECT MODEL TEST - FETCHING VALUES FROM MODEL")
print("="*70)

try:
    from forecast_model_runner import ForecastModelRunner
    
    print("\n1. Creating ForecastModelRunner...")
    runner = ForecastModelRunner()
    
    print("\n2. Loading model...")
    if not runner.load_model():
        print("❌ Failed to load model")
        sys.exit(1)
    
    print("\n3. Loading scaler...")
    if not runner.load_scaler():
        print("❌ Failed to load scaler")
        sys.exit(1)
    
    print("\n4. Getting feature columns...")
    if not runner.get_feature_columns():
        print("❌ Failed to get feature columns")
        sys.exit(1)
    
    print("\n5. Fetching recent data...")
    data = runner.fetch_recent_data(days=7)
    if data is None or data.empty:
        print("❌ Failed to fetch data")
        print("   Trying to create dummy data for testing...")
        # Create dummy data for testing
        dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
        dummy_data = pd.DataFrame({
            'timestamp': dates,
            'bt': np.random.uniform(3, 8, 168),
            'bz_gsm': np.random.uniform(-5, 5, 168),
            'density': np.random.uniform(2, 10, 168),
            'speed': np.random.uniform(300, 600, 168),
            'lon_gsm': np.random.uniform(-180, 180, 168),
            'lat_gsm': np.random.uniform(-90, 90, 168),
            'sunspot_number': np.random.uniform(0, 200, 168),
            'f10_7': np.random.uniform(70, 200, 168),
        })
        dummy_data.set_index('timestamp', inplace=True)
        data = dummy_data
        print(f"   ✓ Created dummy data: {len(data)} rows")
    
    print(f"   ✓ Data fetched: {len(data)} rows")
    
    print("\n6. Preparing features...")
    feature_data, timestamps = runner.prepare_features(data)
    if feature_data is None:
        print("❌ Failed to prepare features")
        sys.exit(1)
    print(f"   ✓ Features prepared: {len(feature_data)} rows, {len(feature_data.columns)} features")
    
    print("\n7. Preparing input sequence...")
    if len(feature_data) < runner.lookback:
        print(f"   Padding data from {len(feature_data)} to {runner.lookback}...")
        last_row = feature_data.iloc[-1:].copy()
        padding_needed = runner.lookback - len(feature_data)
        for _ in range(padding_needed):
            feature_data = pd.concat([feature_data, last_row], ignore_index=True)
    
    input_data = feature_data.iloc[-runner.lookback:].values
    input_scaled = runner.scaler.transform(input_data)
    input_sequence = input_scaled.reshape(1, runner.lookback, len(runner.feature_cols))
    print(f"   ✓ Input sequence shape: {input_sequence.shape}")
    
    print("\n8. CALLING MODEL.PREDICT()...")
    print("   This is the actual model call - generating predictions...")
    predictions_scaled = runner.model.predict(input_sequence, verbose=1)
    print(f"   ✓ Model returned predictions shape: {predictions_scaled.shape}")
    print(f"   ✓ Raw predictions sample: {predictions_scaled.flatten()[:10]}")
    
    print("\n9. Processing predictions...")
    # Handle output shape
    if len(predictions_scaled.shape) == 3:
        predictions_scaled = predictions_scaled[0]
    elif len(predictions_scaled.shape) == 2:
        if predictions_scaled.shape[0] == 1:
            predictions_scaled = predictions_scaled.reshape(runner.forecast_horizon, -1)
    elif len(predictions_scaled.shape) == 1:
        predictions_scaled = predictions_scaled.reshape(runner.forecast_horizon, -1)
    
    actual_targets = predictions_scaled.shape[1] if len(predictions_scaled.shape) > 1 else 1
    print(f"   Actual targets: {actual_targets}")
    
    if actual_targets != len(runner.target_vars):
        if actual_targets == 3:
            runner.target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
            print(f"   Adjusted to 3 targets: {runner.target_vars}")
    
    predictions_scaled = predictions_scaled.reshape(runner.forecast_horizon, len(runner.target_vars))
    
    print("\n10. Inverse transforming...")
    # Simple inverse transform - try to get target stats from unscaled data
    try:
        train_unscaled_path = os.path.join(runner.script_dir, 'train_data_unscaled.csv')
        if os.path.exists(train_unscaled_path):
            train_unscaled = pd.read_csv(train_unscaled_path)
            all_cols = [col for col in train_unscaled.columns if col not in ['Datetime', 'Year', 'DOY', 'Hour']]
            
            predictions = np.zeros((runner.forecast_horizon, len(runner.target_vars)))
            for i, target in enumerate(runner.target_vars):
                if target in all_cols:
                    target_idx = all_cols.index(target)
                    if target_idx < len(runner.scaler.mean_):
                        mean = runner.scaler.mean_[target_idx]
                        std = runner.scaler.scale_[target_idx]
                        predictions[:, i] = predictions_scaled[:, i] * std + mean
                        print(f"   ✓ {target}: mean={mean:.2f}, std={std:.2f}")
                    else:
                        predictions[:, i] = predictions_scaled[:, i]
                else:
                    predictions[:, i] = predictions_scaled[:, i]
        else:
            print("   ⚠️  train_data_unscaled.csv not found, using scaled values")
            predictions = predictions_scaled
    except Exception as e:
        print(f"   ⚠️  Inverse transform failed: {e}, using scaled values")
        predictions = predictions_scaled
    
    print("\n11. Creating result DataFrame...")
    current_time = datetime.now()
    forecast_start = current_time + timedelta(hours=1)
    forecast_timestamps = pd.date_range(
        start=forecast_start,
        periods=runner.forecast_horizon,
        freq='H'
    )
    
    result_df = pd.DataFrame(
        predictions,
        index=forecast_timestamps,
        columns=runner.target_vars
    )
    
    print("\n" + "="*70)
    print("✅ MODEL PREDICTIONS SUCCESSFULLY GENERATED")
    print("="*70)
    print(f"Rows: {len(result_df)}")
    print(f"Columns: {list(result_df.columns)}")
    print(f"Date range: {result_df.index[0]} to {result_df.index[-1]}")
    print(f"\nFirst 5 rows:")
    print(result_df.head())
    print(f"\nLast 5 rows:")
    print(result_df.tail())
    print(f"\nValue statistics:")
    print(result_df.describe())
    print(f"\nSample values (first row):")
    for col in result_df.columns:
        print(f"  {col}: {result_df[col].iloc[0]:.6f}")
    print(f"\nSample values (last row):")
    for col in result_df.columns:
        print(f"  {col}: {result_df[col].iloc[-1]:.6f}")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

