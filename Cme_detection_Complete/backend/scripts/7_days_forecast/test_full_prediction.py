import sys
sys.path.insert(0, '.')
from forecast_model_runner import ForecastModelRunner

print("Testing full prediction flow...")
r = ForecastModelRunner()

print("\n1. Loading model...")
r.load_model()
print(f"   Model input shape: {r.model.input_shape}")
expected_features = r.model.input_shape[2] if len(r.model.input_shape) > 2 else None
print(f"   Expected features: {expected_features}")

print("\n2. Loading scaler...")
r.load_scaler()

print("\n3. Getting features...")
r.get_feature_columns()
print(f"   Feature count: {len(r.feature_cols)}")

if expected_features and len(r.feature_cols) != expected_features:
    print(f"   ⚠️  WARNING: Feature count mismatch! Model expects {expected_features}, but we have {len(r.feature_cols)}")

print("\n4. Fetching data...")
data = r.fetch_recent_data(days=7)

print("\n5. Preparing features...")
features, timestamps = r.prepare_features(data)

print("\n6. Making prediction...")
try:
    result = r.make_predictions()
    print(f"\n✅ SUCCESS! Predictions shape: {result.shape}")
    print(f"   Columns: {list(result.columns)}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()





