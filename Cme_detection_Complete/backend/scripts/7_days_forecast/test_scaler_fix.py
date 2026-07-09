import sys
sys.path.insert(0, '.')
from forecast_model_runner import ForecastModelRunner

r = ForecastModelRunner()
print('1. Loading model...')
r.load_model()
print('2. Loading scaler...')
r.load_scaler()
print(f'   Scaler type: {type(r.scaler)}')
print(f'   Has transform: {hasattr(r.scaler, "transform")}')
print('3. Getting features...')
r.get_feature_columns()
print(f'   Features: {len(r.feature_cols)}')
print('4. Fetching data...')
data = r.fetch_recent_data(days=7)
print(f'   Data rows: {len(data)}')
print('5. Preparing features...')
features, timestamps = r.prepare_features(data)
print(f'   Features shape: {features.shape}')
print('6. Testing scaler transform...')
input_data = features.iloc[-72:].values
scaled = r.scaler.transform(input_data)
print(f'   Scaled shape: {scaled.shape}')
print('\n✓ All steps completed!')





