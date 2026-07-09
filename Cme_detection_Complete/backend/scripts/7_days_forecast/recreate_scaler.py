import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Load training data
train_unscaled_path = 'train_data_unscaled.csv'
if not os.path.exists(train_unscaled_path):
    print(f"❌ Training data not found: {train_unscaled_path}")
    exit(1)

print(f"Loading training data from {train_unscaled_path}...")
train_data = pd.read_csv(train_unscaled_path)

# Exclude non-feature columns
exclude_cols = ['Datetime', 'Year', 'DOY', 'Hour']
target_vars = ['Dst_Index_nT', 'ap_index_nT', 'Sunspot_Number', 'Kp_10']

# Get all columns that should be scaled (features + targets)
all_cols = [col for col in train_data.columns if col not in exclude_cols]
print(f"Total columns to scale: {len(all_cols)}")
print(f"Columns: {all_cols}")

# Prepare data for scaling
X = train_data[all_cols].fillna(0).values
print(f"Data shape: {X.shape}")

# Fit scaler
print("\nFitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

print(f"✓ Scaler fitted successfully")
print(f"  Mean shape: {scaler.mean_.shape}")
print(f"  Scale shape: {scaler.scale_.shape}")
print(f"  Mean range: [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
print(f"  Scale range: [{scaler.scale_.min():.2f}, {scaler.scale_.max():.2f}]")

# Save scaler
scaler_path = 'scaler.pkl'
print(f"\nSaving scaler to {scaler_path}...")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ Scaler saved successfully!")

# Verify it can be loaded
print("\nVerifying scaler can be loaded...")
with open(scaler_path, 'rb') as f:
    loaded_scaler = pickle.load(f)

print(f"✓ Scaler loaded successfully")
print(f"  Type: {type(loaded_scaler)}")
print(f"  Has transform: {hasattr(loaded_scaler, 'transform')}")
print(f"  Mean shape: {loaded_scaler.mean_.shape}")
print(f"  Scale shape: {loaded_scaler.scale_.shape}")

# Test transform
print("\nTesting transform...")
test_data = X[:10]
scaled = loaded_scaler.transform(test_data)
print(f"✓ Transform works! Input shape: {test_data.shape}, Output shape: {scaled.shape}")

print("\n" + "="*70)
print("✅ SCALER RECREATED SUCCESSFULLY")
print("="*70)





