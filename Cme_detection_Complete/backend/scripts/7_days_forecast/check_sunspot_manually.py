"""
Manual Check: Sunspot Number in Model
Run this script to verify if Sunspot Number is used in the model
"""
import pandas as pd
import os

print("="*70)
print("MANUAL CHECK: SUNSPOT NUMBER IN MODEL")
print("="*70)

# Check 1: Is Sunspot_Number in the data file?
print("\n1. Checking data file (omni_data_updatedyears.csv)...")
data_file = 'omni_data_updatedyears.csv'
if os.path.exists(data_file):
    df = pd.read_csv(data_file, nrows=10)
    has_sunspot = 'Sunspot_Number' in df.columns
    print(f"   ✓ Sunspot_Number column exists: {has_sunspot}")
    if has_sunspot:
        print(f"   ✓ Sample values: {df['Sunspot_Number'].head(5).tolist()}")
else:
    print(f"   ❌ Data file not found: {data_file}")

# Check 2: Is Sunspot_Number in training data?
print("\n2. Checking training data (train_data_scaled.csv)...")
train_file = 'train_data_scaled.csv'
if os.path.exists(train_file):
    train_df = pd.read_csv(train_file, nrows=5)
    has_sunspot_train = 'Sunspot_Number' in train_df.columns
    print(f"   ✓ Sunspot_Number in training data: {has_sunspot_train}")
    if has_sunspot_train:
        print(f"   ✓ Sample values: {train_df['Sunspot_Number'].head(5).tolist()}")
else:
    print(f"   ❌ Training file not found: {train_file}")

# Check 3: Is Sunspot_Number a FEATURE (input) or TARGET (output)?
print("\n3. Checking if Sunspot_Number is FEATURE or TARGET...")
target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
exclude_cols = ['Datetime', 'Year', 'DOY', 'Hour'] + target_vars

if os.path.exists(train_file):
    train_df = pd.read_csv(train_file, nrows=1)
    all_cols = train_df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    is_feature = 'Sunspot_Number' in feature_cols
    is_target = 'Sunspot_Number' in target_vars
    
    print(f"   ✓ Sunspot_Number is a FEATURE (input): {is_feature}")
    print(f"   ✓ Sunspot_Number is a TARGET (output): {is_target}")
    
    if is_feature:
        print(f"   → Sunspot_Number is used as INPUT to help predict DST, Kp, Ap")
        print(f"   → It's feature #{feature_cols.index('Sunspot_Number') + 1} of {len(feature_cols)} features")
    if is_target:
        print(f"   → Sunspot_Number is predicted as OUTPUT")
    if not is_feature and not is_target:
        print(f"   ❌ Sunspot_Number is NOT used in the model")

# Check 4: What are the actual TARGET variables?
print("\n4. Model OUTPUTS (what the model predicts):")
print(f"   Target Variables: {target_vars}")
print(f"   Total Targets: {len(target_vars)}")
print(f"   → Dst_Index_nT (Disturbance Storm Time)")
print(f"   → Kp_10 (Planetary K-index)")
print(f"   → ap_index_nT (Planetary A-index)")

# Check 5: Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if is_feature:
    print("✅ Sunspot_Number IS in the model")
    print("   → Used as INPUT FEATURE (helps predict DST, Kp, Ap)")
    print("   → NOT predicted as OUTPUT")
    print("\n   The model uses Sunspot_Number to help predict:")
    print("   - DST Index")
    print("   - Kp Index") 
    print("   - Ap Index")
    print("\n   But Sunspot_Number itself is NOT predicted.")
else:
    print("❌ Sunspot_Number is NOT in the model")

print("\n" + "="*70)
print("To add Sunspot Number as OUTPUT:")
print("1. Add 'Sunspot_Number' to target_vars list")
print("2. Retrain the model with 4 targets instead of 3")
print("3. Update output shape to (168, 4) instead of (168, 3)")
print("="*70)

