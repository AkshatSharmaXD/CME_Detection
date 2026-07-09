"""
Quick Manual Check - Run this to verify Sunspot Number
"""
import pandas as pd

print('='*70)
print('QUICK MANUAL CHECK: SUNSPOT NUMBER')
print('='*70)

# Check 1: Data file
print('\n1. Check if Sunspot_Number exists in data file:')
df = pd.read_csv('omni_data_updatedyears.csv', nrows=1)
has_sunspot = 'Sunspot_Number' in df.columns
print(f'   Result: {has_sunspot} ✅' if has_sunspot else '   Result: False ❌')

# Check 2: Training data
print('\n2. Check if Sunspot_Number in training data:')
train_df = pd.read_csv('train_data_scaled.csv', nrows=1)
has_in_training = 'Sunspot_Number' in train_df.columns
print(f'   Result: {has_in_training} ✅' if has_in_training else '   Result: False ❌')

# Check 3: Is it OUTPUT?
print('\n3. Check if Sunspot_Number is OUTPUT (predicted):')
targets = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
is_output = 'Sunspot_Number' in targets
print(f'   Target variables: {targets}')
print(f'   Sunspot_Number is OUTPUT: {is_output} ❌' if not is_output else f'   Sunspot_Number is OUTPUT: {is_output} ✅')

# Check 4: Is it INPUT (feature)?
print('\n4. Check if Sunspot_Number is INPUT (feature):')
exclude = ['Datetime', 'Year', 'DOY', 'Hour'] + targets
features = [c for c in train_df.columns if c not in exclude]
is_feature = 'Sunspot_Number' in features
print(f'   Total features: {len(features)}')
print(f'   Sunspot_Number is FEATURE: {is_feature} ✅' if is_feature else f'   Sunspot_Number is FEATURE: {is_feature} ❌')
if is_feature:
    position = features.index('Sunspot_Number') + 1
    print(f'   Position: Feature #{position} of {len(features)}')

# Summary
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
if is_feature:
    print(f'✅ Sunspot_Number IS in the model')
    print(f'   → Used as INPUT FEATURE (Feature #{position})')
    print(f'   → Helps predict: DST, Kp, Ap')
    print(f'   → NOT predicted as OUTPUT')
else:
    print('❌ Sunspot_Number is NOT in the model')
print('='*70)

