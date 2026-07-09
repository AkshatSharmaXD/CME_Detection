"""
Fix model compatibility issues for Keras 2.13
"""
import os
import json
import zipfile
import tempfile
import shutil
import sys

model_path = os.path.join(os.path.dirname(__file__), 'models', 'lstm_baseline_dst.keras')
fixed_model_path = os.path.join(os.path.dirname(__file__), 'models', 'lstm_baseline_dst_fixed.keras')

print("="*70)
print("FIXING MODEL COMPATIBILITY")
print("="*70)
print(f"Model: {model_path}")
print(f"Output: {fixed_model_path}")

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

# Create temp directory
temp_dir = tempfile.mkdtemp()
print(f"Temp dir: {temp_dir}")

try:
    # Extract model
    print("\n1. Extracting model...")
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print("   [OK] Extracted")
    
    # Fix function
    def fix_config(obj, path=""):
        fixes = []
        if isinstance(obj, dict):
            # Fix batch_shape
            if 'batch_shape' in obj:
                batch_shape = obj.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    obj['input_shape'] = batch_shape[1:]
                    fixes.append(f"batch_shape -> input_shape at {path}")
            
                    # DON'T modify DTypePolicy - it will be handled by custom_objects
            # Only fix batch_shape to preserve weight loading
            
            # Recursively fix
            for key, value in obj.items():
                fixes.extend(fix_config(value, f"{path}.{key}" if path else key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                fixes.extend(fix_config(item, f"{path}[{i}]" if path else f"[{i}]"))
        return fixes
    
    # Fix all JSON files
    print("\n2. Fixing config files...")
    all_fixes = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    fixes = fix_config(config)
                    if fixes:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)
                        all_fixes.extend(fixes)
                        print(f"   [OK] Fixed {file}: {len(fixes)} fixes")
                except Exception as e:
                    print(f"   ⚠️  Error fixing {file}: {e}")
    
    print(f"\n   Total fixes: {len(all_fixes)}")
    for fix in all_fixes[:10]:  # Show first 10
        print(f"      - {fix}")
    if len(all_fixes) > 10:
        print(f"      ... and {len(all_fixes) - 10} more")
    
    # Repackage - preserve ALL files including weights
    print("\n3. Repackaging model...")
    with zipfile.ZipFile(fixed_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zip_ref.write(file_path, arcname)
    
    # Verify all files are there
    with zipfile.ZipFile(fixed_model_path, 'r') as zip_ref:
        files_in_zip = zip_ref.namelist()
        print(f"   Files in fixed model: {len(files_in_zip)}")
        if 'config.json' in files_in_zip:
            print("   [OK] config.json present")
        weight_files = [f for f in files_in_zip if 'weights' in f.lower() or f.endswith('.weights.h5')]
        print(f"   Weight files: {len(weight_files)}")
    
    print(f"   [OK] Saved to {fixed_model_path}")
    
    # Test load with custom_objects for DTypePolicy
    print("\n4. Testing fixed model...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from keras.mixed_precision import Policy
        from keras.utils import custom_object_scope
        
        custom_objects = {'DTypePolicy': Policy}
        with custom_object_scope(custom_objects):
            model = keras.models.load_model(fixed_model_path, compile=False)
        print(f"   [OK] Model loads successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print("\n" + "="*70)
        print("[SUCCESS] MODEL FIXED SUCCESSFULLY!")
        print("="*70)
        print(f"\nFixed model saved to: {fixed_model_path}")
        print("You can now use this model instead of the original.")
    except Exception as e:
        print(f"   ❌ Failed to load fixed model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

