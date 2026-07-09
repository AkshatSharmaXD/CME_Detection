"""Check the LSTM model structure and details"""
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import InputLayer
from keras.mixed_precision import Policy
from keras.utils import custom_object_scope
import numpy as np
import json
import zipfile
import tempfile
import shutil

# Custom InputLayer to handle batch_shape -> input_shape conversion
class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Convert batch_shape to input_shape if present
        if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]  # Remove batch dimension
        super().__init__(*args, **kwargs)

# Custom deserializer for DTypePolicy
def deserialize_dtype_policy(config=None, **kwargs):
    """Convert old DTypePolicy to new Policy format"""
    if config is None:
        config = kwargs
    if isinstance(config, dict):
        dtype = config.get('dtype', config.get('name', 'float32'))
        name = config.get('name', dtype)
    else:
        name = 'float32'
    return Policy(name)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'lstm_baseline_dst.keras')

print("="*70)
print("LSTM MODEL ANALYSIS")
print("="*70)

if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    exit(1)

print(f"✓ Model file found: {model_path}")
print(f"  File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB\n")

# Load model with compatibility fixes (same approach as forecast_model_runner.py)
print("Loading model...")
custom_objects = {
    'DTypePolicy': deserialize_dtype_policy,
    'InputLayer': CompatibleInputLayer,
}

# Extract model and fix config
temp_dir = tempfile.mkdtemp()
try:
    with zipfile.ZipFile(model_path, 'r') as z:
        z.extractall(temp_dir)
    
    # Fix config: convert batch_shape to input_shape
    config_path = os.path.join(temp_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    def fix_config(obj):
        if isinstance(obj, dict):
            if 'batch_shape' in obj:
                batch_shape = obj.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    obj['input_shape'] = batch_shape[1:]
            for v in obj.values():
                fix_config(v)
        elif isinstance(obj, list):
            for item in obj:
                fix_config(item)
    
    fix_config(config)
    
    # Load architecture from JSON with custom objects
    with custom_object_scope(custom_objects):
        model_json = json.dumps(config)
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
    
    # Load weights (optional - just for checking structure)
    weights_path = os.path.join(temp_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("✓ Model loaded successfully with weights\n")
        except:
            print("✓ Model architecture loaded (weights skipped)\n")
    else:
        print("✓ Model architecture loaded\n")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

# Model summary
print("="*70)
print("MODEL ARCHITECTURE")
print("="*70)
model.summary()

print("\n" + "="*70)
print("MODEL DETAILS")
print("="*70)
print(f"Input Shape: {model.input_shape}")
print(f"Output Shape: {model.output_shape}")
print(f"Total Parameters: {model.count_params():,}")
print(f"Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

print("\n" + "="*70)
print("LAYER BREAKDOWN")
print("="*70)
for i, layer in enumerate(model.layers):
    print(f"\n{i+1}. {layer.name}")
    print(f"   Type: {type(layer).__name__}")
    print(f"   Input Shape: {layer.input_shape}")
    print(f"   Output Shape: {layer.output_shape}")
    if hasattr(layer, 'units'):
        print(f"   Units: {layer.units}")
    if hasattr(layer, 'activation'):
        print(f"   Activation: {layer.activation}")
    if hasattr(layer, 'dropout'):
        print(f"   Dropout: {layer.dropout}")

print("\n" + "="*70)
print("TRAINING DATA INFO (from notebook)")
print("="*70)
print("Training Samples: 77,569 sequences")
print("Test Samples: 19,213 sequences")
print("Date Range: 1996-08-01 to 2025-05-31")
print("Total Years: 29 years of data (1996-2025)")
print("\nInput:")
print("  - Lookback: 72 hours (3 days)")
print("  - Features: 26 features")
print("  - Shape: (samples, 72, 26)")
print("\nOutput:")
print("  - Forecast: 168 hours (7 days)")
print("  - Targets: 3 (Dst_Index_nT, Kp_10, ap_index_nT)")
print("  - Shape: (samples, 168, 3)")

print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)
config = model.get_config()
print(f"Model Type: {type(model).__name__}")
if hasattr(model, 'optimizer'):
    print(f"Optimizer: {type(model.optimizer).__name__}")
if hasattr(model, 'loss'):
    print(f"Loss Function: {model.loss}")

print("\n" + "="*70)
print("✓ Model Analysis Complete")
print("="*70)

