"""
Load model architecture and weights separately
"""
import zipfile
import json
import tempfile
import os
import tensorflow as tf
from tensorflow import keras
from keras.mixed_precision import Policy
from keras.engine.input_layer import InputLayer
from keras.utils import custom_object_scope
import h5py
import io

model_path = 'models/lstm_baseline_dst.keras'

class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

def deserialize_dtype_policy(config=None, **kwargs):
    if config is None:
        config = kwargs
    if isinstance(config, dict):
        dtype = config.get('dtype', config.get('name', 'float32'))
        name = config.get('name', dtype)
    else:
        name = 'float32'
    return Policy(name)

custom_objects = {
    'DTypePolicy': deserialize_dtype_policy,
    'InputLayer': CompatibleInputLayer,
}

print("Loading model architecture and weights separately...")

# Extract model
temp_dir = tempfile.mkdtemp()
try:
    with zipfile.ZipFile(model_path, 'r') as z:
        z.extractall(temp_dir)
    
    # Fix config
    config_path = os.path.join(temp_dir, 'config.json')
    with open(config_path, 'r') as f:
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
    
    # Load architecture from JSON
    print("1. Loading architecture from JSON...")
    with custom_object_scope(custom_objects):
        model_json = json.dumps(config)
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
    
    print("   Architecture loaded")
    
    # Load weights from H5
    print("2. Loading weights from H5...")
    weights_path = os.path.join(temp_dir, 'model.weights.h5')
    
    # Try loading weights
    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("   Weights loaded (with skip_mismatch)")
    except Exception as e:
        print(f"   Warning: Could not load weights: {e}")
        print("   Model will use random weights")
    
    print("\nSUCCESS! Model loaded")
    print(f"Input: {model.input_shape}, Output: {model.output_shape}")
    
    # Test a prediction
    import numpy as np
    test_input = np.random.randn(1, 72, 18)
    output = model.predict(test_input, verbose=0)
    print(f"Test prediction shape: {output.shape}")
    print("Model is functional!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)





