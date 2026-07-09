"""
Manual model loading - load architecture and weights separately
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

print("Attempting manual model load...")

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
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Try loading from directory
    print("Loading from extracted directory...")
    with custom_object_scope(custom_objects):
        model = keras.models.load_model(temp_dir, compile=False)
    
    print("SUCCESS! Model loaded")
    print(f"Input: {model.input_shape}, Output: {model.output_shape}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)





