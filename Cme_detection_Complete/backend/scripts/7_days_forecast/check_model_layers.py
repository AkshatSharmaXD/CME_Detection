"""
Check actual model layer names and shapes
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

temp_dir = tempfile.mkdtemp()
try:
    with zipfile.ZipFile(model_path, 'r') as z:
        z.extractall(temp_dir)
    
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
    
    with custom_object_scope(custom_objects):
        model_json = json.dumps(config)
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
    
    print("Model layers and their expected weights:")
    for i, layer in enumerate(model.layers):
        print(f"\n{i}. {layer.name} ({type(layer).__name__})")
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            print(f"   Expected {len(weights)} weight arrays:")
            for j, w in enumerate(weights):
                print(f"     [{j}]: shape={w.shape}, dtype={w.dtype}")
        else:
            print(f"   No weights (e.g., InputLayer, Dropout)")

finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)





