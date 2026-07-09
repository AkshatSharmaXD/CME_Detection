"""
Manually map and load weights from H5 file
"""
import zipfile
import h5py
import io
import numpy as np

model_path = 'models/lstm_baseline_dst.keras'

print("Inspecting weights structure...")
with zipfile.ZipFile(model_path, 'r') as z:
    weights_data = z.read('model.weights.h5')
    f = h5py.File(io.BytesIO(weights_data), 'r')
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
    
    print("Weights in file:")
    f.visititems(print_structure)
    
    # Check layers structure
    if 'layers' in f:
        print("\nLayers structure:")
        for layer_name in f['layers'].keys():
            print(f"  Layer: {layer_name}")
            layer_group = f['layers'][layer_name]
            for key in layer_group.keys():
                print(f"    {key}: {list(layer_group[key].keys()) if hasattr(layer_group[key], 'keys') else 'data'}")





