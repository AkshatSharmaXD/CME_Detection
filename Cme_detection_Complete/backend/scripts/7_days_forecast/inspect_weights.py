"""
Inspect model weights structure in detail
"""
import zipfile
import h5py
import io
import json

model_path = 'models/lstm_baseline_dst.keras'

print("="*70)
print("MODEL WEIGHTS INSPECTION")
print("="*70)

# Get config to see layer structure
with zipfile.ZipFile(model_path, 'r') as z:
    config = json.loads(z.read('config.json'))
    print("\nModel layers (from config):")
    for i, layer in enumerate(config.get('layers', [])):
        print(f"  {i}. {layer.get('class_name', 'Unknown')} - name: {layer.get('config', {}).get('name', 'N/A')}")
    
    print("\n" + "="*70)
    print("Weights structure in H5 file:")
    print("="*70)
    
    weights_data = z.read('model.weights.h5')
    f = h5py.File(io.BytesIO(weights_data), 'r')
    
    if 'layers' in f:
        for layer_name in sorted(f['layers'].keys()):
            print(f"\nLayer: {layer_name}")
            layer_group = f['layers'][layer_name]
            
            if 'vars' in layer_group:
                vars_group = layer_group['vars']
                print(f"  vars: {len(vars_group.keys())} weight arrays")
                for i in sorted([int(k) for k in vars_group.keys()]):
                    weight = vars_group[str(i)]
                    print(f"    [{i}]: shape={weight.shape}, dtype={weight.dtype}")
            
            if 'cell' in layer_group:
                cell_group = layer_group['cell']
                if 'vars' in cell_group:
                    vars_group = cell_group['vars']
                    print(f"  cell/vars: {len(vars_group.keys())} weight arrays")
                    for i in sorted([int(k) for k in vars_group.keys()]):
                        weight = vars_group[str(i)]
                        print(f"    [{i}]: shape={weight.shape}, dtype={weight.dtype}")





