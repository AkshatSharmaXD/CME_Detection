import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

class NumpyCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module == 'numpy._core':
            module = 'numpy.core'
        return super().find_class(module, name)

with open('scaler.pkl', 'rb') as f:
    unpickler = NumpyCompatibleUnpickler(f)
    obj = unpickler.load()

print(f'Loaded object type: {type(obj)}')
print(f'Shape: {obj.shape if hasattr(obj, "shape") else "N/A"}')
print(f'Is array: {isinstance(obj, np.ndarray)}')

if isinstance(obj, np.ndarray):
    print(f'\nArray values (first 10): {obj[:10]}')
    print(f'Array min: {obj.min()}, max: {obj.max()}, mean: {obj.mean()}')
    
    # Check if there's a std file
    import os
    std_path = 'scaler_std.pkl'
    if os.path.exists(std_path):
        print(f'\nFound std file: {std_path}')
        with open(std_path, 'rb') as f:
            std_obj = NumpyCompatibleUnpickler(f).load()
            print(f'Std shape: {std_obj.shape}')
            # Create scaler
            scaler = StandardScaler()
            scaler.mean_ = obj
            scaler.scale_ = std_obj
            scaler.var_ = std_obj ** 2
            print(f'\nCreated StandardScaler with mean shape {scaler.mean_.shape} and scale shape {scaler.scale_.shape}')
            print(f'Has transform: {hasattr(scaler, "transform")}')
    else:
        print(f'\nNo std file found. Checking if array might be scale values...')
        # Maybe it's scale, not mean?
        print('Trying to create scaler assuming this is mean, and scale=1...')
        scaler = StandardScaler()
        scaler.mean_ = obj
        scaler.scale_ = np.ones_like(obj)
        scaler.var_ = np.ones_like(obj)
        print(f'Created StandardScaler (assuming unit scale)')





