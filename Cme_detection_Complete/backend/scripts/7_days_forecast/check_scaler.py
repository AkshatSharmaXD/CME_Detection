import pickle
import os

scaler_path = 'scaler.pkl'
if os.path.exists(scaler_path):
    # Try normal load first
    try:
        with open(scaler_path, 'rb') as f:
            obj = pickle.load(f)
        print(f'Normal load - Type: {type(obj)}')
        print(f'Has transform: {hasattr(obj, "transform")}')
        if hasattr(obj, 'shape'):
            print(f'Shape: {obj.shape}')
        print(f'Dir: {[x for x in dir(obj) if not x.startswith("_")][:10]}')
    except Exception as e:
        print(f'Normal load failed: {e}')
    
    # Try with compatibility unpickler
    class NumpyCompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            elif module == 'numpy._core':
                module = 'numpy.core'
            return super().find_class(module, name)
    
    try:
        with open(scaler_path, 'rb') as f:
            unpickler = NumpyCompatibleUnpickler(f)
            obj = unpickler.load()
        print(f'\nCompatible load - Type: {type(obj)}')
        print(f'Has transform: {hasattr(obj, "transform")}')
        if hasattr(obj, 'shape'):
            print(f'Shape: {obj.shape}')
    except Exception as e:
        print(f'Compatible load failed: {e}')





