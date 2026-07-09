import tensorflow as tf
from tensorflow import keras
from keras.mixed_precision import Policy
from keras.engine.input_layer import InputLayer

class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

# Simple DTypePolicy deserializer function
def deserialize_dtype_policy(config=None, **kwargs):
    # Handle both config dict and keyword args
    if config is None:
        config = kwargs
    if isinstance(config, dict):
        dtype = config.get('dtype', config.get('name', 'float32'))
        name = config.get('name', dtype)
    else:
        name = 'float32'
    return Policy(name)

try:
    from keras.utils import custom_object_scope
    with custom_object_scope({'DTypePolicy': deserialize_dtype_policy, 'InputLayer': CompatibleInputLayer}):
        model = keras.models.load_model('models/lstm_baseline_dst.keras', compile=False)
    print('SUCCESS! Model loaded')
    print(f'Input: {model.input_shape}, Output: {model.output_shape}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

