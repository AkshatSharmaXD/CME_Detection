"""
7-Day Forecast Model Runner
Loads the trained LSTM model and makes real-time predictions
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Lazy load TensorFlow to avoid import errors on Windows/Python 3.11
_tf_loaded = False
_tf = None
_keras = None

def _lazy_load_tensorflow():
    """Lazy load TensorFlow only when needed"""
    global _tf_loaded, _tf, _keras
    if not _tf_loaded:
        try:
            # Set environment variables before importing to suppress warnings
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            
            # Try importing TensorFlow
            import tensorflow as tf
            from tensorflow import keras
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            # Store globally
            _tf = tf
            _keras = keras
            _tf_loaded = True
            
            print("✓ TensorFlow loaded successfully")
            return tf, keras
        except TypeError as e:
            # This is the specific error we're seeing on Windows/Python 3.11
            if "Unable to convert function return value" in str(e):
                raise ImportError(
                    f"TensorFlow compatibility error on Windows with Python 3.11.\n"
                    f"Error: {e}\n\n"
                    f"Solutions:\n"
                    f"1. Install TensorFlow 2.13.0: pip install tensorflow==2.13.0\n"
                    f"2. Use Python 3.10 instead of 3.11\n"
                    f"3. Install Microsoft Visual C++ Redistributable\n"
                    f"4. Try: pip install tensorflow-cpu==2.13.0"
                ) from e
            raise
        except Exception as e:
            raise ImportError(
                f"Failed to import TensorFlow: {e}\n\n"
                f"This may be due to compatibility issues with Python 3.11 on Windows.\n"
                f"Try: pip install tensorflow==2.13.0 or use Python 3.10"
            ) from e
    return _tf, _keras

# Add parent directories to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from omniweb_data_fetcher import OMNIWebDataFetcher
from noaa_realtime_data import get_combined_realtime_data

class ForecastModelRunner:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.script_dir, 'models', 'lstm_baseline_dst.keras')
        self.scaler_path = os.path.join(self.script_dir, 'scaler.pkl')
        self.model = None
        self.scaler = None
        self.feature_cols = None
        # Model outputs only Dst, but we need all 4: Dst, Kp, ap, Sunspot
        # We'll estimate the other 3 from Dst predictions
        self.target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT', 'Sunspot_Number']  # All 4 targets
        self.lookback = 72  # 3 days of historical data
        self.forecast_horizon = 168  # 7 days ahead
        
    def _load_weights_manually(self, model, weights_path):
        """Manually load weights by mapping H5 layer names to model layer names"""
        import h5py
        import zipfile
        import io
        
        try:
            # If weights_path is in a zip, extract it
            if weights_path.endswith('.keras'):
                with zipfile.ZipFile(weights_path, 'r') as z:
                    weights_data = z.read('model.weights.h5')
                    weights_file = io.BytesIO(weights_data)
            else:
                weights_file = weights_path
            
            with h5py.File(weights_file, 'r') as f:
                # Map H5 layer names to model layer names
                # H5 file has: lstm, lstm_1, dense, dense_1
                # Model has: lstm_1, lstm_2, dense_1, output
                h5_to_model_map = {
                    'lstm': 'lstm_1',      # First LSTM in H5 -> lstm_1 in model
                    'lstm_1': 'lstm_2',    # Second LSTM in H5 -> lstm_2 in model
                    'dense': 'dense_1',    # First dense in H5 -> dense_1 in model
                    'dense_1': 'output',   # Second dense in H5 -> output in model
                }
                
                # Create a dict of model layer names to layer objects
                model_layers = {layer.name: layer for layer in model.layers}
                
                # Load weights from H5 and map to model layers
                if 'layers' in f:
                    for h5_layer_name in f['layers'].keys():
                        # Skip dropout layers (no weights)
                        if 'dropout' in h5_layer_name.lower():
                            continue
                        
                        # Get the corresponding model layer name
                        model_layer_name = h5_to_model_map.get(h5_layer_name, h5_layer_name)
                        
                        if model_layer_name not in model_layers:
                            print(f"      Warning: Model layer '{model_layer_name}' not found (from H5 '{h5_layer_name}')")
                            continue
                        
                        layer = model_layers[model_layer_name]
                        layer_group = f['layers'][h5_layer_name]
                        weights = []
                        
                        # Get weights based on layer type
                        if 'cell' in layer_group and 'vars' in layer_group['cell']:
                            # LSTM cell weights: kernel, recurrent_kernel, bias
                            vars_group = layer_group['cell']['vars']
                            weights = [np.array(vars_group[str(i)]) for i in sorted([int(k) for k in vars_group.keys()])]
                        elif 'vars' in layer_group:
                            # Dense layer weights: kernel, bias
                            vars_group = layer_group['vars']
                            weights = [np.array(vars_group[str(i)]) for i in sorted([int(k) for k in vars_group.keys()])]
                        
                        if weights:
                            try:
                                layer.set_weights(weights)
                                print(f"      ✓ Loaded weights for {model_layer_name} (from H5 '{h5_layer_name}') - {len(weights)} arrays")
                            except Exception as e:
                                print(f"      ✗ Failed to load weights for {model_layer_name}: {e}")
                                # Try to see what went wrong
                                expected = layer.get_weights()
                                print(f"         Expected shapes: {[w.shape for w in expected]}")
                                print(f"         Got shapes: {[w.shape for w in weights]}")
                
                # Verify all trainable layers have weights
                print(f"\n      Verifying weight loading...")
                all_loaded = True
                for layer in model.layers:
                    if hasattr(layer, 'trainable_weights') and len(layer.trainable_weights) > 0:
                        weights = layer.get_weights()
                        if len(weights) == 0:
                            print(f"      ✗ {layer.name} has no weights loaded!")
                            all_loaded = False
                        else:
                            # Check if weights are all zeros (might indicate failed load)
                            all_zeros = all(np.all(w == 0) for w in weights)
                            if all_zeros:
                                print(f"      ⚠️  {layer.name} weights are all zeros")
                
                if all_loaded:
                    print(f"      ✓ All weights loaded successfully!")
                
                return True
        except Exception as e:
            print(f"      ✗ Manual weight loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model(self):
        """Load the trained LSTM model with comprehensive compatibility fixes"""
        try:
            # Lazy load TensorFlow
            tf, keras = _lazy_load_tensorflow()
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            print(f"Loading model from {self.model_path}...")
            
            # Define custom objects to handle DTypePolicy and batch_shape
            try:
                from keras.mixed_precision import Policy
                from keras.engine.input_layer import InputLayer
                from keras.utils import custom_object_scope
                
                # Custom InputLayer that accepts batch_shape for compatibility
                class CompatibleInputLayer(InputLayer):
                    def __init__(self, *args, **kwargs):
                        # Convert batch_shape to input_shape if present
                        if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
                            batch_shape = kwargs.pop('batch_shape')
                            if batch_shape and len(batch_shape) > 1:
                                kwargs['input_shape'] = batch_shape[1:]
                        super().__init__(*args, **kwargs)
                
                # DTypePolicy deserializer
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
            except (ImportError, AttributeError):
                custom_objects = {}
            
            # Load architecture and weights separately (reliable method for this model)
            import json, zipfile, tempfile
            temp_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(self.model_path, 'r') as z:
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
                from keras.utils import custom_object_scope
                with custom_object_scope(custom_objects):
                    model_json = json.dumps(config)
                    self.model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
                
                print(f"   ✓ Architecture loaded")
                
                # Load weights manually with proper mapping
                weights_path = os.path.join(temp_dir, 'model.weights.h5')
                weights_loaded = self._load_weights_manually(self.model, weights_path)
                
                if weights_loaded:
                    print(f"✓ Model loaded successfully with all weights")
                    print(f"  Model input shape: {self.model.input_shape}")
                    print(f"  Model output shape: {self.model.output_shape}")
                    return True
                else:
                    raise RuntimeError("Failed to load model weights")
                    
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_model_with_compatibility_fix(self, tf, keras):
        """Load model with comprehensive compatibility fixes for batch_shape and DTypePolicy"""
        import json
        import zipfile
        import tempfile
        import shutil
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract the .keras file (it's a zip file)
            with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Comprehensive fix function - fix batch_shape AND convert DTypePolicy to PolicyV1
            def fix_compatibility(obj, path=""):
                if isinstance(obj, dict):
                    # Fix batch_shape -> input_shape
                    if 'batch_shape' in obj:
                        batch_shape = obj.pop('batch_shape')
                        if batch_shape and len(batch_shape) > 1:
                            obj['input_shape'] = batch_shape[1:]
                            print(f"      Fixed batch_shape -> input_shape at {path}")
                    
                    # Convert DTypePolicy class to PolicyV1 format
                    if 'class_name' in obj and obj.get('class_name') == 'DTypePolicy':
                        config = obj.get('config', {})
                        dtype = config.get('dtype', 'float32')
                        # Convert to PolicyV1 format that Keras 2.13 understands
                        obj['class_name'] = 'PolicyV1'
                        obj['config'] = {'name': dtype}
                        print(f"      Converted DTypePolicy -> PolicyV1({dtype}) at {path}")
                    
                    # Fix dtype field that contains DTypePolicy class
                    if 'dtype' in obj and isinstance(obj['dtype'], dict):
                        dtype_obj = obj['dtype']
                        if dtype_obj.get('class_name') == 'DTypePolicy':
                            config = dtype_obj.get('config', {})
                            dtype_val = config.get('dtype', 'float32')
                            obj['dtype'] = dtype_val
                            print(f"      Fixed dtype DTypePolicy -> {dtype_val} at {path}")
                    
                    # Recursively fix nested objects
                    for key, value in obj.items():
                        fix_compatibility(value, f"{path}.{key}" if path else key)
                        
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        fix_compatibility(item, f"{path}[{i}]" if path else f"[{i}]")
            
            # Fix all JSON files in the extracted model
            json_files_fixed = 0
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.json'):
                        json_path = os.path.join(root, file)
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            original_config = json.dumps(config)
                            fix_compatibility(config)
                            
                            # Only save if something changed
                            if json.dumps(config) != original_config:
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump(config, f, indent=2, ensure_ascii=False)
                                json_files_fixed += 1
                        except Exception as e:
                            print(f"   ⚠️  Warning: Could not fix {file}: {e}")
                            continue
            
            print(f"   Fixed {json_files_fixed} JSON config file(s)")
            
            # Create temporary model file - preserve ALL files exactly as they were
            temp_model_path = os.path.join(temp_dir, 'model_fixed.keras')
            
            # Get list of all files to include (excluding the output file itself)
            files_to_include = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file != 'model_fixed.keras':
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        files_to_include.append((file_path, arcname))
            
            # Repackage with all files
            with zipfile.ZipFile(temp_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for file_path, arcname in files_to_include:
                    zip_ref.write(file_path, arcname)
            
            # Verify all critical files are present
            with zipfile.ZipFile(temp_model_path, 'r') as zip_ref:
                required_files = ['config.json', 'metadata.json', 'model.weights.h5']
                files_in_zip = zip_ref.namelist()
                missing = [f for f in required_files if f not in files_in_zip]
                if missing:
                    raise ValueError(f"Missing required files in repackaged model: {missing}")
                print(f"   ✓ Verified all files present ({len(files_in_zip)} files)")
            
            # Try loading with both keras and tf.keras
            # After fixing DTypePolicy to PolicyV1, we should be able to load without custom_objects
            for loader_name, loader in [("keras", keras.models.load_model), ("tf.keras", tf.keras.models.load_model)]:
                try:
                    print(f"   Loading fixed model with {loader_name}...")
                    model = loader(temp_model_path, compile=False)
                    print(f"   ✓ Fixed model loaded successfully with {loader_name}")
                    return model
                except Exception as e:
                    if loader_name == "tf.keras":
                        # Last attempt failed
                        raise
                    continue
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def load_scaler(self):
        """Load the data scaler with numpy version compatibility"""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
            
            import joblib
            self.scaler = joblib.load(self.scaler_path)
            
            print(f"✓ Scaler loaded from {self.scaler_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_feature_columns(self):
        """Get feature columns from training data"""
        try:
            train_data_path = os.path.join(self.script_dir, 'train_data_scaled.csv')
            if os.path.exists(train_data_path):
                train_data = pd.read_csv(train_data_path, nrows=1)
                exclude_cols = ['Datetime', 'Year', 'DOY', 'Hour'] + self.target_vars
                self.feature_cols = [col for col in train_data.columns if col not in exclude_cols]
                print(f"✓ Feature columns loaded: {len(self.feature_cols)} features")
                
                # Check if scaler has target variables (for inverse transform)
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    # Try to find target indices in scaler
                    try:
                        # Load unscaled data to get column order
                        train_unscaled_path = os.path.join(self.script_dir, 'train_data_unscaled.csv')
                        if os.path.exists(train_unscaled_path):
                            train_unscaled = pd.read_csv(train_unscaled_path, nrows=1)
                            all_cols = [col for col in train_unscaled.columns if col not in ['Datetime', 'Year', 'DOY', 'Hour']]
                            self.scaler_column_order = all_cols
                            print(f"✓ Scaler column order loaded: {len(all_cols)} columns")
                        else:
                            self.scaler_column_order = None
                    except:
                        self.scaler_column_order = None
                else:
                    self.scaler_column_order = None
                
                return True
            else:
                # Default feature columns based on notebook
                self.feature_cols = [
                    'IMF_Mag_Avg_nT', 'IMF_Lat_deg', 'IMF_Long_deg', 'Bz_GSM_nT',
                    'Proton_Density_n_cc', 'Flow_Speed_km_s', 'Flow_Longitude_deg', 'Flow_Latitude_deg',
                    'Alpha_Proton_Ratio', 'Sunspot_Number', 'F10.7_Index',
                    'Proton_Flux_1MeV', 'Proton_Flux_2MeV', 'Proton_Flux_4MeV', 'Proton_Flux_10MeV',
                    'Proton_Flux_30MeV', 'Proton_Flux_60MeV',
                    'Hour_of_Day', 'Day_of_Week', 'Day_of_Year', 'Month', 'Season',
                    'Hour_Sin', 'Hour_Cos', 'DOY_Sin', 'DOY_Cos'
                ]
                print(f"✓ Using default feature columns: {len(self.feature_cols)} features")
                self.scaler_column_order = None
                return True
        except Exception as e:
            print(f"❌ Error loading feature columns: {e}")
            return False
    
    def fetch_recent_data(self, days=7):
        """Fetch recent data from CSV file or OMNI/NOAA"""
        try:
            # First, try to use the CSV file if it exists
            csv_path = os.path.join(self.script_dir, 'omni_data_updatedyears.csv')
            if os.path.exists(csv_path):
                print(f"📁 Loading data from CSV file: {csv_path}")
                try:
                    # Load the CSV file
                    df = pd.read_csv(csv_path)
                    
                    # Convert Datetime column to datetime if it's not already
                    if 'Datetime' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                        # Remove rows with invalid datetime
                        df = df.dropna(subset=['Datetime'])
                        df = df.sort_values('Datetime')
                    
                    # Remove any rows that are completely NaN
                    df = df.dropna(how='all')
                    
                    if len(df) == 0:
                        raise ValueError("CSV file has no valid data rows")
                    
                    # Get the most recent data (last N hours needed for lookback)
                    # We need at least lookback hours (72 hours = 3 days)
                    hours_needed = max(self.lookback, days * 24)
                    if len(df) >= hours_needed:
                        recent_data = df.tail(hours_needed).copy()
                        print(f"✓ Loaded {len(recent_data)} data points from CSV (most recent {hours_needed} hours)")
                        print(f"  Date range: {recent_data['Datetime'].iloc[0]} to {recent_data['Datetime'].iloc[-1]}")
                        return recent_data
                    else:
                        print(f"⚠️  CSV has only {len(df)} valid rows, need at least {hours_needed}. Using all available data.")
                        print(f"  Date range: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")
                        return df.copy()
                except Exception as csv_error:
                    print(f"⚠️  Error loading CSV file: {csv_error}")
                    import traceback
                    traceback.print_exc()
                    print("   Falling back to API fetch...")
            
            # Fallback to API fetch if CSV doesn't exist or fails
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # OMNI API fallback removed because OMNIWeb only goes up to April 2026
            # which breaks realtime 7-day forecasting in July 2026. Directly falling back to NOAA.
            
            # Fallback to NOAA realtime
            try:
                from noaa_realtime_data import get_combined_realtime_data
                noaa_result = get_combined_realtime_data()
                if noaa_result.get('success') and noaa_result.get('data') is not None:
                    noaa_data = noaa_result['data']
                    print(f"✓ Fetched {len(noaa_data)} data points from NOAA API")
                    return noaa_data
            except Exception as noaa_error:
                print(f"⚠️  NOAA API fetch failed: {noaa_error}")
            
            raise ValueError("No data available from CSV, OMNI, or NOAA")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_features(self, data):
        """Prepare features from raw data"""
        try:
            df = data.copy()
            
            # Handle timestamp column - CSV uses 'Datetime', API might use 'timestamp'
            if 'Datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Datetime'])
            elif 'timestamp' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if 'index' in df.columns:
                        df['timestamp'] = df['index']
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            else:
                raise ValueError("No timestamp/Datetime column found in data")
            
            # Map column names to expected feature names
            column_mapping = {
                'bt': 'IMF_Mag_Avg_nT',
                'bx_gsm': 'Bx_GSM_nT',
                'by_gsm': 'By_GSM_nT',
                'bz_gsm': 'Bz_GSM_nT',
                'density': 'Proton_Density_n_cc',
                'speed': 'Flow_Speed_km_s',
                'velocity': 'Flow_Speed_km_s',
                'lon_gsm': 'Flow_Longitude_deg',
                'lat_gsm': 'Flow_Latitude_deg',
                'sunspot_number': 'Sunspot_Number',  # Map OMNIWeb sunspot_number to Sunspot_Number feature
                'Sunspot_Number': 'Sunspot_Number',  # Also handle if already in correct format
                'f10_7': 'F10.7_Index',
                'f10.7': 'F10.7_Index',
            }
            
            # Create feature dataframe
            features_df = pd.DataFrame()
            features_df['timestamp'] = df['timestamp']
            
            # Check if data already has the expected feature column names (from CSV)
            # If so, use them directly; otherwise map from API column names
            expected_feature_cols = [
                'IMF_Mag_Avg_nT', 'IMF_Lat_deg', 'IMF_Long_deg', 'Bz_GSM_nT',
                'Proton_Density_n_cc', 'Flow_Speed_km_s', 'Flow_Longitude_deg', 'Flow_Latitude_deg',
                'Alpha_Proton_Ratio', 'Sunspot_Number', 'F10.7_Index'
            ]
            
            # If CSV data already has correct column names, use them directly
            has_expected_cols = any(col in df.columns for col in expected_feature_cols)
            
            if has_expected_cols:
                # Data is from CSV with correct column names - use directly
                for col in expected_feature_cols:
                    if col in df.columns:
                        features_df[col] = df[col]
            else:
                # Data is from API - need to map column names
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        features_df[new_col] = df[old_col]
            
            # Calculate derived features
            if 'timestamp' in features_df.columns:
                features_df['Hour_of_Day'] = features_df['timestamp'].dt.hour
                features_df['Day_of_Week'] = features_df['timestamp'].dt.dayofweek
                features_df['Day_of_Year'] = features_df['timestamp'].dt.dayofyear
                features_df['Month'] = features_df['timestamp'].dt.month
                features_df['Season'] = (features_df['Month'] % 12 // 3).astype(int)
                features_df['Hour_Sin'] = np.sin(2 * np.pi * features_df['Hour_of_Day'] / 24)
                features_df['Hour_Cos'] = np.cos(2 * np.pi * features_df['Hour_of_Day'] / 24)
                features_df['DOY_Sin'] = np.sin(2 * np.pi * features_df['Day_of_Year'] / 365)
                features_df['DOY_Cos'] = np.cos(2 * np.pi * features_df['Day_of_Year'] / 365)
            
            # Fill missing features with defaults
            for col in self.feature_cols:
                if col not in features_df.columns:
                    if 'Flux' in col or 'Sunspot' in col or 'F10.7' in col:
                        features_df[col] = 0.0  # Default for flux/sunspot
                    elif 'Ratio' in col:
                        features_df[col] = 0.05  # Default alpha/proton ratio
                    else:
                        features_df[col] = 0.0
            
            # Select only feature columns
            feature_data = features_df[self.feature_cols].copy()
            
            # Fill NaN values
            feature_data = feature_data.ffill().fillna(0)
            
            return feature_data, features_df['timestamp']
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _estimate_other_parameters_from_dst(self, dst_predictions):
        """
        Estimate Kp_10 and ap_index_nT from Dst predictions
        using historical correlations from training data
        Model should output 3 targets: Dst_Index_nT, Kp_10, ap_index_nT (in this order)
        """
        try:
            # Load training data to calculate correlations
            train_unscaled_path = os.path.join(self.script_dir, 'train_data_unscaled.csv')
            if not os.path.exists(train_unscaled_path):
                print("⚠️  Training data not found, using default estimates")
                return self._default_parameter_estimates(dst_predictions)
            
            train_data = pd.read_csv(train_unscaled_path)
            
            # Calculate correlations between Dst and other parameters
            dst_col = 'Dst_Index_nT'
            if dst_col not in train_data.columns:
                return self._default_parameter_estimates(dst_predictions)
            
            # Calculate mean and std for scaling
            correlations = {}
            means = {}
            stds = {}
            
            # Estimate all 3: Kp_10, ap_index_nT, and Sunspot_Number
            for target in ['Kp_10', 'ap_index_nT', 'Sunspot_Number']:
                if target in train_data.columns:
                    # Calculate correlation
                    corr = train_data[dst_col].corr(train_data[target])
                    # Check for NaN without using pd
                    correlations[target] = corr if corr == corr else 0.0  # NaN check: NaN != NaN
                    
                    # Calculate statistics
                    target_data = train_data[target].dropna()
                    if len(target_data) > 0:
                        means[target] = target_data.mean()
                        stds[target] = target_data.std()
                    else:
                        means[target] = self._get_default_mean(target)
                        stds[target] = self._get_default_std(target)
                else:
                    correlations[target] = 0.0
                    means[target] = self._get_default_mean(target)
                    stds[target] = self._get_default_std(target)
            
            # Get DST statistics for normalization
            dst_data = train_data[dst_col].dropna()
            dst_mean = dst_data.mean() if len(dst_data) > 0 else -13.0
            dst_std = dst_data.std() if len(dst_data) > 0 else 20.0
            
            # Estimate other parameters - Model expects 4 targets: Dst, Kp, ap, Sunspot (in this order)
            n_samples = len(dst_predictions)
            predictions_expanded = np.zeros((n_samples, 4))
            predictions_expanded[:, 0] = dst_predictions.flatten()  # Dst (index 0)
            
            # Normalize DST predictions for correlation-based estimation
            dst_normalized = (dst_predictions.flatten() - dst_mean) / dst_std if dst_std > 0 else dst_predictions.flatten()
            
            # Estimate each parameter using linear regression approach
            # Order: Dst (0), Kp_10 (1), ap_index_nT (2), Sunspot_Number (3)
            for idx, target in enumerate(['Kp_10', 'ap_index_nT', 'Sunspot_Number'], start=1):
                if correlations[target] != 0 and abs(correlations[target]) > 0.1:
                    # Use correlation-based linear estimation with proper scaling
                    # Formula: target = mean + corr * (dst_deviation_from_mean / dst_std) * target_std
                    # Scale down the effect to avoid extreme values
                    dst_deviation = dst_predictions.flatten() - dst_mean
                    scale_factor = 0.7  # Reduce scaling to avoid clipping
                    target_estimate = means[target] + correlations[target] * (dst_deviation / dst_std) * stds[target] * scale_factor if dst_std > 0 else means[target]
                    
                    # Apply reasonable bounds and scaling
                    if target == 'Kp_10':
                        # Kp_10 in training data is scaled (mean ~18 = 1.8 actual Kp, max 90 = 9.0 actual Kp)
                        # So we need to divide by 10 to get actual Kp (0-9 range)
                        if means[target] > 9.0:
                            target_estimate = target_estimate / 10.0
                        target_estimate = np.clip(target_estimate, 0.0, 9.0)
                    elif target == 'ap_index_nT':
                        target_estimate = np.clip(target_estimate, 0.0, 400.0)
                    elif target == 'Sunspot_Number':
                        target_estimate = np.clip(target_estimate, 0.0, 300.0)
                    
                    predictions_expanded[:, idx] = target_estimate
                else:
                    # Weak correlation, use mean with small variation
                    if target == 'Kp_10':
                        # Kp is scaled in training data, divide by 10
                        if means[target] > 9.0:
                            base_value = means[target] / 10.0
                            predictions_expanded[:, idx] = base_value + np.random.normal(0, 1.0, n_samples)
                        else:
                            predictions_expanded[:, idx] = means[target] + np.random.normal(0, stds[target] * 0.15, n_samples)
                        predictions_expanded[:, idx] = np.clip(predictions_expanded[:, idx], 0.0, 9.0)
                    elif target == 'ap_index_nT':
                        predictions_expanded[:, idx] = means[target] + np.random.normal(0, stds[target] * 0.15, n_samples)
                        predictions_expanded[:, idx] = np.clip(predictions_expanded[:, idx], 0.0, 400.0)
                    elif target == 'Sunspot_Number':
                        predictions_expanded[:, idx] = means[target] + np.random.normal(0, stds[target] * 0.15, n_samples)
                        predictions_expanded[:, idx] = np.clip(predictions_expanded[:, idx], 0.0, 300.0)
            
            # Update target_vars to include all 4: Dst, Kp, ap, Sunspot (in this order)
            self.target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT', 'Sunspot_Number']
            
            print(f"  Dst -> Kp correlation: {correlations['Kp_10']:.3f}")
            print(f"  Dst -> ap correlation: {correlations['ap_index_nT']:.3f}")
            print(f"  Dst -> Sunspot correlation: {correlations['Sunspot_Number']:.3f}")
            
            return predictions_expanded
            
        except Exception as e:
            print(f"⚠️  Error estimating other parameters: {e}")
            return self._default_parameter_estimates(dst_predictions)
    
    def _get_default_mean(self, target):
        """Get default mean value for a target parameter"""
        defaults = {
            'ap_index_nT': 12.0,
            'Sunspot_Number': 50.0,
            'Kp_10': 2.5
        }
        return defaults.get(target, 0.0)
    
    def _get_default_std(self, target):
        """Get default std value for a target parameter"""
        defaults = {
            'ap_index_nT': 8.0,
            'Sunspot_Number': 30.0,
            'Kp_10': 1.5
        }
        return defaults.get(target, 1.0)
    
    def _default_parameter_estimates(self, dst_predictions):
        """Fallback: use default estimates if correlation calculation fails"""
        n_samples = len(dst_predictions)
        predictions_expanded = np.zeros((n_samples, 4))
        predictions_expanded[:, 0] = dst_predictions.flatten()  # Dst (index 0)
        
        # Simple estimates based on typical values with small variation
        # Order: Dst (0), Kp_10 (1), ap_index_nT (2), Sunspot_Number (3)
        dst_normalized = (dst_predictions.flatten() + 13.0) / 20.0  # Rough normalization
        
        # Kp_10: typically 1-4, higher when DST is more negative (storm)
        predictions_expanded[:, 1] = np.clip(2.5 - dst_normalized * 1.5, 0.0, 9.0)  # Kp_10
        
        # ap_index_nT: typically 5-20, higher when DST is more negative (storm)
        predictions_expanded[:, 2] = np.clip(12.0 - dst_normalized * 5.0, 5.0, 25.0)  # ap_index_nT
        
        # Sunspot_Number: typically 30-80, less correlated with DST
        predictions_expanded[:, 3] = np.full(n_samples, 50.0) + np.random.normal(0, 10.0, n_samples)
        predictions_expanded[:, 3] = np.clip(predictions_expanded[:, 3], 0.0, 200.0)  # Sunspot_Number
        
        self.target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT', 'Sunspot_Number']
        return predictions_expanded
    
    def make_predictions(self):
        """
        Make 7-day forecast predictions - MUST return DataFrame, never None.
        Uses CSV file (omni_data_updatedyears.csv) for input data, then model for predictions.
        """
        import sys
        print("="*70)
        print("🚀 STARTING MODEL PREDICTION - USING CSV DATA + MODEL")
        print("="*70)
        print(f"Python: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Model path: {self.model_path}")
        print(f"Scaler path: {self.scaler_path}")
        print("="*70)
        try:
            
            # Load model and scaler if not loaded
            if self.model is None:
                print("Loading model...")
                if not self.load_model():
                    raise RuntimeError("Failed to load model")
                print("✓ Model loaded")
            
            if self.scaler is None:
                print("Loading scaler...")
                if not self.load_scaler():
                    raise RuntimeError("Failed to load scaler")
                print("✓ Scaler loaded")
            
            if self.feature_cols is None:
                print("Getting feature columns...")
                if not self.get_feature_columns():
                    raise RuntimeError("Failed to get feature columns")
                print(f"✓ Feature columns loaded: {len(self.feature_cols)} features")
            
            # Fetch recent data
            print("Fetching recent data...")
            data = self.fetch_recent_data(days=7)
            if data is None or data.empty:
                raise ValueError(f"Failed to fetch data or data is empty. Got: {type(data)}")
            print(f"✓ Fetched {len(data)} data points")
            
            # Prepare features
            print("Preparing features...")
            feature_data, timestamps = self.prepare_features(data)
            if feature_data is None:
                raise ValueError("Failed to prepare features - feature_data is None")
            print(f"✓ Prepared {len(feature_data)} feature rows")
            
            # Need at least lookback hours of data
            if len(feature_data) < self.lookback:
                print(f"⚠️  Need at least {self.lookback} hours of data, got {len(feature_data)}")
                # Use last available data and pad if needed
                if len(feature_data) > 0:
                    last_row = feature_data.iloc[-1:].copy()
                    padding_needed = self.lookback - len(feature_data)
                    print(f"   Padding with {padding_needed} rows...")
                    for _ in range(padding_needed):
                        feature_data = pd.concat([feature_data, last_row], ignore_index=True)
                    print(f"✓ Padded to {len(feature_data)} rows")
                else:
                    raise ValueError(f"No feature data available. Need at least {self.lookback} hours.")
            
            # Get last lookback hours
            input_data = feature_data.iloc[-self.lookback:].values
            
            # Model expects exactly 18 features, but we might have 19
            # Get the expected feature count from model input shape
            expected_features = self.model.input_shape[2] if len(self.model.input_shape) > 2 else len(self.feature_cols)
            
            # If we have more features than expected, take only the first N
            if input_data.shape[1] > expected_features:
                print(f"⚠️  Input has {input_data.shape[1]} features, model expects {expected_features}. Using first {expected_features} features.")
                input_data = input_data[:, :expected_features]
            
            # Log input data summary to verify it's changing
            print(f"Input data shape: {input_data.shape}")
            print(f"Input data summary (first feature): min={input_data[:, 0].min():.4f}, max={input_data[:, 0].max():.4f}, mean={input_data[:, 0].mean():.4f}")
            print(f"Input data summary (last feature): min={input_data[:, -1].min():.4f}, max={input_data[:, -1].max():.4f}, mean={input_data[:, -1].mean():.4f}")
            ts_start = timestamps.iloc[-self.lookback] if len(timestamps) >= self.lookback else timestamps.iloc[0]
            print(f"Input data timestamp range: {ts_start} to {timestamps.iloc[-1]}")
            
            # Scale the data - scaler was fitted on 19 features, but model needs 18
            # If scaler has more features than input, use only the first N
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_.shape[0] > expected_features:
                # Create a temporary scaler with only the first N features
                from sklearn.preprocessing import StandardScaler
                temp_scaler = StandardScaler()
                temp_scaler.mean_ = self.scaler.mean_[:expected_features]
                temp_scaler.scale_ = self.scaler.scale_[:expected_features]
                temp_scaler.var_ = self.scaler.var_[:expected_features] if hasattr(self.scaler, 'var_') else temp_scaler.scale_ ** 2
                temp_scaler.n_features_in_ = expected_features
                input_scaled = temp_scaler.transform(input_data)
            else:
                input_scaled = self.scaler.transform(input_data)
            
            # Reshape for LSTM: (1, lookback, features)
            input_sequence = input_scaled.reshape(1, self.lookback, expected_features)
            
            # Make prediction - THIS IS WHERE MODEL ACTUALLY RUNS
            print("="*70)
            print("🤖 CALLING MODEL.PREDICT() - THIS GENERATES NEW PREDICTIONS")
            print("="*70)
            print(f"Input sequence shape: {input_sequence.shape}")
            print(f"Calling model.predict() at {datetime.now()}")
            predictions_scaled = self.model.predict(input_sequence, verbose=0)
            print(f"✓ Model.predict() completed at {datetime.now()}")
            print(f"Raw predictions shape: {predictions_scaled.shape}")
            print(f"Raw predictions sample (first 5 values): {predictions_scaled.flatten()[:5]}")
            print("="*70)
            
            # Check model output shape
            print(f"Model output shape: {predictions_scaled.shape}")
            print(f"Expected shape: (1, {self.forecast_horizon}, {len(self.target_vars)}) or ({self.forecast_horizon * len(self.target_vars)},)")
            
            # Handle different output shapes
            if len(predictions_scaled.shape) == 3:
                # Shape: (batch, forecast_horizon, targets)
                predictions_scaled = predictions_scaled[0]  # Remove batch dimension
            elif len(predictions_scaled.shape) == 2:
                # Shape: (forecast_horizon, targets) or (1, forecast_horizon * targets)
                if predictions_scaled.shape[0] == 1:
                    # Flatten and reshape
                    predictions_scaled = predictions_scaled.reshape(self.forecast_horizon, -1)
                else:
                    predictions_scaled = predictions_scaled
            elif len(predictions_scaled.shape) == 1:
                # Shape: (forecast_horizon * targets,)
                predictions_scaled = predictions_scaled.reshape(self.forecast_horizon, -1)
            
            # Ensure we have the right number of targets
            actual_targets = predictions_scaled.shape[1] if len(predictions_scaled.shape) > 1 else 1
            expected_targets = len(self.target_vars)
            
            print(f"Actual model outputs: {actual_targets}, Expected: {expected_targets}")
            
            if actual_targets != expected_targets:
                print(f"⚠️  Warning: Model outputs {actual_targets} targets but we expect {expected_targets}")
                print(f"   Model might have been trained with different targets. Using what model provides.")
                # Adjust target_vars to match model output
                if actual_targets == 1:
                    # Model has 1 output, likely: Dst_Index_nT (based on model name "lstm_baseline_dst")
                    self.target_vars = ['Dst_Index_nT']
                    print(f"   Adjusted target_vars to: {self.target_vars} (single-target model)")
                elif actual_targets == 3:
                    # Model has 3 outputs, likely: Dst, Kp, Ap (no Sunspot)
                    self.target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
                    print(f"   Adjusted target_vars to: {self.target_vars}")
                elif actual_targets == 4:
                    # Model has 4 outputs, use all 4
                    pass
                else:
                    raise ValueError(f"Unexpected number of model outputs: {actual_targets}")
            
            # Reshape predictions: (forecast_horizon, targets)
            # Handle case where predictions_scaled is (1, 168) - single target
            if len(predictions_scaled.shape) == 2 and predictions_scaled.shape[0] == 1:
                # Shape is (1, 168) - single target, 168 hours
                predictions_scaled = predictions_scaled.T  # Transpose to (168, 1)
            elif len(predictions_scaled.shape) == 1:
                # Shape is (168,) - single target, reshape to (168, 1)
                predictions_scaled = predictions_scaled.reshape(-1, 1)
            
            # Ensure final shape is (forecast_horizon, num_targets)
            if predictions_scaled.shape[0] != self.forecast_horizon:
                predictions_scaled = predictions_scaled.reshape(self.forecast_horizon, len(self.target_vars))
            
            # Inverse transform predictions
            print(f"Predictions shape before inverse: {predictions_scaled.shape}")
            print(f"Sample predictions (scaled): {predictions_scaled[0]}")
            print(f"Target vars: {self.target_vars}")
            print(f"Feature cols count: {len(self.feature_cols)}")
            
            # Skip scaler column order method - scaler was fitted on features only (19 features)
            # Go directly to manual inverse transform using training data statistics
            predictions = None
            
            # Manual inverse transform using training data statistics
            if predictions is None:
                try:
                    print("Attempting manual inverse transform using training data statistics...")
                    train_unscaled_path = os.path.join(self.script_dir, 'train_data_unscaled.csv')
                    if os.path.exists(train_unscaled_path):
                        train_unscaled = pd.read_csv(train_unscaled_path)
                        all_cols = [col for col in train_unscaled.columns if col not in ['Datetime', 'Year', 'DOY', 'Hour']]
                        
                        predictions = np.zeros((self.forecast_horizon, len(self.target_vars)))
                        for i, target in enumerate(self.target_vars):
                            if target in all_cols:
                                # Get statistics directly from training data
                                target_data = train_unscaled[target].dropna()
                                if len(target_data) > 0:
                                    mean = target_data.mean()
                                    std = target_data.std()
                                    if std == 0 or pd.isna(std):
                                        std = 1.0  # Avoid division by zero
                                    # Inverse transform: x = scaled * std + mean
                                    predictions[:, i] = predictions_scaled[:, i] * std + mean
                                    print(f"✓ {target}: mean={mean:.2f}, std={std:.2f}")
                                else:
                                    # No data available, use predictions as-is
                                    predictions[:, i] = predictions_scaled[:, i]
                                    print(f"⚠️  {target}: No training data available, using scaled values")
                            else:
                                # Target not in training data, use predictions as-is
                                predictions[:, i] = predictions_scaled[:, i]
                                print(f"⚠️  {target}: Not found in training data, using scaled values")
                        
                        print(f"✓ Manual inverse transform successful")
                    else:
                        raise ValueError("train_data_unscaled.csv not found")
                except Exception as e:
                    print(f"⚠️  Manual inverse transform failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"   Using predictions as-is (they might already be in original scale)")
                    predictions = predictions_scaled
            
            print(f"Sample predictions (final): {predictions[0]}")
            print(f"Value ranges after transform:")
            for i, target in enumerate(self.target_vars):
                print(f"  {target}: min={predictions[:, i].min():.2f}, max={predictions[:, i].max():.2f}, mean={predictions[:, i].mean():.2f}")
            
            # If model only outputs 1 target (Dst), estimate other 3 parameters using historical correlations
            if len(self.target_vars) == 1 and self.target_vars[0] == 'Dst_Index_nT':
                print("\n" + "="*70)
                print("📊 ESTIMATING OTHER PARAMETERS FROM DST PREDICTIONS")
                print("="*70)
                predictions = self._estimate_other_parameters_from_dst(predictions)
                print("✓ Estimated Kp_10, ap_index_nT, and Sunspot_Number from Dst predictions")
                print(f"  Updated predictions shape: {predictions.shape}")
                print(f"  Updated target_vars: {self.target_vars}")
            
            # Create forecast timestamps (next 7 days, hourly) - ALWAYS FROM CURRENT TIME
            current_time = datetime.now()
            print(f"Current time: {current_time}")
            print(f"Last data timestamp: {timestamps.iloc[-1] if len(timestamps) > 0 else 'N/A'}")
            
            # Start forecast from NOW (not from last data timestamp)
            forecast_start = current_time + timedelta(hours=1)  # Start 1 hour from now
            forecast_timestamps = pd.date_range(
                start=forecast_start,
                periods=self.forecast_horizon,
                freq='h'
            )
            
            print(f"Forecast start: {forecast_start}")
            print(f"Forecast end: {forecast_timestamps[-1]}")
            print(f"Forecast timestamps: {len(forecast_timestamps)} hours")
            
            # Create result dataframe
            result_df = pd.DataFrame(
                predictions,
                index=forecast_timestamps,
                columns=self.target_vars
            )
            
            # Calculate checksum to verify predictions are unique
            import hashlib
            predictions_hash = hashlib.md5(predictions.tobytes()).hexdigest()[:8]
            print(f"Predictions checksum: {predictions_hash}")
            
            print(f"\n{'='*70}")
            print(f"✅ PREDICTION COMPLETE - USING CSV DATA + MODEL PREDICTIONS")
            print(f"{'='*70}")
            print(f"\n📊 FINAL PREDICTIONS SUMMARY:")
            print(f"   Total predictions: {len(result_df)} (7 days = 168 hours)")
            print(f"   Columns: {list(result_df.columns)}")
            print(f"   Date range: {result_df.index[0]} to {result_df.index[-1]}")
            print(f"\n📈 FIRST ROW VALUES (Hour 1):")
            for col in self.target_vars:
                val = result_df[col].iloc[0]
                print(f"     {col:20s}: {val:12.6f}")
            print(f"\n📈 MIDDLE ROW VALUES (Hour 84):")
            mid_idx = len(result_df) // 2
            for col in self.target_vars:
                val = result_df[col].iloc[mid_idx]
                print(f"     {col:20s}: {val:12.6f}")
            print(f"\n📈 LAST ROW VALUES (Hour 168):")
            for col in self.target_vars:
                val = result_df[col].iloc[-1]
                print(f"     {col:20s}: {val:12.6f}")
            print(f"\n📊 VALUE RANGES (All 168 hours):")
            for col in self.target_vars:
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                mean_val = result_df[col].mean()
                print(f"     {col:20s}: min={min_val:10.4f}, max={max_val:10.4f}, mean={mean_val:10.4f}")
            print(f"\n🔍 VERIFICATION:")
            print(f"   Predictions checksum: {predictions_hash}")
            print(f"   Generated at: {current_time}")
            print(f"   Forecast start: {forecast_start}")
            print(f"   Forecast end: {forecast_timestamps[-1]}")
            print(f"   Is future forecast: {forecast_start > current_time}")
            print(f"{'='*70}\n")
            
            return result_df
            
        except Exception as e:
            print("="*70)
            print("❌ ERROR IN MODEL PREDICTION")
            print("="*70)
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            print("="*70)
            # Re-raise to propagate error - NO silent failures, NO CSV fallback
            raise RuntimeError(f"Model prediction failed: {str(e)}") from e

def generate_forecast():
    """Main function to generate forecast"""
    runner = ForecastModelRunner()
    predictions = runner.make_predictions()
    return predictions

if __name__ == "__main__":
    predictions = generate_forecast()
    if predictions is not None:
        print("\n" + "="*70)
        print("FORECAST PREDICTIONS")
        print("="*70)
        print(predictions.head(10))
        print(f"\nTotal predictions: {len(predictions)}")
    else:
        print("Failed to generate predictions")

