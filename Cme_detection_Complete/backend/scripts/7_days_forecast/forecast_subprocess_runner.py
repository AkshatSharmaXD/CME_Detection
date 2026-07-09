"""
Subprocess runner for forecast predictions
This script runs in a separate process to isolate TensorFlow import issues
"""
import sys
import os
import json
import pickle
import traceback

# Add parent directories to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def main():
    """Main entry point for subprocess"""
    try:
        # Import here to catch any import errors
        # This subprocess isolates TensorFlow import issues
        from forecast_model_runner import ForecastModelRunner
        
        # Create runner instance
        runner = ForecastModelRunner()
        
        # Make predictions
        predictions_df = runner.make_predictions()
        
        # Convert DataFrame to JSON-serializable format
        result = {
            'success': True,
            'data': {
                'index': predictions_df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'columns': predictions_df.columns.tolist(),
                'values': predictions_df.values.tolist()
            }
        }
        
        # Output result as JSON (to stdout, errors go to stderr)
        print(json.dumps(result))
        return 0
        
    except ImportError as e:
        # Special handling for TensorFlow import errors
        error_msg = str(e)
        if "TensorFlow" in error_msg or "tensorflow" in error_msg.lower() or "Unable to convert function return value" in error_msg:
            error_result = {
                'success': False,
                'error': f"TensorFlow import failed in subprocess. This is a known issue with Python 3.11 on Windows.\n\n"
                        f"Error: {error_msg}\n\n"
                        f"SOLUTIONS:\n"
                        f"1. Install TensorFlow 2.13.0: python -m pip install tensorflow==2.13.0 --user\n"
                        f"2. Use Python 3.10 instead of 3.11\n"
                        f"3. Install Microsoft Visual C++ Redistributable\n"
                        f"4. Try: python -m pip install tensorflow-cpu==2.13.0 --user",
                'error_type': 'TensorFlowImportError',
                'traceback': traceback.format_exc()
            }
        else:
            error_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        # Output JSON to stdout (errors are in the JSON)
        print(json.dumps(error_result))
        return 1
    except Exception as e:
        # Return error as JSON
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_result))
        return 1

if __name__ == "__main__":
    sys.exit(main())

