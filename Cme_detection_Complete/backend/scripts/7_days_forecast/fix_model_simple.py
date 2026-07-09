"""
Simple model fix - modify config in-place in zip
"""
import zipfile
import json
import os
import shutil

model_path = os.path.join(os.path.dirname(__file__), 'models', 'lstm_baseline_dst.keras')
backup_path = model_path + '.backup'

print("Creating backup...")
shutil.copy2(model_path, backup_path)

print("Fixing model...")
with zipfile.ZipFile(model_path, 'r') as zin:
    with zipfile.ZipFile(model_path + '.tmp', 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'config.json':
                config = json.loads(data)
                # Fix batch_shape
                def fix(obj):
                    if isinstance(obj, dict):
                        if 'batch_shape' in obj:
                            batch_shape = obj.pop('batch_shape')
                            if batch_shape and len(batch_shape) > 1:
                                obj['input_shape'] = batch_shape[1:]
                        for v in obj.values():
                            fix(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            fix(item)
                fix(config)
                data = json.dumps(config, indent=2).encode('utf-8')
            zout.writestr(item, data)

os.replace(model_path + '.tmp', model_path)
print("Done! Model fixed. Backup saved to:", backup_path)





