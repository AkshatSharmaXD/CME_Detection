"""
Check Sunspot Number from eda_1996.ipynb notebook
This shows what the notebook says about Sunspot_Number
"""
import json
import re

print("="*70)
print("CHECKING eda_1996.ipynb NOTEBOOK FOR SUNSPOT NUMBER")
print("="*70)

# Read the notebook
notebook_path = 'notebook/eda_1996.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"\n✓ Notebook loaded: {notebook_path}")
    print(f"   Total cells: {len(notebook['cells'])}\n")
    
    # Search for Sunspot_Number mentions
    sunspot_mentions = []
    target_mentions = []
    feature_mentions = []
    
    for i, cell in enumerate(notebook['cells']):
        cell_text = ''.join(cell.get('source', []))
        
        # Check for Sunspot_Number
        if 'Sunspot_Number' in cell_text or 'sunspot' in cell_text.lower():
            # Check if it's mentioned as target
            if 'target' in cell_text.lower() and 'Sunspot' in cell_text:
                target_mentions.append({
                    'cell': i,
                    'type': cell.get('cell_type', 'unknown'),
                    'text': cell_text[:200] + '...' if len(cell_text) > 200 else cell_text
                })
            
            # Check if it's mentioned as feature
            if ('feature' in cell_text.lower() or 'input' in cell_text.lower()) and 'Sunspot' in cell_text:
                feature_mentions.append({
                    'cell': i,
                    'type': cell.get('cell_type', 'unknown'),
                    'text': cell_text[:200] + '...' if len(cell_text) > 200 else cell_text
                })
            
            sunspot_mentions.append(i)
    
    print(f"Found {len(sunspot_mentions)} cells mentioning Sunspot_Number\n")
    
    # Check data quality section
    print("="*70)
    print("DATA QUALITY INFORMATION (from notebook)")
    print("="*70)
    
    for i, cell in enumerate(notebook['cells']):
        cell_text = ''.join(cell.get('source', []))
        output_text = ''.join([''.join(output.get('text', [])) for output in cell.get('outputs', []) if 'text' in output])
        
        # Look for data quality info
        if 'Sunspot_Number' in output_text and ('99.67' in output_text or 'completeness' in output_text.lower()):
            print(f"\nCell {i} - Data Quality:")
            print(output_text[:500])
            break
    
    # Check for target variables definition
    print("\n" + "="*70)
    print("TARGET VARIABLES DEFINITION")
    print("="*70)
    
    for i, cell in enumerate(notebook['cells']):
        cell_text = ''.join(cell.get('source', []))
        if 'target' in cell_text.lower() and ('Dst' in cell_text or 'Kp' in cell_text):
            if 'Sunspot' in cell_text:
                print(f"\nCell {i} mentions Sunspot as target:")
                print(cell_text[:300])
    
    # Check for feature list
    print("\n" + "="*70)
    print("FEATURE LIST")
    print("="*70)
    
    for i, cell in enumerate(notebook['cells']):
        cell_text = ''.join(cell.get('source', []))
        output_text = ''.join([''.join(output.get('text', [])) for output in cell.get('outputs', []) if 'text' in output])
        
        if 'Sunspot_Number' in output_text and ('IMF_Mag' in output_text or 'feature' in output_text.lower()):
            print(f"\nCell {i} - Feature List:")
            # Extract feature list
            lines = output_text.split('\n')
            for line in lines:
                if 'Sunspot_Number' in line:
                    print(f"   {line}")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY FROM NOTEBOOK")
    print("="*70)
    print(f"✅ Sunspot_Number mentioned in {len(sunspot_mentions)} cells")
    print(f"✅ Found {len(target_mentions)} mentions as potential target")
    print(f"✅ Found {len(feature_mentions)} mentions as feature")
    print("\nNote: The notebook shows Sunspot_Number was considered,")
    print("      but the final trained model (forecast_7days.ipynb) uses")
    print("      only 3 targets: DST, Kp, Ap")
    print("="*70)
    
except Exception as e:
    print(f"Error reading notebook: {e}")
    print("\nAlternative: Open the notebook in Jupyter and search for 'Sunspot_Number'")

