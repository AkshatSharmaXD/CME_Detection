# 🔍 How to Check Sunspot Number Manually - Step by Step

## Method 1: Quick Python Check (Easiest)

### Step 1: Open Terminal/Command Prompt
Navigate to the forecast directory:
```bash
cd "D:\sih\Cme_detection_Phased 2\backend\scripts\7_days_forecast"
```

### Step 2: Run This Command
```bash
python check_sunspot_manually.py
```

**This will show you:**
- ✅ If Sunspot_Number exists in data
- ✅ If it's a feature (input) or target (output)
- ✅ Complete summary

---

## Method 2: Check Data File Directly

### Step 1: Open the CSV file
Open: `omni_data_updatedyears.csv` in Excel or any text editor

### Step 2: Look for the column
- Scroll to find column named: **`Sunspot_Number`**
- You'll see it has values like: 26.0, 30.0, etc.

### Step 3: Check training data
Open: `train_data_scaled.csv`
- Look for column: **`Sunspot_Number`**
- Values will be scaled (like: -0.65, 0.23, etc.)

---

## Method 3: Check Notebook (Most Detailed)

### Step 1: Open Jupyter Notebook
Open: `notebook/forecast_7days.ipynb`

### Step 2: Go to Cell 4
Look for this code:
```python
target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
feature_cols = [col for col in train_scaled.columns if col not in exclude_cols]
```

### Step 3: Check the output
The output will show:
```
Feature columns (26):
  1. IMF_Mag_Avg_nT
  2. IMF_Lat_deg
  ...
  10. Sunspot_Number  ← HERE IT IS!
  ...
```

### Step 4: Check targets
Look for:
```
Target variables (3): ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
```
**Notice**: Sunspot_Number is NOT in this list (it's only in features)

---

## Method 4: Simple Python One-Liner

### In Terminal:
```bash
python -c "import pandas as pd; df = pd.read_csv('omni_data_updatedyears.csv', nrows=1); print('✅ Sunspot_Number exists:', 'Sunspot_Number' in df.columns); print('Columns:', [c for c in df.columns if 'sunspot' in c.lower()])"
```

**Expected Output:**
```
✅ Sunspot_Number exists: True
Columns: ['Sunspot_Number']
```

---

## Method 5: Check Model Code

### Step 1: Open the model runner
Open: `forecast_model_runner.py`

### Step 2: Look at line 78
You'll see:
```python
'Alpha_Proton_Ratio', 'Sunspot_Number', 'F10.7_Index',
```

### Step 3: Look at line 32
You'll see:
```python
self.target_vars = ['Dst_Index_nT', 'Kp_10', 'ap_index_nT']
```
**Notice**: Sunspot_Number is NOT in target_vars (only in features)

---

## ✅ What You Should See

### ✅ Sunspot_Number IS:
- In `omni_data_updatedyears.csv` (column name)
- In `train_data_scaled.csv` (column name)
- In feature list (Feature #10 of 26)
- Used by model as INPUT

### ❌ Sunspot_Number is NOT:
- In target_vars list
- Predicted as OUTPUT
- Shown in model predictions

---

## 📋 Quick Checklist

- [ ] Open `omni_data_updatedyears.csv` → Find `Sunspot_Number` column
- [ ] Open `train_data_scaled.csv` → Find `Sunspot_Number` column  
- [ ] Run `python check_sunspot_manually.py` → See it's Feature #10
- [ ] Check `forecast_7days.ipynb` Cell 4 → See it in feature_cols
- [ ] Check `forecast_7days.ipynb` Cell 4 → See it NOT in target_vars

---

## 🎯 Conclusion

**Sunspot_Number is in the model as INPUT (Feature #10)**
**But NOT as OUTPUT (only DST, Kp, Ap are outputs)**

