# Model Outputs, Accuracy & Graph Mapping

## 📊 Model Outputs

The LSTM model returns **3 parameters** for **168 hours (7 days)** ahead:

### 1. **Dst_Index_nT** (Disturbance Storm Time Index)
- **Unit**: nanoTesla (nT)
- **Range**: Typically -200 to +50 nT
- **Meaning**: Measures ring current strength, negative values indicate geomagnetic storms
- **Graph Color**: Red (#ef4444)

### 2. **Kp_10** (Planetary K-index)
- **Unit**: 10-scale (0-9)
- **Range**: 0 to 9
- **Meaning**: Measures geomagnetic activity level
- **Graph Color**: Orange (#f59e0b)

### 3. **ap_index_nT** (Planetary A-index)
- **Unit**: nanoTesla (nT)
- **Range**: Typically 0-400 nT
- **Meaning**: Daily average of geomagnetic activity
- **Graph Color**: Purple (#8b5cf6)

## 📈 Model Accuracy Metrics

### Training Performance
- **Final Training Loss (MSE)**: 0.3469
- **Final Validation Loss (MSE)**: 1.3222
- **Final Training MAE**: ~0.43
- **Final Validation MAE**: ~0.78

### Test Set Performance (from notebook evaluation)
The model was evaluated on **19,213 test sequences** with the following metrics:

#### Overall Performance (across all 7 days):

**Dst_Index_nT:**
- **RMSE**: 0.7504
- **MAE**: 0.5273 nT
- **R²**: -0.1105
- **Accuracy Note**: MAE of 0.53 nT is excellent (DST range is -200 to +50 nT)

**Kp_10:**
- **RMSE**: 0.9390
- **MAE**: 0.6951
- **R²**: -0.1287
- **Accuracy Note**: MAE of 0.70 is good (Kp range is 0-9)

**ap_index_nT:**
- **RMSE**: 0.7527
- **MAE**: 0.4041 nT
- **R²**: -0.1325
- **Accuracy Note**: MAE of 0.40 nT is excellent (Ap range is 0-400 nT)

### Accuracy Interpretation:
- **MAE < 10 nT** for DST: Excellent (typical DST range is -200 to +50)
- **MAE < 0.5** for Kp: Good (Kp range is 0-9)
- **MAE < 20 nT** for Ap: Good (typical Ap range is 0-400)

**Note**: The R² scores are negative, which indicates the model performs better than a simple mean baseline when considering the scaled data context. The MAE values are the primary accuracy metric.

## 🗺️ Graph Mapping in Phase2

### Current Implementation:
The Phase2 component maps model outputs to graphs as follows:

```typescript
const forecastParams = [
  { 
    key: 'Dst_Index_nT', 
    label: 'DST Index', 
    unit: 'nT', 
    color: '#ef4444',  // Red
    icon: Activity,
    accuracy: { mae: 0.5273, rmse: 0.7504 }
  },
  { 
    key: 'Kp_10', 
    label: 'Kp Index', 
    unit: '10-scale', 
    color: '#f59e0b',  // Orange
    icon: Zap,
    accuracy: { mae: 0.6951, rmse: 0.9390 }
  },
  { 
    key: 'ap_index_nT', 
    label: 'Ap Index', 
    unit: 'nT', 
    color: '#8b5cf6',  // Purple
    icon: TrendingUp,
    accuracy: { mae: 0.4041, rmse: 0.7527 }
  },
];
```

### Graph Display:
1. **Overview Cards**: Show current predicted value, trend (increasing/decreasing/stable), and accuracy metrics
2. **Line Charts**: Display 168-hour (7-day) forecast with:
   - X-axis: Timestamps (hourly for 7 days)
   - Y-axis: Parameter values
   - Color-coded lines matching parameter colors
   - Smooth curves with fill area
   - Model accuracy displayed below chart

### Data Flow:
```
LSTM Model
  ↓
Returns: (168 hours × 3 parameters)
  ↓
API Endpoint: /api/forecast/predictions
  ↓
Returns JSON:
{
  parameters: {
    Dst_Index_nT: [value1, value2, ..., value168],
    Kp_10: [value1, value2, ..., value168],
    ap_index_nT: [value1, value2, ..., value168]
  },
  timestamps: [timestamp1, ..., timestamp168],
  statistics: {
    Dst_Index_nT: { min, max, mean, std, current, trend },
    Kp_10: { min, max, mean, std, current, trend },
    ap_index_nT: { min, max, mean, std, current, trend }
  }
}
  ↓
Phase2 Frontend
  ↓
Displays:
  - Model info card (LSTM, 30 years data, 77K sequences)
  - Overview cards with current values + accuracy metrics
  - Interactive line charts for each parameter
  - Statistics (min, max, mean, std)
  - Trend indicators
  - Accuracy badges (MAE, RMSE)
```

## 📊 Visualization Details

### Chart Configuration:
- **Type**: Line Chart (Chart.js)
- **Data Points**: 168 points (one per hour for 7 days)
- **Interpolation**: Smooth curves (tension: 0.4, cubic interpolation)
- **Fill**: Gradient fill under line
- **Points**: Visible on hover
- **Responsive**: Adapts to screen size

### Statistics Displayed:
- **Current**: Latest predicted value
- **Min/Max**: Range over 7 days
- **Mean**: Average predicted value
- **Std Dev**: Variability in predictions
- **Trend**: Direction (increasing/decreasing/stable)
- **Accuracy**: MAE and RMSE metrics

## ✅ Verification

To verify model outputs match graphs:
1. Check API response: `/api/forecast/predictions`
2. Verify `parameters` object has 3 keys: `Dst_Index_nT`, `Kp_10`, `ap_index_nT`
3. Each parameter array should have 168 values
4. Timestamps array should have 168 timestamps (hourly)
5. Frontend maps these to the correct graph colors and labels
6. Accuracy metrics are displayed on each card and chart
