# LSTM Model Information

## Model File
- **File**: `models/lstm_baseline_dst.keras`
- **Format**: Keras SavedModel format (.keras)
- **Type**: LSTM (Long Short-Term Memory) Neural Network

## Training Dataset
- **Training Samples**: 77,569 sequences
- **Test Samples**: 19,213 sequences
- **Date Range**: 2008-12-01 to 2019-12-31
- **Total Years**: ~11 years of historical data
- **Data Source**: OMNI data (NASA/NOAA)

## Model Architecture

### Input Layer
- **Lookback Period**: 72 hours (3 days of historical data)
- **Features**: 26 features
- **Input Shape**: `(samples, 72, 26)`

### LSTM Layers
1. **First LSTM Layer**
   - Units: 128
   - Activation: tanh
   - Return Sequences: True
   - Dropout: 0.2

2. **Second LSTM Layer**
   - Units: 64
   - Activation: tanh
   - Return Sequences: False
   - Dropout: 0.2

### Dense Layers
3. **Dense Layer**
   - Units: 128
   - Activation: ReLU
   - Dropout: 0.2

4. **Output Layer**
   - Units: 168 * 3 = 504 (forecast_horizon * targets)
   - Output Shape: `(samples, 168, 3)` after reshaping

### Output
- **Forecast Horizon**: 168 hours (7 days ahead)
- **Targets**: 3 parameters
  - `Dst_Index_nT` (Disturbance Storm Time Index)
  - `Kp_10` (Planetary K-index)
  - `ap_index_nT` (Planetary A-index)
- **Output Shape**: `(samples, 168, 3)`

## Feature Set (26 Features)

### Solar Wind Parameters
1. IMF_Mag_Avg_nT (Interplanetary Magnetic Field Magnitude)
2. IMF_Lat_deg (IMF Latitude)
3. IMF_Long_deg (IMF Longitude)
4. Bz_GSM_nT (Bz component in GSM coordinates)
5. Proton_Density_n_cc (Proton Density)
6. Flow_Speed_km_s (Solar Wind Speed)
7. Flow_Longitude_deg (Flow Longitude)
8. Flow_Latitude_deg (Flow Latitude)
9. Alpha_Proton_Ratio

### Solar Activity
10. Sunspot_Number
11. F10.7_Index (Solar Radio Flux)

### Proton Flux (Multiple Energies)
12. Proton_Flux_1MeV
13. Proton_Flux_2MeV
14. Proton_Flux_4MeV
15. Proton_Flux_10MeV
16. Proton_Flux_30MeV
17. Proton_Flux_60MeV

### Time Features
18. Hour_of_Day
19. Day_of_Week
20. Day_of_Year
21. Month
22. Season
23. Hour_Sin (Cyclical encoding)
24. Hour_Cos (Cyclical encoding)
25. DOY_Sin (Day of Year - Cyclical)
26. DOY_Cos (Day of Year - Cyclical)

## Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: MSE (Mean Squared Error)
- **Metrics**: MAE (Mean Absolute Error)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 0.2 (20%)
- **Early Stopping**: Patience=10, restore_best_weights=True

## Model Capabilities
- **Multi-step Forecasting**: Predicts 168 hours (7 days) ahead
- **Multi-output**: Predicts 3 parameters simultaneously
- **Temporal Dependencies**: Captures long-term patterns using 72-hour lookback
- **Real-time Predictions**: Can generate forecasts from recent data

## Usage
The model takes the last 72 hours of space weather data and predicts the next 168 hours (7 days) for:
- DST Index (geomagnetic activity)
- Kp Index (planetary geomagnetic activity)
- Ap Index (planetary geomagnetic activity)

## Data Preprocessing
- Data is scaled using StandardScaler (saved as `scaler.pkl`)
- Features are normalized before input to the model
- Predictions need to be inverse-transformed to original scale

