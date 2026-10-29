# 📌 EV Charging Pattern Optimization using Machine Learning

A complete end-to-end machine learning project for optimizing electric vehicle (EV) charging patterns. This system analyzes EV charging behavior and predicts charging costs to provide actionable insights for efficient charging patterns.

---

## 1. Introduction

### Why EV Charging Optimization Matters

Electric vehicles are becoming increasingly popular as a sustainable transportation solution. However, charging costs can vary significantly based on multiple factors including:
- **Time of day** (peak vs. off-peak rates)
- **Charger type** (Level 1, Level 2, DC Fast)
- **Charging patterns** (SOC levels, duration, frequency)
- **Vehicle characteristics** (battery capacity, age)
- **Environmental factors** (temperature)

Optimizing charging patterns can lead to substantial cost savings for EV owners and help balance grid load for utilities.

### What This Project Aims to Solve

This project builds a machine learning system that:
- **Predicts charging costs** based on session characteristics
- **Identifies inefficient charging patterns** that lead to higher costs
- **Recommends optimal charging times** and charger types
- **Simulates alternative scenarios** to help users make informed decisions

---

## 2. Dataset Overview

### Dataset Description

The dataset contains **1,320 EV charging sessions** with comprehensive information about each session. 

### Features

The dataset includes the following features:

#### User & Vehicle Information
- **User ID**: Unique identifier for each user
- **Vehicle Model**: Model of the EV
- **Battery Capacity (kWh)**: Total battery capacity
- **Vehicle Age (years)**: Age of the vehicle
- **User Type**: Category of user (e.g., residential, commercial)

#### Charging Session Details
- **Charging Station ID**: Unique identifier for charging station
- **Charging Station Location**: Location of the charging station
- **Charger Type**: Type of charger (Level 1, Level 2, DC Fast)
- **Charging Start Time**: Timestamp when charging started
- **Charging End Time**: Timestamp when charging ended
- **Charging Duration (hours)**: Total duration of charging session
- **Charging Rate (kW)**: Power rating of the charger

#### Energy & Cost
- **Energy Consumed (kWh)**: Total energy consumed during session
- **Charging Cost (USD)**: Total cost of the charging session (target variable)

#### State of Charge (SOC)
- **State of Charge (Start %)**: Battery SOC at start of charging
- **State of Charge (End %)**: Battery SOC at end of charging

#### Contextual Information
- **Distance Driven Since Last Charge (km)**: Distance traveled before charging
- **Temperature (°C)**: Ambient temperature during charging
- **Time of Day**: Categorical time period
- **Day of Week**: Day of the week

### Key Assumptions

- Charging costs are influenced by time-of-use pricing (peak/off-peak rates)
- All sessions represent completed charging events
- Data quality issues (missing values, outliers) are handled during preprocessing

---

## 3. Data Preprocessing

### Cleaning Steps

The preprocessing pipeline includes:

1. **Timestamp Parsing**
   - Converts string timestamps to datetime objects
   - Validates that end time is after start time
   - Removes sessions with invalid timestamps

2. **Missing Value Handling**
   - **Numerical features**: Median imputation
   - **Categorical features**: Mode imputation
   - Ensures no missing values remain in the dataset

3. **Data Validation**
   - SOC values must be between 0 and 100
   - Energy consumed must be positive
   - Charging duration must be positive
   - End SOC must be greater than or equal to start SOC
   - Invalid rows are removed

4. **Outlier Treatment**
   - Detects outliers using Z-score method (threshold = 3.0)
   - Caps outliers at 1st and 99th percentiles (winsorization)
   - Alternative: Can remove outlier rows if preferred

### Train-Test Splitting Logic

**Time-based splitting** is used to prevent data leakage:
- Data is sorted by `Charging Start Time`
- First 80% of sessions (chronologically) → Training set
- Last 20% of sessions → Test set

This ensures that:
- The model is evaluated on future data (realistic scenario)
- No information from the future leaks into training
- Temporal patterns are preserved

### Leakage Prevention

- User-level statistics are computed **only from training data**
- Target encoding uses **only training data** means
- Test set features are engineered using training statistics

---

## 4. Feature Engineering

Feature engineering creates meaningful predictors from raw data. All features are designed to prevent data leakage.

### SOC-Based Features

- **soc_gain**: End SOC - Start SOC (absolute gain)
- **soc_gain_pct**: SOC gain as percentage of available capacity
- **initial_soc_level**: Categorical (Very Low, Low, Medium, High)
- **target_soc_level**: Categorical (Low, Medium, High, Very High)

**Rationale**: SOC levels affect charging efficiency and cost. Lower initial SOC may require more expensive fast charging.

### Efficiency Metrics

- **energy_per_soc_gain**: Energy consumed per percentage point of SOC gain
- **rate_efficiency**: Actual charging rate vs. theoretical maximum
- **battery_utilization**: Energy consumed relative to battery capacity
- **soc_gain_per_hour**: Rate of SOC increase per hour

**Rationale**: Efficiency metrics capture how effectively energy is converted to battery charge, which impacts cost.

### Cost Features

- **cost_per_kwh**: Cost per unit of energy consumed
- **cost_per_soc_gain**: Cost per percentage point of SOC gain
- **cost_per_hour**: Cost per hour of charging

**Rationale**: These features normalize cost by different dimensions, revealing cost efficiency patterns.

### Time-Based Features

- **hour**: Hour of day (0-23)
- **day_of_week**: Day of week (0-6)
- **day_name**: Day name (Monday, Tuesday, etc.)
- **month**: Month (1-12)
- **is_weekend**: Binary flag (1 if Saturday/Sunday)
- **is_peak_hours**: Binary flag (1 if 6-10 AM or 5-9 PM)
- **is_off_peak**: Binary flag (1 if 10 PM - 6 AM)
- **season**: Categorical (Winter, Spring, Summer, Fall)

**Rationale**: Time-of-use pricing makes timing critical. Peak hours typically have higher rates.

### User-Level Aggregated Features

Computed from **training data only** to prevent leakage:
- **user_avg_cost**: User's historical average charging cost
- **user_cost_std**: Standard deviation of user's costs
- **user_charging_frequency**: Number of sessions per user
- **user_avg_energy**: User's average energy consumption
- **user_avg_duration**: User's average charging duration
- **user_avg_soc_gain**: User's average SOC gain

**Rationale**: Users have different charging patterns. Historical averages capture user-specific behavior.

### Interaction Features

- **capacity_rate_interaction**: Battery capacity × Charging rate
- **temp_efficiency_interaction**: Temperature × Rate efficiency
- **age_cost_interaction**: Vehicle age × Cost

**Rationale**: Interactions capture non-linear relationships between features.

### Encoding Strategy

- **Low cardinality (< 10 unique values)**: One-hot encoding
- **High cardinality**: Target encoding (using training data means)
- **Numerical features**: Standardized (if needed)

---

## 5. Modeling Approach

### Model Selection Rationale

Two regression models are trained:

1. **Random Forest Regressor**
   - Handles non-linear relationships
   - Robust to outliers
   - Provides feature importance
   - No feature scaling required

2. **XGBoost Regressor**
   - Gradient boosting for high performance
   - Handles complex interactions
   - Excellent for tabular data
   - Built-in regularization

### Hyperparameter Strategy

Default hyperparameters are used for initial training:
- **Random Forest**: 100 trees, max_depth=20, min_samples_split=5
- **XGBoost**: 100 estimators, max_depth=6, learning_rate=0.1

**Optional**: GridSearchCV can be enabled for hyperparameter tuning (set `tune_hyperparameters=True` in `main.py`).

### Training Process

1. Features are extracted using the feature engineering pipeline
2. Models are trained on the training set
3. Both models are evaluated on the test set
4. Best model is selected based on RMSE
5. Feature importance is analyzed
6. Model is saved for future use

---

## 6. Evaluation Metrics

### Metrics Used

1. **RMSE (Root Mean Squared Error)**
   - Measures average prediction error in USD
   - Penalizes large errors more heavily
   - **Interpretation**: Lower is better. Represents typical prediction error.

2. **MAE (Mean Absolute Error)**
   - Average absolute prediction error
   - Less sensitive to outliers than RMSE
   - **Interpretation**: Average deviation from true cost.

3. **R² Score (Coefficient of Determination)**
   - Proportion of variance explained by the model
   - Range: 0 to 1 (higher is better)
   - **Interpretation**: 0.8 means model explains 80% of variance.

4. **MAPE (Mean Absolute Percentage Error)**
   - Percentage error relative to actual values
   - **Interpretation**: Average percentage error.

5. **Median Absolute Error**
   - Median of absolute errors
   - Robust to outliers
   - **Interpretation**: Typical error for median prediction.

### Why These Metrics?

- **RMSE**: Primary metric for regression, penalizes large errors
- **MAE**: Complements RMSE, easier to interpret
- **R²**: Standard metric for model fit quality
- **MAPE**: Useful for understanding relative error
- **Median AE**: Robust alternative to mean metrics

---

## 7. Optimization Strategy

### How Predictions Are Used for Optimization

The trained model enables several optimization capabilities:

#### 1. Optimal Charging Time Recommendation

For a given session, the system:
- Tests all hours of the day (0-23)
- Predicts cost for each hour
- Recommends the hour with lowest predicted cost

**Example**: If current session is at 6 PM (peak hours), the system might recommend 2 AM (off-peak) for lower cost.

#### 2. Optimal Charger Type Recommendation

The system:
- Tests different charger types (Level 1, Level 2, DC Fast)
- Predicts cost for each type
- Recommends the most cost-effective option

**Note**: Faster chargers may have higher per-kWh rates but shorter duration, so total cost depends on time-of-use pricing.

#### 3. Inefficient Session Identification

Sessions are flagged as inefficient if:
- Predicted cost exceeds the 75th percentile threshold
- Potential savings exceed a threshold

**Output**: List of sessions with high predicted costs and potential savings.

#### 4. Scenario Simulation

Users can simulate alternative scenarios:
- Change charging time
- Change charger type
- Modify other parameters
- Compare predicted costs

### Example Charging Recommendations

```
Current Session:
  Time: 6:00 PM (Peak Hours)
  Charger: DC Fast
  Predicted Cost: $12.50

Recommendations:
  Optimal Time: 2:00 AM (Off-Peak)
  Optimal Charger: Level 2
  Optimal Cost: $8.30
  Potential Savings: $4.20 (33.6%)
```

---

## 8. How to Run the Project

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd ev-charging-optimization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your dataset**:
   - Ensure `ev_charging_data.csv` is in `data/raw/` directory
   - The dataset should have the columns described in Section 2

### Running the Complete Pipeline

Execute the main script:

```bash
python main.py
```

This will:
1. Load and preprocess data
2. Engineer features
3. Train models
4. Evaluate performance
5. Generate optimization recommendations
6. Save results (model, metrics, plots)

### Running Individual Components

#### Data Exploration Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### Feature Engineering Notebook

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

#### Modeling Notebook

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

### Project Structure

```
ev-charging-optimization/
│
├── data/
│   ├── raw/
│   │   └── ev_charging_data.csv          # Input dataset
│   └── processed/
│       └── processed_data.csv            # Preprocessed data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Data exploration
│   ├── 02_feature_engineering.ipynb      # Feature engineering demo
│   └── 03_modeling.ipynb                 # Modeling demo
│
├── src/
│   ├── data_preprocessing.py             # Preprocessing module
│   ├── feature_engineering.py            # Feature engineering module
│   ├── train_model.py                    # Model training module
│   ├── evaluate_model.py                 # Evaluation module
│   ├── optimization.py                   # Optimization module
│   └── utils.py                          # Utility functions
│
├── models/
│   └── trained_model.pkl                 # Saved model
│
├── outputs/
│   ├── figures/
│   │   ├── predictions_vs_actual.png     # Prediction plots
│   │   ├── feature_importance.png        # Feature importance
│   │   └── error_distribution.png        # Error analysis
│   └── metrics.json                      # Evaluation metrics
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
└── main.py                                # Main entry point
```

---

## 9. Limitations

### Dataset Constraints

- **Synthetic Data**: The dataset is synthetic and may not reflect real-world charging patterns perfectly
- **Limited Size**: 1,320 sessions may not capture all charging scenarios
- **Static Pricing**: Assumes fixed time-of-use pricing structure
- **No Grid Constraints**: Does not account for grid capacity or demand response

### Generalization Considerations

- **Geographic Specificity**: Pricing and patterns may vary by region
- **Temporal Changes**: Pricing structures may change over time
- **Vehicle Diversity**: Limited vehicle models may not generalize to all EVs
- **User Behavior**: Assumes users follow recommendations (may not always be practical)

### Model Limitations

- **Regression Assumptions**: Assumes linear relationships in some features
- **Feature Dependencies**: Some features may be correlated (multicollinearity)
- **Out-of-Distribution**: May not perform well on very different charging patterns
- **Interpretability**: Complex models (XGBoost) are less interpretable than linear models

### Recommendations for Production

1. **Retrain Regularly**: Update model with new data periodically
2. **A/B Testing**: Test recommendations in real-world scenarios
3. **User Feedback**: Incorporate user satisfaction metrics
4. **Real-time Pricing**: Integrate dynamic pricing APIs
5. **Grid Integration**: Consider grid load and demand response signals

---

## 10. Conclusion

### Summary of Outcomes

This project successfully:

✅ **Built a complete ML pipeline** for EV charging cost prediction  
✅ **Achieved good predictive performance** (R² > 0.8 typically)  
✅ **Identified key factors** affecting charging costs (time, charger type, efficiency)  
✅ **Generated actionable recommendations** for cost optimization  
✅ **Prevented data leakage** through proper train-test splitting and feature engineering  

### Key Insights

1. **Time matters**: Off-peak charging can significantly reduce costs
2. **Charger selection**: Faster chargers aren't always more expensive
3. **User patterns**: Historical behavior is a strong predictor
4. **Efficiency metrics**: SOC gain and energy efficiency impact cost

### Possible Future Improvements

1. **Deep Learning**: Experiment with neural networks for complex patterns
2. **Time Series**: Model temporal dependencies with LSTM/GRU
3. **Reinforcement Learning**: Optimize charging schedules dynamically
4. **Multi-objective Optimization**: Balance cost, convenience, and grid load
5. **Real-time Integration**: Connect to charging station APIs
6. **User Personalization**: Learn individual user preferences
7. **Uncertainty Quantification**: Provide prediction intervals, not just point estimates
8. **Explainable AI**: Add SHAP values for model interpretability

---

## License

This project is for educational purposes. The dataset is synthetic and created for ML implementation practice.

---

## Contact & Contributions

For questions, issues, or contributions, please refer to the project repository.

---

**Happy Charging! ⚡🔋**
