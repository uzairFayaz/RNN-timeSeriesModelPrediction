# Milk Production Prediction Model

## Overview
This project builds a time series prediction model to forecast milk production for the next 12 months based on historical data. The model uses deep learning techniques with TensorFlow/Keras and LSTM (Long Short-Term Memory) networks to capture time-based dependencies in the data.

## Dataset
The dataset consists of monthly milk production records from 1962 to 1975. It includes:
- **Month**: The timestamp for milk production records.
- **Milk Production**: The quantity of milk produced per month (in some unit).

## Model Architecture
The model is built using LSTM (Long Short-Term Memory) networks, which are a type of recurrent neural network (RNN). The key components include:
- **Data Preprocessing**: Normalization, reshaping, and creating time-series sequences.
- **LSTM Layers**: Capturing temporal dependencies in data.
- **Dropout Regularization**: Preventing overfitting.
- **Dense Layer**: Producing the final prediction.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

## Training the Model
1. Load the dataset and preprocess it (convert timestamps, normalize values, create sequences).
2. Split the dataset into training and testing sets.
3. Define the LSTM model architecture.
4. Train the model using historical data.
5. Evaluate the model’s performance using loss functions (MSE, RMSE).

## Making Predictions
- The trained model is used to predict milk production for the next 12 months.
- The results are stored in the `Generated` column of the test dataset.

## How to Run the Project
```python
# Load necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load and preprocess data
# Define LSTM model
# Train the model
# Make predictions
```

## Visualizing Predictions
To compare actual vs. predicted milk production:
```python
plt.figure(figsize=(12,6))
plt.plot(test_set.index, test_set['Milk Production'], label='Actual Milk Production')
plt.plot(test_set.index[-12:], predictions, label='Predicted Milk Production', linestyle='dashed')
plt.legend()
plt.show()
```

## Potential Improvements
- Experiment with different LSTM architectures.
- Tune hyperparameters (batch size, epochs, learning rate).
- Use more advanced time-series forecasting techniques (Prophet, ARIMA, Transformer-based models).

## Author
Uzair Fayaz

## License
This project is open-source and available for modification and distribution.

