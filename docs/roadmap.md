# Wind Power Prediction Using Gaussian Processes

## Tasks

Use a GP to forecast wind power generation, incorporating wind speed and power data. At each time point, predict the hourly output for the next day. Analyze the forecast errors for the different forecast horizons using the RMSE and CRPS. Use the last available year of data as your test set.

Predict the sum of Offshore Wind and Onshore_Wind (you can choose whether to forecast both and then sum the forecasts or forecast the sum) from the file: Realised_Supply_Germany

Bonus task: Test different feature selection approaches for the weather input.

## Data

The data is available in the following files:

- Realised_Supply_Germany.csv
- Realised_Demand_Germany.csv
- Weather_Data_Germany.csv
- Prices_Europe.csv

## Roadmap

- [x] **Check Data Availabe**: What data is available? What are the time periods of the datasets? Are there any missing values?
- [x] **Data Preprocessing**: Load and preprocess the data, including handling missing values and merging datasets.
- [x] **Feature Engineering**: Create relevant features from the weather data and the power generation data.
- [ ] **Model Selection**: Choose a GP Regression model.
- [x] **Model Training**: Train the GP model on the training set, using the last available year of data as the test set.
- [x] **Model Evaluation**: Evaluate the model's performance using RMSE and CRPS metrics.
- [x] **Visualization**: Visualize the forecast results and errors for different forecast horizons.
- [x] **Documentation**: Document the code and results, including explanations of the methods used and their implications.
- [ ] **Bonus Task**: Implement different feature selection approaches for the weather input and evaluate their impact on the model's performance.

