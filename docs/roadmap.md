# Wind Power Prediction Using Gaussian Processes

## Tasks

Use a GP to forecast wind power generation, incorporating wind speed and power data. At each time point, predict the hourly output for the next day. Analyze the forecast errors for the different forecast horizons using the RMSE and CRPS. Use the last available year of data as your test set.

Predict the sum of Offshore Wind and Onshore_Wind (you can choose whether to forecast both and then sum the forecasts or forecast the sum) from the file: Realised_Supply_Germany

Bonus task: Test different feature selection approaches for the weather input.

## Data

The data is available in the following files:

- Realised_Supply_Germany.csv
- Weather_Data_Germany.csv

## Roadmap

- [x] **Check Data Availabe**: What data is available? What are the time periods of the datasets? Are there any missing values?
- [x] **Data Preprocessing**: Load and preprocess the data, including handling missing values and merging datasets.
- [x] **Feature Engineering**: Create relevant features from the weather data and the power generation data.
- [x] **Model Selection**: Choose a GP Regression model.
- [x] **Model Training**: Train the GP model on the training set, using the last available year of data as the test set.
- [x] **Model Evaluation**: Evaluate the model's performance using RMSE and CRPS metrics.
- [x] **Visualization**: Visualize the forecast results and errors for different forecast horizons.
- [x] **Documentation**: Document the code and results, including explanations of the methods used and their implications.
- [ ] **Bonus Task**: Implement different feature selection approaches for the weather input and evaluate their impact on the model's performance.
- [ ] **Discussion**: Discuss the learnings and takeaways. How to continue and why GPs might not be the optimal architecture for the given task.

## Questions

- Switch presentation date with other students? - Si
- Single notebook? I have some imports to reduce space inside the notebook. - Si
- Enough markdown text? - Si
- ApproxGP:
  - Is the intended architecture, correct? Compared to something like XGBoost, it is computationally way more expensive. 2 hours for 100 epochs with the current features.
  - Retraining the model is expensive, how to go about it? - Feature selection
  - Kernel choice? - Fine
- Feature selection:
  - How to select features based on correlation and mutual information?
  - Lasso feature selection on train and permutation importance on test provide completely different results. What to do?
  
  Answer
  - Locality of weather data e.g. north south split
  - Reduce number of lags e.g. [1, 3, 6, 12, 18, 24]
  - Apply the target vs lags and time features correlation and MI scores
  - Select the joint set of all selection methods

- Discussion TODO
