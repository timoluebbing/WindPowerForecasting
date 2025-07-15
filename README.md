# Wind  Power Forecasting using Gaussian Processes

A collection of notebooks and modules for forecasting wind power production in Germany using Gaussian Processes and related techniques.

## Data

The dataset consists of historical wind power production data from the "Realised Supply Germany" dataset, along with weather data from multiple stations. The data is available in CSV format ([download](https://nc.mlcloud.uni-tuebingen.de/index.php/s/K8cQs6WeYZKNnas)) and includes:

- Wind power production (in MW) from 2017 to 2023
- Weather data (temperature, wind speed, etc.) from various weather stations in Germany.

![Wind Power Composition](/figures/wind_power_composition.png)

## Project Structure

- **data/** — raw CSV data
- **src/** — helper modules (`preprocessing`, `visualizations`, `checkpoint`, `feature_selection`, etc.)  
- **checkpoints/** — saved model checkpoints  
- **experiments/**  
  - **tutorial.ipynb** — end-to-end workflow (see below)  
  - simple_model.ipynb — testing baseline GP regression on a small horizon  
  - `approximateGP.ipynb`, `demo_approx.ipynb`, (testing)

## Tutorial

See [tutorial.ipynb](experiments/tutorial.ipynb) for a detailed step-by-step guide:

1. **Data Loading & Preprocessing**  
   - Load and resample “Realised Supply Germany” & weather data  
   - Clean, merge, and average weather stations  

2. **Feature Engineering**  
   - Create cyclical time features (hour/day/month/year)  
   - Generate lag features via a sliding-window approach  

3. **Target Scaling**  
   - Standardize the wind-sum target on training data  

4. **Feature Selection**  
   - Compute Pearson correlation & mutual information scores  
   - Visualize feature metrics  
   - Fit LassoCV pipelines to rank and select top features  

5. **Approximate GP Forecasting (SVGP)**  
   - Define `SparseGPModel` with inducing points  
   - Set up one model per forecast horizon (24-step ahead)  
   - Load/save epoch-aware checkpoints  

6. **Training & Evaluation**  
   - Continue training from saved epochs or train from scratch  
   - Compute RMSE and CRPS on the test set  
   - Visualize multi‐step forecasts, uncertainty bands, and error metrics  

## Results

While the used models are not fully optimizated with regards to hyperparameters and the purpose of this tutorial is not to achive state of the art results, but rather provide a comprehensive overview of the ML pipeline for time series forecasting, the following results were achieved:

Full horizon best test sample:
![Forecasting Results](/figures/test_sample_6450.png)

Full horizon worst test sample:
![Forecasting Results](/figures/test_sample_725.png)

The SVGP model performs resonably well for short forecasting horizons, but struggles with the long-term forecast horizon. Single horizon forecast along the first 2 month of the test set (horizon=3):
![Single Horizon Forecast](/figures/test_horizon_3_n_hours_1344.png)


## Installation

```bash
git clone https://github.com/timoluebbing/WindPowerForecasting
cd WindPowerForecasting
pip install -r requirements.txt
```

> [!IMPORTANT]
> Training the SVGP model relies on GPU support. Please install a `torch` version compatible with your CUDA version.

## Usage

Run the notebook [tutorial.ipynb](experiments/tutorial.ipynb) to execute the entire workflow from data loading to forecasting. Feel free to add and modify the code as needed for your experiments. Please have a look at the **Future Work** section at the end of the tutorial for next steps.

---

*Developed by Timo Lübbing for educational purposes (University of Tübingen).*
