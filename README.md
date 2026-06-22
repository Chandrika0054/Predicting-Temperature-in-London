# Predicting-Temperature-in-London

# Predicting Temperature in London —  End-to-End Regression Pipeline using Scikit-Learn and MLflow Tracking

As the climate changes, predicting the weather is critical. Since the weather depends on a lot of different factors, running a lot of experiments to determine what the best approach is to predict the weather is important. In this project, I'm running experiments for different regression models predicting the mean temperature, using a combination of sklearn and MLflow.

## Project Brief
 
Use machine learning to predict the mean temperature in London, England, logging your root mean squared error (RMSE) metrics using `mlflow`.
 
- Build a model to predict `"mean_temp"` with an RMSE of 3 or less.
- Use MLflow to log any models you train, their hyperparameters, and respective RMSE scores (ensuring you include `"rmse"` as part of the metric name).
- Search all of your `mlflow` runs and store the results as a variable called `experiment_results`.
## Approach
 
**1. Loading & cleaning the data** — Read `london_weather.csv`, parse `date` with `pd.to_datetime(format="%Y%m%d")`, and extract `month` and `year`.
 
**2. EDA** — Use a correlation heatmap to identify predictors. `sunshine`, `global_radiation`, and `cloud_cover` are most strongly related to `mean_temp`.
 
**3. Feature selection** — Use `month`, `cloud_cover`, `sunshine`, `global_radiation`, `precipitation`, and `pressure`; drop rows with a missing target. Two deliberate choices here:
- **`max_temp` and `min_temp` are excluded** despite correlating ~0.9 with `mean_temp` — mean temperature is essentially derived from them, so including them would be **data leakage** (artificially low error that wouldn't generalise).
- **`month` is kept** even though its linear correlation is weak (~0.23): temperature is seasonal/cyclical, a non-linear pattern that tree-based models can capture through splits.
  
**4. Preprocessing the data with no leakage** — `train_test_split`, then `SimpleImputer` and `StandardScaler` fit on the training set only (`fit_transform` on train, `transform` on test).
 
**5. Machine learning training and evaluation** — Fit Linear Regression, Decision Tree, and Random Forest models across several `max_depth` values. Each model, its hyperparameters, and its RMSE (metric name contains `"rmse"`) are logged in a dedicated MLflow run.
 
**6. Search runs of logged results** — Collect every run into a DataFrame: `experiment_results = mlflow.search_runs()`.
 
## Result
 
Across all runs, the **Random Forest at `max_depth=10`** was the best model, achieving **RMSE ≈ 2.82 °C** on the test set — the only run that meets the target of RMSE ≤ 3.
 
| max_depth | Linear Reg | Decision Tree | Random Forest |
|:--:|:--:|:--:|:--:|
| 1 | 3.87 | 4.75 | 4.70 |
| 2 | 3.87 | 3.92 | 3.83 |
| 10 | 3.87 | 3.07 | **2.82** |
 
Tree-based models improve sharply with depth (shallow models underfit), and the ensemble Random Forest edges out a single tree at every depth by reducing variance. Linear Regression stays flat at 3.87 — it has no `max_depth`, so it acts as a baseline and shows the feature–temperature relationship is non-linear. The full comparison lives in `experiment_results` and the MLflow UI.
 
## Tech stack
 
Python · pandas · scikit-learn · matplotlib / seaborn · MLflow
 
## Run
 
```bash
pip install pandas scikit-learn matplotlib seaborn mlflow
jupyter notebook        # run the analysis
mlflow ui               # view experiments
