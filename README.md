# Find The Best Darts Local Forecasting Model

<p>A single Python script executes 26 of the Darts Local Forecasting Models and compares their accuracy. The seasonal and trend arguments required by the models are automatically derived and applied to each model.</p>
![cover image](https://github.com/markwkiehl/medium_darts_lfm/blob/a740984279afc59e3d231a0168f43a4eee908a37/darts%2025%20dataset%20LFM%20plot%20(1).png "")

## Selecting a Forecasting Model
Back in February 2024 I published ["Darts Time Series TFM Forecasting"](https://medium.com/@markwkiehl/darts-time-series-tfm-forecasting-8275ccc93a43) where I presented a complete solution for the optimization of Darts Torch Forecasting Models, and a methodology to follow that allowed you to run any model. See that article for more information about Darts, Darts time series terminology, and how to select a forecasting model.

## Darts Local Forecasting Models
<p>This article focuses on the <b>Darts Local Forecasting Models (LFM)</b>, and provides a solution for running nearly all models (26 of them) against a time series data set. The time series can be univariate, multivariate, and it can include past/future/static covariate series and encoders. Not every model supports all of these options, but the functions I am providing will automatically resolve all of that for you.
</p>

## Local Forecasting Model Arguments
<p>When you look over the arguments the Local Forecasting Models take, you will see:</p>
<ul>
<li>model_mode, trend_mode, use_trend</li>
<li>seasonal_period, seasonality_mode, decomposition_type, is_seasonal</li>
<li>output_chunk_length</li>
<li>lags_past_covariates, lags_future_covariates</li>
<li>add_encoders</li>
<li>and more..</li>
</ul>
<p>The script finds any trend and seasonality in the data, and then generates the appropriate arguments for each Local Forecasting Model. Through my extensive testing, I found that some of the arguments for some models don't accept 'lags=None', despite what the documentation states. I made adjustments as necessary for each model to set lags=1 when seasonality doesn't exist.
</p>
![All 26 LFM Results](https://github.com/markwkiehl/medium_darts_lfm/raw/0c5097c3b96cb9506078f048c3970073ebfda911/darts%2025%20dataset%20LFM%20plot%20(1).png)
![The best model among 26 LFM](https://github.com/markwkiehl/medium_darts_lfm/raw/0c5097c3b96cb9506078f048c3970073ebfda911/darts%2025%20dataset%20LFM%20plot%20(2).png)
![Results for 7 LFM that support multivariate series](https://github.com/markwkiehl/medium_darts_lfm/raw/0c5097c3b96cb9506078f048c3970073ebfda911/darts%207x%20LFM%20multivariate%20(1).png)

## Conclusions
<p>The script and functions work with univariate & multivariate series with a datetime or range/integer index, but multiple series are not supported. Past, future, and static covariates are supported. I hope that my extensive testing and development has resulted in something that other users can quickly utilize to quickly find the best Darts Local Forecasting Model for their data. If you have something that doesn't work well, please let me know and I would enjoy looking at it. 
</p>

## How Model Specific Arguments & Capabilities Are Defined
<p>The ability to fine tune arguments by model is possible because the capabilities and arguments for each are stored in a Python [NamedTuple](https://docs.python.org/3/library/collections.html#collections.namedtuple). The capabilities are for the most part what you see in the Darts Forecasting Models table. This data structure is also used to filter out models that are not applicable to the data being processed.</p>
```
# Define a data structure to hold the model objects, capabilities, & their specific arguments
class DartsModel(NamedTuple):
 model_name: str
 model_class: object
 args_dict: object
 univariate: bool = True
 multivariate: bool = False
 multiple_series: bool = False
 past_covariates: bool = False
 future_covariates: bool = False
 static_covariates: bool = False
 val_series: bool = False
 add_encoders: bool = True
 min_max_scale: bool = False
 ```

## Encoders & Feature Scaling
<p>Many models accept encoders via the 'add_encoders' argument. My script attempts to generate encoders automatically for each model that supports them based on the series index and any seasonality. 
Some models require or work better when the data is min/max scaled to a range of 0.0 to +1.0. This scaling is automatically applied based on the model, and it can be optionally set to apply to all of the models for the current data being processed.
</p>

## Data Splitting
<p>The script maximizes the length of the training series (train), and sets the prediction length (pred_steps or what Darts methods call 'n') to either three times the seasonal period, or 1/10 of the length of the source series. This can be adjusted, but considering that the goal is simply to find the best model among those applicable, it isn't real important. Optimization of the best model, or one close to the best in the ranking provided is up to the user.
</p>

## Visualization
<p>You can configure the script to show a plot of the forecast for each model, or a single figure with multiple plots for all models. You can also have the best model plotted.  
</p>
```raw.githubusercontent.com```
