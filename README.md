# Find The Best Darts Local Forecasting Model

A single Python script executes 26 of the Darts Local Forecasting Models and compares their accuracy. The seasonal and trend arguments required by the models are automatically derived and applied to each model.

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
<p>The script finds any trend and seasonality in the data, and then generates the appropriate arguments such as those listed above for each Local Forecasting Model. Through my extensive testing, I found that some of the arguments for some models don't accept 'lags=None', despite what the documentation states. I made adjustments as necessary for each model to set lags=1 when seasonality doesn't exist.
</p>
