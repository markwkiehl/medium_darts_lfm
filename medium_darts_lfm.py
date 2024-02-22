#
#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/



def df_has_nan_or_nat(df=None, results_by_col=False, verbose=False):
    # By default (results_by_col=False), returns True if any column in 
    # Pandas DataFrame has NaN / NaT.  
    # If results_by_col=True, returns column index, column name, True/False if has NaN/NaT, count of NaN/Nat

    import pandas as pd
    from darts import TimeSeries

    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series): raise Exception("Value Error: df must be a Pandas dataframe or series")

    if results_by_col==False:
        if verbose:
            # print the rows that have NaN/NaT
            print("Rows with NaN/NaT:")
            print(df[df.isnull().any(axis=1)])
        return df.isnull().sum().any()

    has_nan_nat = False
    results = []
    for idx, colname in enumerate(df.columns):
        ds = df.iloc[:,idx]
        #if ds.hasnans and verbose: print(idx, colname, ds.hasnans, ds.isnull().sum())
        if ds.hasnans: has_nan_nat = True
        results.append([idx, colname, ds.hasnans, ds.isnull().sum()])

    if verbose:
        # print the rows that have NaN/NaT
        print(df[df.isnull().any(axis=1)])

    return tuple(results)
    

def sec_to_dhms(s):
    d = s // (24 * 3600) 

    s = s % (24 * 3600) 
    h = s // 3600

    s %= 3600
    m = s // 60

    s %= 60

    long_str = str(d) + " days " + str(h) + " hours " + str(m) + " min " + str(round(s,1)) + " sec"
    
    short_str = ""
    if d > 0: short_str += str(d) + " days "
    if h > 0: short_str += str(h) + " hours "
    if m > 0: short_str += str(m) + " min "
    short_str += str(round(s,1)) + " sec"

    return short_str


def reset_matplotlib():
    # Reset the colors to default for v2.0 (category10 color palette used by Vega and d3 originally developed at Tableau)
    import matplotlib as mpl
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])


def get_subplots_nrows_ncols(nplots=None):
    # returns nrows, ncols for matplotlib subplot of 
    # multiple plots stacked column and row wise.
    # Maximum of 4 columns returned.
    # Some combinations yield empty plots.

    if nplots == 1: return 1, 1
    if nplots == 2: return 2, 1
    if nplots == 3: return 3, 1
    if nplots == 4: return 2, 2
    if nplots == 5: return 3, 2

    if nplots % 3 == 0:
        ncols = 3
        nrows = nplots // ncols
        if nrows < ncols: ncols, nrows = nrows, ncols
        if nrows < 8: return nrows, ncols
    
    if nplots % 2 == 0:
        ncols = 2
        nrows = nplots // ncols
        if nrows < ncols: ncols, nrows = nrows, ncols
        if nplots % 4 == 0 and nplots//4 < nrows: return nplots//4, 4
        if nrows < 8: return nrows, ncols
    
    if nplots % 4 == 0:
        ncols = 4
        nrows = nplots // ncols
        if nrows < ncols: ncols, nrows = nrows, ncols
        return nrows, ncols

    ncols = 4
    nrows = (nplots // ncols) + 1
    return nrows, ncols


def index_has_gaps(df=None, verbose=False):
    # Returns True if gaps are found in the index of dataframe df
    # Works with index of type datetime or numeric

    import pandas as pd
    import math

    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series): raise Exception("Argument 'df' passed to index_has_gaps() must be a Pandas dataframe or series")

    # USAGE:
    #from savvy_time_series import index_has_gaps
    #print("index_has_gaps() ", index_has_gaps(df=df, verbose=True))


    # Determine if the index is of type datetime (<class 'pandas.core.indexes.datetimes.DatetimeIndex'>), or numeric (<class 'pandas.core.indexes.range.RangeIndex'>)
    if isinstance(df.index, pd.DatetimeIndex):
        if verbose: print("Dataframe index is of type datetime")
    else:
        if verbose: print("Dataframe index is of type numeric")

    # Identify gaps in the index of df by creating a new dataframe
    # df_tmp with the difference of each ajoining index value. 
    #import pandas as pd
    df_tmp = df.index.to_series().diff().to_frame()
    #            Time
    #Time
    #0.000000       NaN
    #0.000083  0.000083

    # Drop the first value in the first column with the value of NaT (datetime) or NaN (numeric)
    df_tmp.dropna(inplace=True)
    #if verbose: print(df_tmp)
    #            Time
    #Time
    #0.000083  0.000083
    #0.000167  0.000083

    df_tmp.rename(columns={df_tmp.columns[0]: 'Diff'}, inplace=True, errors="raise")
    #df_tmp.rename(index={0: "Diff"}, inplace=True, errors="raise")  # KeyError: '[1] not found in axis'

    # Calculate the mode for column 'Diff' and get the maximum value as mode_max.
    #print(df_tmp['Diff'].mode())
    mode_median = df_tmp['Diff'].mode().median()
    mode_max = df_tmp['Diff'].mode().max()
    median = df_tmp['Diff'].median()
    if verbose: print("Col 'diff' has a mode().median() of: ", mode_median)    # 1 days 00:00:00
    if verbose: print("Col 'diff' has a mode().max() of: ", mode_max)    # 1 days 00:00:00
    if verbose: print("Col 'diff' has a median of: ", median)      # 1 days 00:00:00
    bool_mask = mode_max

    if isinstance(df.index, pd.DatetimeIndex):
        # Dataframe index of type datetime
        idx_freq_val, idx_freq_unit = get_df_index_freq_val_unit(df=df)
        if idx_freq_unit == "D" and idx_freq_val > 28:
            # monthly interval of ~ 31 days or 365.  This requires special processing.
            # Create a boolean mask based on column 'Diff' values and mode_median
            if verbose: print("Using bool_mask of mode_median of ", mode_median)
            bool_mask = mode_median

        # Create a boolean mask based on column 'Diff' values and bool_mask
        # in order find gaps.
        #if verbose: print(df_tmp['Diff'])
        df_tmp['bool_mask'] = df_tmp['Diff'].gt(bool_mask)
        #if verbose: print(df_tmp['bool_mask'].to_string())
        count_of_true = df_tmp.loc[df_tmp['bool_mask'] == True, 'bool_mask'].count()
        if verbose: print(str(count_of_true) + " gaps found in the index for threshold of " + str(bool_mask))
        del df_tmp
        if count_of_true > 0:
            return True
        else:
            return False


    else:
        # Dataframe index of type numeric
        if verbose: print("Col 'diff' has a min of: ", df_tmp['Diff'].min()) 
        if verbose: print("Col 'diff' has a max of: ", df_tmp['Diff'].max()) 
        df_tmp['delta'] = df_tmp['Diff'] - mode_median
        if verbose: print("Col 'delta' min: ", df_tmp['delta'].min())
        if verbose: print("Col 'delta' min: ", df_tmp['delta'].max())
        if verbose: print("Col 'delta' median: ", df_tmp['delta'].median())
        #import numpy as np
        #np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
        if df_tmp['delta'].median() == 0.0 or math.isclose(df_tmp['delta'].median(), 0.0):
            #print("median of col diff - mode_median values for the numeric index are nearly zero")
            del df_tmp
            return False
        bool_mask = mode_median
        df_tmp['bool_mask'] = df_tmp['Diff'].gt(mode_max)
        count_of_true = df_tmp.loc[df_tmp['bool_mask'] == True, 'bool_mask'].count()
        if verbose: print(str(count_of_true) + " gaps found in the index")
        if count_of_true > 0:
            return True
        else:
            return False
        
        

    # count occurrences of mode_max in column 'Diff'
    count_of_mode_max = df_tmp['Diff'].value_counts()[mode_max]
    count_of_gaps = df_tmp.shape[0]-count_of_mode_max
    # OR: if not count_of_mode_max == df_tmp.shape[0]:
    if count_of_gaps > 0:
        if verbose: print(str(df_tmp.shape[0]-count_of_mode_max) + " gaps found in the index")
        return True
    else:
        pass
        return False

    # count all values in name  with relative frequencies
    ds_rel_freq = df_tmp['Diff'].value_counts(normalize=True)
    #print(ds_rel_freq.to_string())
    #Diff
    #1 days    0.784756
    #3 days    0.179281
    #4 days    0.026838
    #...
    if len(ds_rel_freq) > 1:
        if verbose: print("ds_rel_freq has found gaps in the index")
        return True
    else:
        return False

    # Not the best method for a numeric index.  Use one of the prior methods.
    # Create a new column 'bool_mask' with the bool result of existance of mode_max in column 'Diff'
    if verbose: print("mode_max: ", mode_max)
    df_tmp['bool_mask'] = df_tmp['Diff'] == mode_max   
    # Count of values in column 'bool_mask' with value of False
    #print("Count of values in column 'bool_mask' with value of False: ", df_tmp.loc[df_tmp.bool_mask == False, 'bool_mask'].count())
    #print("Count of values in column 'bool_mask' with value of False: ", df_tmp['bool_mask'].value_counts()[False])
    count_of_false = df_tmp.loc[df_tmp.bool_mask == False, 'bool_mask'].count()
    count_of_false = df_tmp['bool_mask'].value_counts()[False]
    #print(df_tmp.to_string())
    if count_of_false > 0:
        if verbose: print(str(df_tmp.shape[0]-count_of_false) + " gaps found in the index")
        return True
    else:
        return False


def ts_is_multiple_series(ts=None):
    # Returns True if ts is a list of Darts TimeSeries.
    # Returns False if ts is not a list or if a list, false if any
    # of the items in the list is not a Darts TimeSeries.

    from darts import TimeSeries

    if not isinstance(ts, list): return False

    is_multiple_series = True
    for series in ts:
        if not isinstance(series, TimeSeries):
            is_multiple_series = False
            break
    return is_multiple_series


def ts_is_multivariate_series(ts=None):
    # Returns True if Darts TimeSeries ts is a multivariate series.
    # Returns False if ts is a univariate series, or a list of TimeSeries.
    
    from darts import TimeSeries

    if isinstance(ts, TimeSeries):
        if ts.width == 1:
            return False
        else:
            return True
    
    if ts_is_multiple_series(ts):
        return False
    else:
        raise Exception("ts is not a TimeSeries or a list of multiple TimeSeries")


def ts_is_multivariate_multiple_series(ts=None):
    # Returns True if Darts TimeSeries ts is a multivariate series
    # or a multiple series (list)

    from darts import TimeSeries

    if isinstance(ts, list):
        return True
    elif isinstance(ts, TimeSeries):
        if ts.width > 1:
            return True
        else:
            return False
    else:
        raise Exception("ts is not a Darts TimeSeries or a list of TimeSeries!")


def ts_trend_mode(ts=None, verbose=False):
    # Identify any tend as:
    #   model_mode: ModelMode.NONE, ModelMode.ADDITIVe,ModelMode.MULTIPLICATIVE
    #   trend_str:  "additive", "multiplicative"

    # If ts is a multiple series or multivariate series, then a list is returned
    # with the values for each series. 

    from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
    from darts.utils.statistics import stationarity_tests, remove_trend

    model_mode = ModelMode.NONE
    trend_str = None

    def get_trend(ts=None, verbose=False):
        # Returns model_mode, trend_str
        model_mode = ModelMode.NONE
        trend_str = "additive"
        if stationarity_tests(ts):
            #print("The series is stationary")
            pass
        else:
            # darts.utils.statistics.extract_trend_and_seasonality(ts, freq=None, model=ModelMode.MULTIPLICATIVE, method='naive', **kwargs)
            #print("The series is non-stationary.  Removing trend..")
            err = False
            try:
                ts_detrended = remove_trend(ts=ts, model=ModelMode.MULTIPLICATIVE)
            except Exception as e:
                if verbose: print("remove_trend() ERROR: " + repr(e) + " .. trying model=ModelMode.ADDITIVE ..")
                err = True
            if not err:
                model_mode = ModelMode.MULTIPLICATIVE
                trend_str = "multiplicative"
            else:
                err = False
                try:
                    ts_detrended = remove_trend(ts=ts, model=ModelMode.ADDITIVE)
                except Exception as e:
                    if verbose: print("remove_trend() ERROR: " + repr(e))
                    err = True
                if not err: 
                    model_mode = ModelMode.ADDITIVE
                    trend_str = "additive"
            if err:
                model_mode = ModelMode.NONE
                trend_str = "additive"
            if not err and not stationarity_tests(ts_detrended): 
                #raise Exception("Unable to remove the trend from series ts")
                model_mode = ModelMode.NONE
                trend_str = "additive"
        #if verbose: print("model_mode, trend_str:", model_mode, trend_str)
        return model_mode, trend_str

    if ts_is_multiple_series(ts) == True:
        model_mode = []
        trend_str = []
        for series in ts:
            mm, tstr = get_trend(series, verbose)
            model_mode.append(mm)
            trend_str.append(tstr)

    elif ts_is_multivariate_series(ts) == True:
        model_mode = []
        trend_str = []
        for i in range(0,ts.width):
            series = ts.univariate_component(i)
            mm, tstr = get_trend(series, verbose)
            model_mode.append(mm)
            trend_str.append(tstr)
            
    else:
        # ts is a univariate series
        model_mode, trend_str = get_trend(ts, verbose)
    
    return model_mode, trend_str


def ts_seasonality(ts=None, verbose=False):
    # Identify any seasonality period and modes for model initializatin purposes as:
    #   seasonal_period: int
    #   seasonality_mode:   SeasonalityMode.NONE, SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE
    #   decomposition_type:  "additive", "multiplicative"
    
    # If ts is a multiple series or a multivariate series, a list is returned for each with the
    # seasonal_period, seasonality_mode & decomposition_type for each series. 

    from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
    from darts.utils.statistics import stationarity_tests, remove_trend, remove_seasonality

    # Determine the seasonal periods of ts
    seasonal_periods = ts_seasonal_periods(ts=ts, sort_ascending=False, unique=False)
    if len(seasonal_periods) == 0:
        seasonal_period = None
    else:
        seasonal_period = int(seasonal_periods[0])

    def get_ts_seasonality_mode(ts=None, seasonal_period=None, verbose=verbose):
        # Determine the seasonality mode (SeasonalityMode.NONE, SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE).
        # ts cannot be a multivariate series
        
        seasonality_mode = SeasonalityMode.NONE
        if seasonal_period is None: return seasonality_mode

        err = False
        try:
            ts_non_seasonal = remove_seasonality(ts=ts, freq=seasonal_period, model=SeasonalityMode.MULTIPLICATIVE)
        except Exception as e:
            if verbose: print("remove_seasonality() ERROR: " + repr(e) + " .. trying SeasonalityMode.ADDITIVE ..")
            err = True
        if not err:
            seasonality_mode = SeasonalityMode.MULTIPLICATIVE
        else:
            err = False
            try:
                ts_non_seasonal = remove_seasonality(ts=ts, freq=seasonal_period, model=SeasonalityMode.ADDITIVE)
            except Exception as e:
                if verbose: print("remove_seasonality() ERROR: " + repr(e))
                err = True
            if not err: 
                seasonality_mode = SeasonalityMode.ADDITIVE

        return seasonality_mode


    if ts_is_multiple_series(ts) == True:
        seasonality_mode = []
        decomposition_type = []
        for series in ts:
            sm = get_ts_seasonality_mode(series, seasonal_period)
            seasonality_mode.append(sm)
            if sm == SeasonalityMode.ADDITIVE:
                decomposition_type = "additive"
            else:
                decomposition_type = "multiplicative"
    elif ts_is_multivariate_series(ts) == True:
        seasonality_mode = []
        for i in range(0,ts.width):
            series = ts.univariate_component(i)
            sm = get_ts_seasonality_mode(series, seasonal_period)
            seasonality_mode.append(sm)
            if sm == SeasonalityMode.ADDITIVE:
                decomposition_type = "additive"
            else:
                decomposition_type = "multiplicative"
    else:
        # ts is a univariate series
        seasonality_mode = get_ts_seasonality_mode(ts, seasonal_period)

        if seasonality_mode == SeasonalityMode.ADDITIVE:
            decomposition_type = "additive"
        else:
            decomposition_type = "multiplicative"


    return seasonal_period, seasonality_mode, decomposition_type



def get_darts_local_forecasting_models(ts=None, model_mode=None, trend_str=None, trend_mode=None, use_trend=None, is_stationary=None,
                                       seasonal_period=None, seasonality_mode=None, decomposition_type=None, is_seasonal=None,
                                       output_chunk_length=None,
                                       lags_past_covariates=None, lags_future_covariates=None, 
                                       categorical_past_covariates=None, categorical_future_covariates=None, categorical_static_covariates=None,
                                       add_encoders=None,
                                       verbose=False):
    
    # Returns a tuple of the Local Forecasting Models with model arguments related
    # to trend and seasonality defined, and filtered by those that support the 
    # type of series (ts) in terms of multiple and/or multivariate and univariate series.
    # Consult the table at:  https://github.com/unit8co/darts?tab=readme-ov-file#forecasting-models
    
    from darts import TimeSeries
    from typing import NamedTuple
    
    # Slow performance models with multivariate:  XGBModel, CatBoostModel, 
    from darts.models import (NaiveSeasonal,NaiveDrift,NaiveMovingAverage,NaiveEnsembleModel,ExponentialSmoothing,StatsForecastAutoCES,BATS,TBATS,Theta,FourTheta,StatsForecastAutoTheta,FFT,
                              AutoARIMA,StatsForecastAutoARIMA,StatsForecastAutoETS,Prophet,KalmanForecaster,RegressionModel,RandomForest,LightGBMModel,XGBModel,CatBoostModel,
                              Croston,LinearRegressionModel,
                            )

    if not isinstance(ts, TimeSeries): raise Exception("ts is not a Darts TimeSeries")

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

    # NOTE: models the require feature scaling: Logistic Regression, Support Vector Classifier, MLP Classifier, kNN Classifier
    #       models that do NOT require feature scaling:  Decision Tree, Random Forest, Gradient Boosting, Naive Bays
    # https://substack.com/app-link/post?publication_id=1119889&post_id=139393577&utm_source=post-email-title&utm_campaign=email-post-title&isFreemail=true&r=1ep5cg&token=eyJ1c2VyX2lkIjo4NTE1NDEyOCwicG9zdF9pZCI6MTM5MzkzNTc3LCJpYXQiOjE3MDE2MzA0ODYsImV4cCI6MTcwNDIyMjQ4NiwiaXNzIjoicHViLTExMTk4ODkiLCJzdWIiOiJwb3N0LXJlYWN0aW9uIn0.UOlLq917R5G2xxGKJIAdXZ3iA0nb9n-3sOmWv7yO9g0

    # seasonal_period is used for a lot of model defaults.  
    # seasonal_period cannot be None for model arguments such as lags, so in those cases set to 1 using dim_x.
    # output_chunk_length cannot be None.
    if seasonal_period is None:
        output_chunk_length = len(ts)//10
        dim_x = 1
    else:
        output_chunk_length = seasonal_period
        dim_x = seasonal_period

    m_autoarima = DartsModel(model_name='AutoARIMA', model_class=AutoARIMA, args_dict={'seasonal': is_seasonal, 'm': dim_x, 'stationary': is_stationary, 'add_encoders':add_encoders},
                            future_covariates=True)
    
    m_stats_forecast_auto_arima = DartsModel('StatsForecastAutoARIMA', StatsForecastAutoARIMA, {'season_length': dim_x},
                            future_covariates=True, add_encoders=False, min_max_scale=True)
    # add_encoders not supported by StatsForecastAutoARIMA    
    
    m_stats_forecast_auto_ets = DartsModel('StatsForecastAutoETS', StatsForecastAutoETS, {'season_length': dim_x, 'model': "ZZZ", 'add_encoders':add_encoders},
                            future_covariates=True)
    # StatsForecast ETS (Error, Trend, Seasonality): M multiplicative, A additive, Z optimized or N ommited components
    #   model = 'Z' 
    #   model='ZZZ' (ask AutoETS model to figure out the best parameters)
    #   model=‘ANN’ (additive error, no trend, and no seasonality)
    #   model='ZMZ' (multiplicative trend, optimal error and seasonality)
    #   ‘Z’ operates as a placeholder to ask the AutoETS model to figure out the best parameter.
    
    # ValueError: Prophet does not support integer range index. The index of the TimeSeries must be of type pandas.DatetimeIndex
    m_prophet = DartsModel('Prophet', Prophet, {'add_seasonalities':{'name':"daily_seasonality",'seasonal_periods':dim_x,'fourier_order':5, 'mode':trend_str}},
                           add_encoders=False)
    
    m_kalman_forecaster = DartsModel('KalmanForecaster', KalmanForecaster, {'dim_x': dim_x, 'add_encoders':add_encoders})
    
    # Croston best for demand forecasting and intermediate time series.
    # I didn't get anything useful out of the Croston models.
    m_croston_optimized = DartsModel('Croston', Croston, {'version':"optimized", 'add_encoders':add_encoders}, future_covariates=True)
    m_croston_classic = DartsModel('Croston', Croston, {'version':"classic", 'add_encoders':add_encoders}, future_covariates=True)
    m_croston_sba = DartsModel('Croston', Croston, {'version':"sba", 'add_encoders':add_encoders}, future_covariates=True)
    m_croston_tsb = DartsModel('Croston', Croston, {'version':"tsb", 'alpha_d':0.2, 'alpha_p':0.2, 'add_encoders':add_encoders}, future_covariates=True)

    m_regression = DartsModel('RegressionModel', RegressionModel, {'lags': dim_x, 'output_chunk_length': output_chunk_length, 'lags_past_covariates':lags_past_covariates, 'lags_future_covariates':lags_future_covariates, 'add_encoders':add_encoders},
                              multivariate=True, multiple_series=True, past_covariates=True, future_covariates=True, static_covariates=True)

    m_linear_regression = DartsModel('LinearRegressionModel', LinearRegressionModel, {'lags': dim_x, 'output_chunk_length': output_chunk_length, 'lags_past_covariates':lags_past_covariates, 'lags_future_covariates':lags_future_covariates, 'add_encoders':add_encoders},
                              multivariate=True, multiple_series=True, past_covariates=True, future_covariates=True, static_covariates=True)
    
    m_random_forest = DartsModel('RandomForest', RandomForest, {'lags': dim_x, 'output_chunk_length': output_chunk_length, 'lags_past_covariates':lags_past_covariates, 'lags_future_covariates':lags_future_covariates, 'add_encoders':add_encoders},
                              multivariate=True, multiple_series=True, past_covariates=True, future_covariates=True, static_covariates=True)

    m_light_gbm_model = DartsModel('LightGBMModel', LightGBMModel, {'lags': dim_x, 'output_chunk_length': output_chunk_length, 'lags_past_covariates':lags_past_covariates, 'lags_future_covariates':lags_future_covariates, 'categorical_past_covariates':categorical_past_covariates, 'categorical_future_covariates':categorical_future_covariates, 'categorical_static_covariates':categorical_static_covariates, 'add_encoders':add_encoders, 'verbose':-1},
                              multivariate=True, multiple_series=True, past_covariates=True, future_covariates=True, static_covariates=True)

    m_xgboost = DartsModel('XGBModel', XGBModel, {'lags': dim_x, 'output_chunk_length': output_chunk_length, 'lags_past_covariates':lags_past_covariates, 'lags_future_covariates':lags_future_covariates, 'add_encoders':add_encoders},
                              multivariate=True, multiple_series=True, past_covariates=True, future_covariates=True, static_covariates=True)

    m_catboostmodel = DartsModel('CatBoostModel', CatBoostModel, {'lags': dim_x, 'output_chunk_length': output_chunk_length, 'lags_past_covariates':lags_past_covariates, 'lags_future_covariates': lags_future_covariates, 'add_encoders':add_encoders},
                              multivariate=True, multiple_series=True, past_covariates=True, future_covariates=True, static_covariates=True)

    m_stats_forecast_auto_ces = DartsModel('StatsForecastAutoCES', StatsForecastAutoCES, {'season_length': dim_x},
                                           add_encoders=False,
                                           min_max_scale=False)     # Only slightly better with min_max_scale=True

    m_exponentional_smoothing = DartsModel('ExponentialSmoothing', ExponentialSmoothing, {'trend':model_mode, 'seasonal':seasonality_mode, 'seasonal_periods': seasonal_period},
                                           add_encoders=False)

    m_bats = DartsModel('BATS', BATS, {'use_trend':model_mode, 'seasonal_periods': [dim_x]},
                        add_encoders=False)
    
    m_tbats = DartsModel('TBATS', TBATS, {'use_trend':use_trend, 'seasonal_periods': [dim_x]},
                         add_encoders=False)

    m_theta = DartsModel('Theta', Theta, {'season_mode':seasonality_mode, 'seasonality_period':seasonal_period},
                         add_encoders=False)

    m_four_theta = DartsModel('FourTheta', FourTheta, {'season_mode':seasonality_mode, 'seasonality_period':seasonal_period, 'model_mode':model_mode, 'trend_mode':trend_mode},
                              add_encoders=False)

    m_stats_forecast_auto_theta = DartsModel('StatsForecastAutoTheta', StatsForecastAutoTheta, {'season_length': dim_x, 'decomposition_type':decomposition_type},
                                             add_encoders=False)

    m_fft_none = DartsModel('FFT', FFT, {'trend':None}, add_encoders=False)      # 'required_matches':{'day'}, 
    m_fft_poly = DartsModel('FFT', FFT, {'trend':'poly'}, add_encoders=False)      # 'required_matches':{'day'}, 

    m_naive_seasonal = DartsModel('NaiveMovingAverage', NaiveMovingAverage, {'input_chunk_length':output_chunk_length}, 
                                  multivariate=True, add_encoders=False)

    m_naive_seasonal = DartsModel('NaiveSeasonal', NaiveSeasonal, {'K':dim_x}, 
                                  multivariate=True, add_encoders=False)
    
    # Future Development:
    #m_naive_drift = DartsModel('NaiveDrift', NaiveDrift, {}, multivariate=True)
    #m_naive_ensemble = DartsModel('NaiveEnsembleModel', NaiveEnsembleModel, {[m_naive_seasonal,m_naive_drift]}, multivariate=True)
    # NaiveEnsembleModel    something to develop later

    # Build a tuple of the Local Forecasting Models to process
    darts_models = (m_random_forest, m_catboostmodel, m_xgboost, m_light_gbm_model, m_linear_regression, m_regression, m_croston_optimized, m_croston_classic, m_croston_sba, m_croston_tsb, m_kalman_forecaster, m_prophet, m_stats_forecast_auto_ets, m_stats_forecast_auto_arima, m_autoarima, m_stats_forecast_auto_ces, m_exponentional_smoothing, m_bats, m_tbats, m_theta, m_four_theta, m_stats_forecast_auto_theta, m_stats_forecast_auto_theta, m_fft_none, m_fft_poly, m_naive_seasonal) # Tuple needs at least two values for later iteration
    #darts_models = (m_linear_regression, m_regression)

    # Remove models from darts_models that don't support multivariate series and multiple series 
    dms = []
    for darts_model in darts_models:
        if ts.width > 1:
            if darts_model.multivariate == True: dms.append(darts_model)
        elif isinstance(ts, list):
            if darts_model.multiple_series == True: dms.append(darts_model)
        else:
            if ts.has_datetime_index:
                if verbose: "Adding " + darts_model.model_name + " because it DOES support a datetime index"
                dms.append(darts_model)
            else:
                # range/int index
                if darts_model.model_name == 'Prophet':
                        # Prophet does not support integer range index. The index of the TimeSeries must be of type pandas.DatetimeIndex
                        print("Not adding model '" + darts_model.model_name + "' because it doesn't support a range/int index")
                else:
                    dms.append(darts_model)
    if not len(darts_models) == len(dms): 
        print("Filtered models from " + str(len(darts_models)) + " to " + str(len(dms)) + " based on models that support multiple and multivariate series.")
        darts_models = tuple(dms)
        print("")
    del dms
    return darts_models


def ts_encoder_min_max_scale_datetime_idx(ts):
    # Extract the year each time index entry and normalize it.
    # add_encoders are supported by nearly every Darts Local Forecasting Model.
    # They are an easy and inexpensive way of adding past and/or future
    # covariate series data to a model by encoding the TimeSeries index.
    # https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html
    if ts.time_index.max().year - ts.time_index.min().year > 1:
        range_val = ts.time_index.max().year - ts.time_index.min().year
        if range_val < 1:
            return None
        else:
            return (ts.time_index.year - ts.time_index.min().year) / range_val
    elif ts.time_index.max().month - ts.time_index.min().month > 1:
        range_val = ts.time_index.max().month - ts.time_index.min().month
        if range_val < 1:
            return None
        else:
            return (ts.time_index.month - ts.time_index.min().month) / range_val
    elif ts.time_index.max().day - ts.time_index.min().day > 1:
        range_val = ts.time_index.max().day - ts.time_index.min().day
        if range_val < 1:
            return None
        else:
            return (ts.time_index.day - ts.time_index.min().day) / range_val
    elif ts.time_index.max().hour - ts.time_index.min().hour > 1:
        range_val = ts.time_index.max().hour - ts.time_index.min().hour
        if range_val < 1:
            return None
        else:
            return (ts.time_index.hour - ts.time_index.min().hour) / range_val
    elif ts.time_index.max().minute - ts.time_index.min().minute > 1:
        range_val = ts.time_index.max().minute - ts.time_index.min().minute
        if range_val < 1:
            return None
        else:
            return (ts.time_index.minute - ts.time_index.min().minute) / range_val
    elif ts.time_index.max().second - ts.time_index.min().second > 1:
        range_val = ts.time_index.max().second - ts.time_index.min().second
        if range_val < 1:
            return None
        else:
            return (ts.time_index.second - ts.time_index.min().second) / range_val
    # Also could do microsecond, nanosecond
    else:
        return None
    if range_val < 1:
        return None
    else:
        return (ts.time_index.year - ts.time_index.min().year) / range_val

def ts_encoder_min_max_scale_range_idx(ts):
    # Extract the year each time index entry and normalize it.
    # add_encoders are supported by nearly every Darts Local Forecasting Model.
    # They are an easy and inexpensive way of adding past and/or future
    # covariate series data to a model by encoding the TimeSeries index.
    range_val = ts.time_index.max() - ts.time_index.min()
    if range_val <= 1:
        return None
    else:
        return (ts.time_index - ts.time_index.min()) / range_val

def ts_encoder_sine(ts=None, seasonal_period=None):
    #import darts.utils.timeseries_generation as tg
    # add_encoders are supported by nearly every Darts Local Forecasting Model.
    # They are an easy and inexpensive way of adding past and/or future
    # covariate series data to a model by encoding the TimeSeries index.
    seasonal_sine = None
    if not seasonal_period is None: 
        seasonal_sine = tg.sine_timeseries(length=len(ts), value_frequency=1/seasonal_period, value_amplitude=1.0, start=ts.time_index.min()) 
        return seasonal_sine.values
    else:
        return seasonal_sine


def try_all_darts_lfm(ts=None, past_covariates=None, future_covariates=None, min_max_scale=False, plot_each_model=False, plot_all_models=False, plot_best_model=True, verbose=True):
    # Fit all Darts Local Forecasting Models (LFM) to the series 'ts' and compile a table
    # of the results by the Root Mean Square Error (RMSE), and include the execution time. 
    
    # Any trend and seasonality is deteted, and then the appropriate arguments for them
    # are passed to each Local Forecasting Model. 
    
    # Defines encoders for the model 'add_encoders' argument when supported. 
    
    # Works with univariate & multivariate series with datetime or range/integer index.
    # Does not handle multiple series.  
    
    # Min/Max scales (0.0 to +1.0) as appropriate by model, but can be forced on by setting
    # argument min_max_scale=True
    
    # Automatically splits the model, defining the prediction duration (n) as either 1/10 of
    # the source series length if index is type range/int or no seasonal period exists, or 
    # 3x the seasonal period.  The remainder of the source series is allocated to training
    # and established to support past/future covariates. 

    # plot_each_model=True will pause execution of each model and display the results in a plot.

    # plot_all_models=True will create a single plot at the end of the results for all models.

    # plot_best_model=True will generate a single plot at the end of the best model by metric (lowest RMSE).

    from darts import TimeSeries, concatenate
    from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
    from darts.dataprocessing.transformers.scaler import Scaler
    from darts.utils.statistics import stationarity_tests, remove_trend, remove_seasonality
    import darts.utils.timeseries_generation as tg
    from darts.metrics import rmse
    import torch

    from savvy_time_series import ts_seasonal_periods, ts_seasonality, ts_trend_mode, reset_matplotlib, get_subplots_nrows_ncols
    from savvy_time_series import get_idx_datetime_attr, sec_to_dhms, get_darts_local_forecasting_models
    import time
    from pathlib import Path
    from operator import itemgetter
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)
    from time_series_data_sets import sine_gaussian_noise_covariate

    # for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    reset_matplotlib()
    scaler = Scaler()  # default uses sklearn's MinMaxScaler. 

    if not isinstance(ts, TimeSeries): raise Exception("Value Error: ts is not a Darts TimeSeries")

    def plt_fcst(ts=None, pred=None, train=None, title=""):
        if ts.width == 1:
            fig, ax = plt.subplots(2, figsize=(10,6))
            fig.suptitle(title)
            ts.plot(ax=ax[0], label="ground truth", linestyle='--', color='tab:gray')
            pred.plot(ax=ax[0], label="pred")
            if not train is None: train.plot(ax=ax[0], label="train")
            ts.slice_intersect(pred).plot(ax=ax[1], label="ground truth", linestyle='--', color='tab:gray')
            pred.plot(ax=ax[1], label="pred")   # , marker='o'
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(ts.width, figsize=(10,6), sharex=True)
            fig.suptitle(title)
            for i in range(0, ts.width):
                gnd_truth = ts.slice_intersect(pred).univariate_component(i)
                fcst = pred.univariate_component(i)
                gnd_truth.plot(ax=ax[i], label="ground truth", linestyle='--', color='tab:gray')
                fcst.plot(ax=ax[i], label="pred")       # , marker='o'
            plt.tight_layout()
            plt.show()

    if verbose and not past_covariates is None: print("cv_past:", past_covariates.time_index.min(), " ..", past_covariates.time_index.max(), "\tlength: ", len(past_covariates.time_index))  
    if verbose and not future_covariates is None: print("cv_future:", future_covariates.time_index.min(), " ..", future_covariates.time_index.max(), "\tlength: ", len(future_covariates.time_index))  

    if not isinstance(plot_each_model, bool): raise Exception("Value Error: plot_each_model must be bool")
    if not isinstance(plot_all_models, bool): raise Exception("Value Error: plot_all_models must be bool")
    if not isinstance(plot_best_model, bool): raise Exception("Value Error: plot_best_model must be bool")
    if plot_each_model == True and plot_all_models == True: raise Exception("Cannot plot each model and then all models, choose one option only")

    # Identify any tend
    trend_str = None
    model_mode, trend_str = ts_trend_mode(ts, verbose=False)
    # ??? How to determine if TrendMode.LINEAR or TrendMode.EXPONENTIAL ???   Maybe getting the slope from NaiveDrift ?
    trend_mode = TrendMode.LINEAR
    if isinstance(model_mode, list):
        # multivariate or multiple series, not applicable as models that support multivariate series don't require trend/stationary input.
        use_trend = False
        is_stationary = False
    else:
        # Univariate series
        if model_mode == ModelMode.NONE:
            use_trend = False
            is_stationary = True
        else:
            use_trend = True
            is_stationary = False
    if verbose:
        print("")
        if use_trend == False:
            print("No trend")
        else:
            if isinstance(model_mode, list):
                print("Has trend")
            else:
                print(str(model_mode).split('.')[1].lower() + " trend")
        print("")
    #if verbose and not isinstance(model_mode, list): 
        #print("model_mode:", model_mode)
        #print("trend_str:", trend_str)
    #if verbose:
        #print("trend_mode:", trend_mode)
        #print("is_stationary:", is_stationary)
        #print("use_trend:", use_trend, "\n")
    # Determine the seasonal periods of ts

    seasonal_periods = ts_seasonal_periods(ts=ts, sort_ascending=False, unique=False)
    if len(seasonal_periods) == 0:
        seasonal_period = None
        is_seasonal = False
    else:
        seasonal_period = int(seasonal_periods[0])
        is_seasonal = True
    from savvy_time_series import ts_seasonality
    seasonal_period, seasonality_mode, decomposition_type = ts_seasonality(ts)
    if verbose:
        if is_seasonal == False:
            print("No seasonality\n")
        else:
            if isinstance(seasonality_mode,list):
                print("Seasonality detected (multivariate series)\n")
            else:
                print(str(seasonality_mode) + " with a period of " + str(seasonal_period), "\n")
        #print("seasonal_periods:",seasonal_periods)
        #print("seasonal_period:", seasonal_period)
        #print("seasonality_mode:", seasonality_mode)
        #print("decomposition_type:", decomposition_type)
        #print("is_seasonal:", is_seasonal, "\n")

    # Split the model into train & test
    # Note: most LFMs do not accept val_series & pred_input
    if seasonal_period is None:
        pred_steps = int(len(ts)//10)
        train = ts[:len(ts)-pred_steps-len(ts)//10-1]
    else:
        pred_steps = int(seasonal_period*3 + 1)
        train = ts[:len(ts)-pred_steps-seasonal_period-1]
    if verbose: print("train:", train.time_index.min(), " ..", train.time_index.max(), "\tlength: ", len(train.time_index))  
    del train       # defined later for each model
    if verbose: print("pred_steps:", pred_steps)

    if seasonal_period is None:
        output_chunk_length = 1
    else:
        output_chunk_length = seasonal_period//2

    lags_past_covariates = None     # = (0,seasonal_period)
    lags_future_covariates = None
    if not seasonal_period is None:
        if not past_covariates is None: 
            lags_past_covariates = list(range(-1,(seasonal_period*-1)-1,-1))      # only values <= -1 if in a list    
            print("lags_past_covariates:", lags_past_covariates)
        if not future_covariates is None: 
            lags_future_covariates = list(range(0,seasonal_period+1))
            print("lags_future_covariates:", lags_future_covariates)

    # categorical features (not supported yet by Darts)
    # https://github.com/unit8co/darts/issues/1514
    categorical_past_covariates = None
    categorical_future_covariates = None
    categorical_static_covariates = None

    # add_encoders are supported by nearly every Darts Local Forecasting Model.
    # They are an easy and inexpensive way of adding past and/or future
    # covariate series data to a model by encoding the TimeSeries index.
    add_encoders = None
    # Define add_encoders
    #print("ts.freq:", ts.freq)     # ts.freq: <Day> datetime or range
    #print("freq_str:", ts.freq_str)     # freq_str: D
    #print("get_idx_datetime_attr(" + ts.freq_str + "):", get_idx_datetime_attr(ts.freq_str))    # get_idx_datetime_attr(D): day
    #print("ts.time_dim:", ts.time_dim)      # ts.time_dim: Date
    if ts.has_datetime_index:
        add_encoders={
            'cyclic': {'future': [get_idx_datetime_attr(ts.freq_str)]},
            'datetime_attribute': {'future': [get_idx_datetime_attr(ts.freq_str)]},
            'position': {'past': ['relative'], 'future': ['relative']},
            #'custom': {'past': [ts_encoder_sine(ts, seasonal_period)]},
            'transformer': Scaler()
        }
        add_encoders['custom'] = {'past': [ts_encoder_min_max_scale_datetime_idx(ts)]}
    else:
        # range / int index
        add_encoders={
            #'cyclic': {'future': [get_idx_datetime_attr(ts.freq_str)]},
            #'datetime_attribute': {'future': [get_idx_datetime_attr(ts.freq_str)]},
            'position': {'past': ['relative'], 'future': ['relative']},
            'transformer': Scaler()
        }
        if not ts.freq_str is None: add_encoders['cyclic'] = {'future': [get_idx_datetime_attr(ts.freq_str)]}
        if ts.time_index.max() > ts.time_index.min():
            add_encoders['custom'] = {'past': [ts_encoder_min_max_scale_range_idx(ts)]}

    #add_encoders = None
    print("add_encoders:", add_encoders)
    print("")

    # CONFIGURE MODEL, TRAIN IT, RUN PREDICTION

    darts_models = get_darts_local_forecasting_models(ts, model_mode, trend_str, trend_mode, use_trend, is_stationary,
                                    seasonal_period, seasonality_mode, decomposition_type, is_seasonal,
                                    output_chunk_length,
                                    lags_past_covariates, lags_future_covariates, 
                                    categorical_past_covariates, categorical_future_covariates, categorical_static_covariates,
                                    add_encoders,
                                    verbose=True)

    if plot_all_models == True:
        nrows, ncols = get_subplots_nrows_ncols(nplots=len(darts_models))
        fig, ax = plt.subplots(nrows, ncols, figsize=(10,6))
        fig.suptitle(str(len(darts_models)) + " Darts Local Forecasting Models")

    # Configure, Train, Fit, Predict all models in darts_models..
    print("Configure, train, fit, predict " + str(len(darts_models)) + " models:")
    results = []
    model_idx = 0
    i = 0
    c = 0
    for darts_model in darts_models:

        t_start =  time.perf_counter()
        #print("model: ", darts_model.model_name)

        model = darts_model.model_class(**darts_model.args_dict)        # The ** gets the dictionary and converts it to keyword arguments

        if seasonal_period is None:
            train = ts[:len(ts)-pred_steps-len(ts)//10-1]
        else:
            train = ts[:len(ts)-pred_steps-seasonal_period-1]
        if darts_model.min_max_scale == True or min_max_scale == True:
            #print("Scaling series 'train' 0.0 to +1.0")
            train = scaler.fit_transform(train)
            # for other series, use series=scaler.transform(series)
            # unscale: series=scaler.inverse_transform(series)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)

        # Train and run prediction based on the model's capabilities as defined by darts_model.
        if darts_model.past_covariates == False and darts_model.future_covariates == False:
            model.fit(series=train)
            pred = model.predict(n=pred_steps)
        elif darts_model.past_covariates == True and darts_model.future_covariates == False:
            model.fit(series=train, past_covariates=past_covariates)
            pred = model.predict(n=pred_steps, past_covariates=past_covariates)
        elif darts_model.past_covariates == False and darts_model.future_covariates == True:
            model.fit(series=train, future_covariates=future_covariates)
            pred = model.predict(n=pred_steps, future_covariates=future_covariates)
        else:
            # The model supports both past and future covariates
            model.fit(series=train, past_covariates=past_covariates, future_covariates=future_covariates)
            pred = model.predict(n=pred_steps, past_covariates=past_covariates, future_covariates=future_covariates)

        if darts_model.min_max_scale == True or min_max_scale == True: pred=scaler.inverse_transform(pred)
        metric_rmse = round(rmse(ts.slice_intersect(pred), pred),5)
        print("Model", model_idx+1, " of ", len(darts_models), darts_model.model_name, "\t\t\tRMSE:", round(metric_rmse,5), "\t", round(time.perf_counter()-t_start,2), "sec", "\t", sec_to_dhms(time.perf_counter()-t_start))
        results.append([round(metric_rmse,5), darts_model.model_name, model_idx, round(time.perf_counter()-t_start,2), str(darts_model.args_dict)])
        
        if plot_all_models:
            title = darts_model.model_name + " RMSE: " + str(round(metric_rmse,5))
            ts.slice_intersect(pred).plot(ax=ax[i,c], linewidth=1.0)
            pred.plot(ax=ax[i,c], linewidth=1.0)
            ax[i,c].set_title(title, fontsize=6)
            ax[i,c].xaxis.set_label_text('')
            ax[i,c].tick_params(axis="x", rotation=25, labelsize=6)
            ax[i,c].tick_params(axis="y",labelsize=6)
            #ax[i,c].legend(fontsize=6)
            ax[i,c].get_legend().remove()
            c += 1
            if c >= ncols:
                c = 0
                i += 1

        model_idx += 1

        if plot_each_model:
            # Plot each model
            title = darts_model.model_name + " RMSE: " + str(round(metric_rmse,5))
            plt_fcst(ts, pred, ts.slice_intersect(train), title)
        
        #break

    if plot_all_models:
        plt.tight_layout()
        plt.show()

    # Print out the results sorted by lowest RMSE first
    print("\n")
    results = sorted(results, key=itemgetter(0))
    template = "{0:>20}|{1:>25}|{2:<10}|{3:>40}"
    print(template.format("RMSE", "model", "model_idx", "execution time")) # header
    for result in results:
        print(template.format(result[0],result[1],result[2],sec_to_dhms(result[3])))

    # Print out the model arguments
    print("\n")
    template = "{0:<10}|{1:<50}"
    print(template.format("model","arguments")) # header
    for result in results:
        print(template.format(result[1],result[4]))
        print("")

    if plot_best_model == True:
        # Plot the best results
        model_idx = results[0][2]
        darts_model = darts_models[model_idx]
        model = darts_model.model_class(**darts_model.args_dict)

        if seasonal_period is None:
            train = ts[:len(ts)-pred_steps-len(ts)//10-1]
        else:
            train = ts[:len(ts)-pred_steps-seasonal_period-1]
        if darts_model.min_max_scale == True or min_max_scale == True:
            #print("Scaling series 'train' 0.0 to +1.0")
            #scaler = Scaler()  # default uses sklearn's MinMaxScaler. 
            train = scaler.fit_transform(train)
            # for other series, use series=scaler.transform(series)
            # unscale: series=scaler.inverse_transform(series)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)

        # Train and run prediction based on the model's capabilities as defined by darts_model.
        if darts_model.past_covariates == False and darts_model.future_covariates == False:
            model.fit(series=train)
            pred = model.predict(n=pred_steps)
        elif darts_model.past_covariates == True and darts_model.future_covariates == False:
            model.fit(series=train, past_covariates=past_covariates)
            pred = model.predict(n=pred_steps, past_covariates=past_covariates)
        elif darts_model.past_covariates == False and darts_model.future_covariates == True:
            model.fit(series=train, future_covariates=future_covariates)
            pred = model.predict(n=pred_steps, future_covariates=future_covariates)
        else:
            # The model supports both past and future covariates
            model.fit(series=train, past_covariates=past_covariates, future_covariates=future_covariates)
            pred = model.predict(n=pred_steps, past_covariates=past_covariates, future_covariates=future_covariates)
        
        if darts_model.min_max_scale == True or min_max_scale == True: pred=scaler.inverse_transform(pred)
        metric_rmse = rmse(ts.slice_intersect(pred), pred)
        title = darts_model.model_name + " RMSE: " + str(round(metric_rmse,5))
        plt_fcst(ts, pred, ts.slice_intersect(train), title)



def get_idx_datetime_attr(freq_str=None):
    # Maps the Darts / Pandas datetime index 
    #   Darts: .freq string '<Day>' and .freq_str 'D' strings 
    #   Pandas: .freq and .freqstr strings
    # to the Darts darts.utils.timeseries_generation.datetime_attribute_timeseries 'attribute' argument corresponds
    # to the Pandas DatetimeIndex string attributes, e.g. “month”, “weekday”, “day”, “hour”, “minute”, “second”.
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-period-aliases
    # Aliases A, H, T, S, L, U, and N are deprecated in favour of the aliases Y, h, min, s, ms, us, and ns
    
    # index .freq       index .freq_str         'attribute' argument for datetime_attribute_timeseries
    #                                           'year'
    #                       'MS                 'month'
    #   <Day>               'D'                 'day'
    #                       'H'                 'hour'
    #                       'T'                 'minute'
    #                       'S'                 'second'
    #                       'U'                 'microsecond'
    #                       'L'                 'nanosecond'

    if freq_str is None: raise Exception("freq_str is None!")

    freqstr_attr = {
        'D': 'day',
        'H': 'hour',
        'bh': 'hour',
        'cbh': 'hour',
        'T': 'minute',
        'min': 'minute',
        'S': 'second',
        'U': 'microsecond',
        'L': 'nanosecond',
        'MS': 'month',    
        'ME': 'month',
        'BME': 'month',
        'CBME': 'month',
        'BMS': 'month',
        'QE': 'quarter',
        'BQE': 'quarter',
        'QS': 'quarter',
        'BQS': 'quarter',
        'YE': 'year',
        'BYE': 'year',
        'YS': 'year',
        'BYS': 'year',
    }
    
    if freq_str[0:3] == 'QS-' and not freq_str in freqstr_attr: freqstr_attr[freq_str] = 'quarter'
    if freq_str[0:2] == 'W-' and not freq_str in freqstr_attr: freqstr_attr[freq_str] = 'week'
    if freq_str[len(freq_str)-1] == 'T' and not freq_str in freqstr_attr: freqstr_attr[freq_str] = 'minute'     # '10T' -> 'T' minute

    if not freq_str in freqstr_attr: raise Exception("ERROR: freq_str '" + freq_str + "' passed to get_idx_datetime_attr() is not recognized.")
    return freqstr_attr[freq_str]


def get_df_index_freq_val_unit(df=None, verbose=False):
    # Note that Pandas df.index.freqstr will return the freq string.

    # Returns the index frequency value and unit.
    # idx_freq_val, idx_freq_unit = get_df_index_freq_val_unit(df)

    # Determines (measures) the index freq and returns the value
    # along with the unit.  For a numeric index, the unit returned will
    # be None, otherwise for datetime the unit will be:
    # Offset Alias  Description
    #   D               day
    #   H               hour
    #   T               minute
    #   S               second
    #   L               millisecond
    #   U               microsecond
    #   N               nanosecond
    #  None a numeric index (not of type datetime)
    
    # Note that the Offset Alias is a TimeDelta resolution_string.
    #   https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    #   https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.freqstr.html
    #   https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects

    import pandas as pd
    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series): raise Exception("Argument 'df' passed to get_df_index_freq_val_unit() must be a Pandas dataframe or series")

    # Measure index spacing by creating a new dataframe
    # df_tmp with the difference of each ajoining index value. 
    df_tmp = df.index.to_series().diff().to_frame()

    # Drop the first row in the first column with the value of NaT (datetime) or NaN (numeric)
    df_tmp.dropna(inplace=True)

    df_tmp.rename(columns={df_tmp.columns[0]: 'Diff'}, inplace=True, errors="raise")

    # Calculate the mode for column 'Diff' and get the maximum value as mode_max.
    if verbose: print("Col 'diff' has a mode().median() of: ", df_tmp['Diff'].mode().median())    # 1 days 00:00:00
    if verbose: print("Col 'diff' has a median of: ", df_tmp['Diff'].median())      # 1 days 00:00:00

    mode = df_tmp['Diff'].mode().median()
    # Note that the mode is of type TimeDelta.  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html#pandas-timedelta
    #print("type(mode): ", type(mode))   # <class 'pandas._libs.tslibs.timedeltas.Timedelta'>

    # Determine if the index is of type datetime (<class 'pandas.core.indexes.datetimes.DatetimeIndex'>), or numeric (<class 'pandas.core.indexes.range.RangeIndex'>)
    if isinstance(df.index, pd.DatetimeIndex):
        if verbose: print("Dataframe index is of type datetime")
    else:
        if verbose: print("Dataframe index is of type numeric")
        return mode, ""


    #print(mode.components)  # Components(days=31, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0, nanoseconds=0)
    #print(mode.components.days) # 31
    #print(mode.resolution_string)   # D
    # NOTE: The maximum resolution of mode.resolution_string is D
    if mode.resolution_string == "D": return mode.days, mode.resolution_string  # days
    elif mode.resolution_string == "H": return mode.components.hours, mode.resolution_string  # hours
    elif mode.resolution_string == "T": return mode.components.minutes, mode.resolution_string  # minutes
    elif mode.resolution_string == "S": return mode.seconds, mode.resolution_string  # seconds
    elif mode.resolution_string == "L": return mode.components.milliseconds, mode.resolution_string  # milliseconds
    elif mode.resolution_string == "U": return mode.microseconds, mode.resolution_string  # microseconds
    elif mode.resolution_string == "N": return mode.nanoseconds, mode.resolution_string  # nanoseconds
    else:
        raise Exception("ERROR: Unexpected .resolution_string of '" + mode.resolution_string + "'")





if __name__ == '__main__':
    pass

    # ---------------------------------------------------------------------------
    # Model nearly all Darts Local Forecasting Models (LFM) against various Darts datasets

    from darts import TimeSeries

    from pathlib import Path
    import matplotlib.pyplot as plt
    #import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)

    # for reproducibility
    np.random.seed(1)

    # Set min_max_scale = True to min/max scale the data to range of 0.0 to +1.0
    # WARNING:  Only use this to override the model specific default setting.
    min_max_scale = False

    from darts.datasets import (
        AirPassengersDataset,
        AusBeerDataset,
        GasRateCO2Dataset,
        HeartRateDataset,
        IceCreamHeaterDataset,
        MonthlyMilkDataset,
        MonthlyMilkDataset,
        SunspotsDataset,
        AustralianTourismDataset,
        TaylorDataset, 
        USGasolineDataset, 
        WineDataset,
        WoolyDataset,
        AustralianTourismDataset,
        ExchangeRateDataset,
    )

    # Create a tuple of the Darts datasets to process.
    datasets = (
        AirPassengersDataset,          # 1 cols x 144 rows     datetime index MS | multiplicative trend | SeasonalityMode.MULTIPLICATIVE with a period of 12     
        IceCreamHeaterDataset,         # 2 cols x 198 rows     datetime index MS | no trend | SeasonalityMode.MULTIPLICATIVE with a period of 12
        GasRateCO2Dataset,             # 2 cols x 296 rows     range index | no trend | SeasonalityMode.ADDITIVE with a period of 23
        USGasolineDataset,             # 1 cols x 1578 rows    datetime index | trend: ModelMode.MULTIPLICATIVE | SeasonalityMode.NONE
        # enable below to see more datasets processed
        #WineDataset,                   # 1 cols x 176 rows     datetime index MS | multiplicative trend | SeasonalityMode.MULTIPLICATIVE with period of 4 (and 12?)
        #WoolyDataset,                  # 1 cols x 119 rows     datetime index QS-OCT | multiplicative trend | SeasonalityMode.MULTIPLICATIVE with a period of 4 (and 30?)
        #AusBeerDataset,                # 1 cols x 211 rows     datetime index QS-OCT | multiplicative trend | SeasonalityMode.MULTIPLICATIVE with a period of 4  
        #MonthlyMilkDataset,            # 1 cols x 168 rows     datetime index MS | multiplicative trend | SeasonalityMode.MULTIPLICATIVE with period of 12
        # Below have many rows and/or columns and take longer to process
        #HeartRateDataset,              # 1 cols x 1800 rows    range index | no trend | SeasonalityMode.NONE
        #SunspotsDataset,               # 1 cols x 2820 rows    datetime index MS | no trend | SeasonalityMode.NONE | random about mean near 40 with + bias    
        #ExchangeRateDataset,           # 8 cols x 7588 rows    range index | no trend | SeasonalityMode.NONE 
        #AustralianTourismDataset,      # 96 cols x 36 rows     range index | no trend | SeasonalityMode.MULTIPLICATIVE with period of 4
        #TaylorDataset,                 # 1 cols x 4032 rows    index range/int type, trend: linear / additive, SeasonalityMode.NONE
    )
    # https://unit8co.github.io/darts/generated_api/darts.datasets.html

    for dataset in datasets:
        ts = dataset().load()
        print("\n\n# ---------------------------------------------------------------------------")
        print("Dataset: ", dataset.__name__, "  " + str(ts.width) + " cols x " + str(len(ts)) + " rows")
        
        if ts.has_datetime_index:
            print("Series has datetime index with frequency of '" + str(ts.freq) + "' or '" + str(ts.freq_str) + "'")
            # Below tests custom function get_idx_datetime_attr() to be sure the value of ts.freq_str is encoded in the function.
            # If an error is raised, go to get_idx_datetime_attr() and edit it.  
            print("get_idx_datetime_attr(ts.freq_str):", get_idx_datetime_attr(ts.freq_str))
        else:
            print("Series has range index (int) with unit '" + str(ts.time_dim) + "' and definition of: " + str(ts.time_index))

        # Check for gaps in the index caused by NaN/NaT
        if index_has_gaps(ts.pd_dataframe()): raise Exception("ts has gaps in the index likely caused by NaN/NaT")

        # Check the series components/columns for NaN/NaT values and attempt to resolve if found.
        if df_has_nan_or_nat(ts.pd_dataframe(), results_by_col=False, verbose=False): 
            df_has_nan_or_nat(ts.pd_dataframe(), results_by_col=False, verbose=True)
            len_ts = len(ts)
            err = False
            try:
                ts = ts.longest_contiguous_slice()
            except Exception as e:
                print("\t", "ERROR: " + repr(e))
                err = True
            if not err:
                print("WARNING: extracted longest continguous slice without NaN/NaT, series was " + str(len_ts) + " rows, now " + str(len(ts)), "\n")
            else:
                # Alternative method of dealing with NaN/NaT
                print("WARNING:  Dropping these rows with NaN/NaT:")
                df = ts.pd_dataframe()
                # print the rows that have NaN/NaT
                print(df[df.isnull().any(axis=1)])
                # drop the rows with NaN/NaT
                df.dropna(inplace=True)
                # Create ts from df, but fix missing dates created by .dropna()
                ts = TimeSeries.from_dataframe(df, fill_missing_dates=True)
            if index_has_gaps(ts.pd_dataframe()): raise Exception("The series has one or more columns with NaN / NaT values that cannot be easily reconciled")

        cv_past = None
        cv_future = None
        # If past and/or future covariates, define them here.

        # Try all 26 Darts Local Forecasting Models with the dataset and determine which models fit the best.
        # At the end of the script, a plot with the results for all models and the best model will be shown, 
        # the RMSE will be calculated for each model, and the arguments used for each model will be shown. 
        try_all_darts_lfm(ts=ts, past_covariates=cv_past, future_covariates=cv_future, min_max_scale=min_max_scale, 
                          plot_each_model=False, plot_all_models=True, plot_best_model=True, verbose=True)

    """
    """

    # ---------------------------------------------------------------------------
