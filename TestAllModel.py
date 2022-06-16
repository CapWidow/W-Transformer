import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv 

import os
import glob
from multiprocessing import freeze_support

from darts import TimeSeries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset,USGasolineDataset
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis, plot_hist
from darts.metrics import mape, mase, rmse,mae,smape



#Importing the testing models
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    VARIMA,
    BATS,
    TBATS,
    StatsForecastAutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FourTheta,
    FFT,
    NBEATSModel,
    TFTModel,
    RNNModel,
    TransformerModel,
)

def eval_model(model, train, val,f ):
    model.fit(train)
    forecast = model.predict(len(val))
    rmseRes = rmse(val, forecast)
    #mapeRes = mape(val, forecast)
    maeRes = mae(val, forecast)
    #smapeRes = smape(val, forecast)
    maseRes = mase(val, forecast,train)
    print('model {} obtains RMSE: {:.2f}%'.format(model,rmseRes))
    #print('model {} obtains MAPE: {:.2f}%'.format(model,mapeRes ))
    print('model {} obtains MAE: {:.2f}%'.format(model, maeRes))
    #print('model {} obtains SMAPE: {:.2f}%'.format(model,smapeRes ))
    print('model {} obtains MASE: {:.2f}%'.format(model,maseRes ))
    with open('Result_13_26_52_MASE_Final.csv', 'a', encoding='UTF8') as t:
        writer = csv.writer(t)
        writer.writerow([model,rmseRes,maeRes,maseRes,len(val),f ])

if __name__ == "__main__":
    freeze_support()
    nforecastList = [13,26,52]

    for nforecast in nforecastList:

        # use glob to get all the csv files 
        # in the folder
        path = 'C:/Users/Admin/ResearchW/MODWT/FinalTest'
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        # loop over the list of csv files
        for f in csv_files:
            with open('Result_13_26_52_MASE_Final.csv', 'a', encoding='UTF8') as r:
                writer = csv.writer(r)
                writer.writerow([f,nforecast])
            # read the csv file
            df = pd.read_csv(f)
            series = TimeSeries.from_dataframe(pd.DataFrame(df['Cases']))
            print(series)
            print('nForecast = ', nforecast)
            train, val = series[:len(series)-nforecast], series[len(series)-nforecast:]

            eval_model(ExponentialSmoothing(),train,val,f)

            #eval_model(Prophet(),train,val)
                
            eval_model(AutoARIMA(),train,val,f)

            #eval_model(Theta(),train,val,f)

            #eval_model(BATS(),train,val,f)
            eval_model(RNNModel(input_chunk_length = 24, model='RNN', hidden_dim=25, n_rnn_layers=1, dropout=0.0, training_length=24,
                                                                                                                pl_trainer_kwargs = {"accelerator": "gpu", 
                                                                                                                                    "gpus": -1, 
                                                                                                                                    "auto_select_gpus": True},),train,val,f)


            eval_model(NBEATSModel(input_chunk_length=12, output_chunk_length=1, n_epochs=200, random_state=0,pl_trainer_kwargs = {"accelerator": "gpu", 
                                                                                                                                    "gpus": -1, 
                                                                                                                                    "auto_select_gpus": True},),train,val,f)


            eval_model(TransformerModel(
                input_chunk_length=12,
                output_chunk_length=1,
                batch_size=32,
                n_epochs=200,
                model_name="air_transformer",
                nr_epochs_val_period=10,
                d_model=16,
                nhead=8,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                activation="relu",
                random_state=42,
                save_checkpoints=True,
                force_reset=True,
                pl_trainer_kwargs = {"accelerator": "gpu", 
                                    "gpus": -1, 
                                    "auto_select_gpus": True},
            ),train,val,f)


