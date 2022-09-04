from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv 

import os
import glob
from multiprocessing import freeze_support
from modwtpy.modwt import modwt, modwtmra, imodwt


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
            with open('Result_13_26_52_MODWT_Final_without_val_Dmodel64_nhead32.csv', 'a', encoding='UTF8') as r:
                writer = csv.writer(r)
                writer.writerow([f,nforecast])
            # read the csv file
            df = pd.read_csv(f)
            series = df['Cases']
            print(series)
            print('nForecast = ', nforecast)
            wt = modwt(series, 'haar', int(np.log(len(series))))
            seriesList = []
            train = []
            test = []
            val = []
            nb_time = len(series)
            nb_val = int(nb_time*0.2)

            for i in range(len(wt)):
                wt_df = TimeSeries.from_dataframe(pd.DataFrame(wt[i]))
                seriesList.append(wt_df)
                
                wt_df_train = wt_df[:nb_time-nforecast]
                train.append(wt_df_train)
                
                #wt_df_test = wt_df[nb_time-nforecast-nb_val:nb_time-nforecast]
                #test.append(wt_df_test)
                
                wt_df_val = wt_df[nb_time-nforecast:]
                val.append(wt_df_val)

            models_transformers = []
            for i in range(len(train)):
                transformers = TransformerModel(
                input_chunk_length=12,
                output_chunk_length=1,
                batch_size=32,
                n_epochs=200,
                model_name="transformer"+str(i),
                nr_epochs_val_period=10,
                #d_model=16,
                #nhead=8,
                d_model=64,
                nhead=32,
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
                )
                models_transformers.append(transformers.fit(series = train[i], verbose=True))
            prediction = []
            for i in range(len(train)):
                model_loaded = TransformerModel.load_from_checkpoint("transformer"+str(i), best=False)
                pred_series = model_loaded.historical_forecasts(seriesList[i],
                                                                start = nb_time-nforecast,
                                                                retrain=False,
                                                                verbose=False,
                                                                )
                prediction.append(pred_series)
            prediction_tmp = prediction[0].pd_dataframe()

            for i in range(1,len(prediction)):
                prediction_tmp[i] = prediction[i].pd_dataframe()

            res = imodwt(prediction_tmp.transpose().to_numpy(),'haar')
            index_train = pd.RangeIndex(start=0, stop=nb_time-nforecast, step=1, name="time")
            index = pd.RangeIndex(start=nb_time-nforecast, stop=nb_time, step=1, name="time")
            
            train_reindex = pd.DataFrame(series[:nb_time-nforecast].reset_index(drop=True)).set_index(index_train)
            val_reindex = pd.DataFrame(series[nb_time-nforecast:].reset_index(drop=True)).set_index(index)
            res_pred = pd.DataFrame(res).set_index(index)

            res = TimeSeries.from_dataframe(res_pred)
            train_reindex = TimeSeries.from_dataframe(train_reindex)
            val_reindex = TimeSeries.from_dataframe(val_reindex)

            with open('Result_13_26_52_MODWT_Final_MASE_without_val.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(['WTransformer',rmse(val_reindex,res),mape(val_reindex,res),mae(val_reindex,res),smape(val_reindex,res),mase(val_reindex,res,train_reindex)])
            