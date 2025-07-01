if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

from .baseModel import (
    _Base_Model_senti2, get_custom_loss_funcs,
    get_RELIABILITY_MODE, get_USE_MULTIPROCESSING,
    AI_SENTI2_CHECKPOINTS_DIRECTORY,
)

import tensorflow as tf
import keras
import keras.layers
import keras.optimizers

from io import StringIO
import numpy
from math import ceil
import time
import pandas as pd
from pathlib import Path
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec, Concatenate
from inspect import signature as _signature, _empty


import modules.backtests as backtests
from modules.diskSavesRam import SaveDictModule, SaveObjModule
import strategies_simulations2

from AI.transformersLayer import _Transformers_GPT
from .utilsFuncs import (
    getLr as _getLr, _randomRepares_X_train,
    generator_all_XY_datas as _generator_all_XY_datas,
)


from holo import Pointer
from holo.files import get_unique_name as _get_unique_name
from holo.types_ext import _Serie_Float, _3dArray_Float, _2dArray_Float

_P = ParamSpec("_P")

class Model_senti2_V7(_Base_Model_senti2):
    """evolution of V5 (perf reason) and V6 (to continue it), with a different training methode\n
    still in developement"""

    VERSION = "V7"
    # set the model fixed parameters
    nbPeriodes:int = 35
    series_selection_inputs:"list[str]" = [
        "CLOSE_n", "EMA_n", "RSI", "DVWMA_n", "MFI", "GROW", "CHOP", "CCI",
        "CMF", "CV", "CMF", "W%R", "CMO", "MI", "MIA", "ERI", "MAO",
        "TR_n", "ATR_en",       "ER", "ERd",      "DVRTX", "DVRTX2",
        "DI+", "DI-", "DX", "ADX_e",      "SD_n", "BB%_e", "BBW_e",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "SCAL", "ANGL2", "ANGL", "ANGL_n", "SHARP", "SHARP_n", "FLAT", "FLAT_n",
        "VIN", "VIP",      "AOOU", "AOOD",      "DCW_n", "DC%",
        "SAFU_n", "SAFL_n", "SAFW_n", "SAFD", "SAF%",

        "senti2_c"
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"
    series_selection_inputs_lagged:"list[str]" = []
    "the names of the series (lagged by 1) that will be selected as inputs (THE ORDER IS IMPORTANT !)"


    series_selection_outputs:"list[str]" = []
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"
    series_selection_outputs_delayed:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series (delayed by 1) that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs) + len(series_selection_inputs_lagged)
    number_series_outputs:int = len(series_selection_outputs) + len(series_selection_outputs_delayed)

    _regularize_kargs:"dict[str, Any]" = {"rescale":True, "preferReRangePlus":True}

    def __init__(self,
            _network:"None|keras.Model"=None, _network_dropout:"float|None"=None,
            _more_parameters:"bool|int|float"=False, _normalization_layers:bool=False,
            fastMode:bool=True, _more_periodes:"int|bool"=False,
            _use_hardSig:bool=True)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()
        if get_RELIABILITY_MODE() is False:
            raise ValueError("incorrect reliability mode: use enable_reliable_mode()")

        # create the network
        self.network:"keras.Model"
        if self.network is None:
            ### parameters
            network_dropout:"float|None"
            if _network_dropout is None: network_dropout = 0.10
            elif _network_dropout <= 0.0: network_dropout = None
            else: network_dropout = _network_dropout

            if _more_periodes is True:
                self.nbPeriodes *= 2
            elif _more_periodes is not False:
                self.nbPeriodes = _more_periodes

            moreParamMult:"float|int" = 1
            if isinstance(_more_parameters, (int, float)):
                moreParamMult = _more_parameters

            inputs = keras.Input(shape=(self.nbPeriodes, self.number_series_inputs))

            outputs = self._create_layers_of_network(
                inputs, use_fastMode=fastMode, network_dropout=network_dropout,
                use_more_parameters=(_more_parameters is not False, moreParamMult),
                use_normalization_layers=_normalization_layers, use_hardSig=_use_hardSig,
            )
            self.network = keras.Model(inputs=inputs, outputs=outputs)


        # create the optimizer
        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self._calcNewLearningRate(1e6),
            )

        # create the vars to save the loss func used at compilation
        self.custom_loss_func:str = "custom_loss_func_1"
        self._custom_loss_func_kwarg:"dict[str, Any]" = {"A":6, "N":4}

        # create the config for the plots
        self._plotCfg:"dict[str, dict[str, Any]]" = {
            "globalCfg":{
                "nbSubPlots":5, "tight_layout":True,
                "hlines":[(50.0, 3), (65.0, 3), (0.0, 2), (0.5, 2), (1.0, 2), (0.0900, 0),
                          (0.1000, 0), (0.0100, 0), (0.0200, 0), (10.0, 1), (20.0, 1)],
                "yLims":[(1.5, -1.0, 2), (10.0, 0.0, 4)]
            },
            "train_loss":{}, "train_val_loss":{"emaSmoothing":10},
            "loss":{}, "val_loss":{"emaSmoothing":10},   "loss_avg":{}, "val_loss_avg":{"emaSmoothing":10},
            "oldLoss":{}, "val_oldLoss":{"emaSmoothing":10}, "oldLoss_avg":{}, "val_oldLoss_avg":{"emaSmoothing":10},
            "nbIters":{"subPlot":1, "emaSmoothing":10}, "val_nbIters":{"subPlot":1, "emaSmoothing":10},
                "nbIters_avg":{"subPlot":1, "emaSmoothing":10}, "val_nbIters_avg":{"subPlot":1, "emaSmoothing":10},
            "Trades%":{"subPlot":2, "emaSmoothing":10}, "val_Trades%":{"subPlot":2, "emaSmoothing":10},
                "Trades%_avg":{"subPlot":2, "emaSmoothing":10}, "val_Trades%_avg":{"subPlot":2, "emaSmoothing":10},
            "WinRate%":{"subPlot":3, "emaSmoothing":10}, "val_WinRate%":{"subPlot":3, "emaSmoothing":10},
                "WinRate%_avg":{"subPlot":3, "emaSmoothing":10}, "val_WinRate%_avg":{"subPlot":3, "emaSmoothing":10},
            "TradesDura[h]":{"subPlot":4, "emaSmoothing":5}, "val_TradesDura[h]":{"subPlot":4, "emaSmoothing":5},
                "TradesDura[h]_avg":{"subPlot":4, "emaSmoothing":5}, "val_TradesDura[h]_avg":{"subPlot":4, "emaSmoothing":5},
        }
        func_nbIter = lambda v:f"{v:>4.1f}"
        func_TradesDura = lambda v:f"{v:.2f}"
        self._resumeCfg:"list[tuple[str, dict[str, Any]]]" = [
            ("train_loss", {"alias":"lossAvgTrain"}), ("train_val_loss", {"alias":"valAvgTrain"}),
                ("loss", {}), ("val_loss", {"alias":"val"}),
                ("loss_avg", {"alias":"lossAvg"}), ("val_loss_avg", {"alias":"valAvg"}),
            ("oldLoss", {}), ("val_oldLoss", {"alias":"oldLossVal"}),
                ("oldLoss_avg", {"alias":"oldLossAvg"}), ("val_oldLoss_avg", {"alias":"val_oldLoss_avg"}),
            ("nbIters", {"func":func_nbIter}), ("val_nbIters", {"alias":"nbItersVal", "func":func_nbIter}),
                ("nbIters_avg", {"alias":"nbItersAvg", "func":func_nbIter}),
                ("val_nbIters_avg", {"alias":"nbItersValAvg", "func":func_nbIter}),
            ("Trades%", {}), ("val_Trades%", {"alias":"TradesVal%"}),
                ("Trades%_avg", {"alias":"Trades%Avg"}), ("val_Trades%_avg", {"alias":"TradesVal%Avg"}),
            ("WinRate%", {}), ("val_WinRate%", {"alias":"WinRateVal%"}),
                ("WinRate%_avg", {"alias":"WinRate%Avg"}), ("val_WinRate%_avg", {"alias":"WinRateVal%Avg"}),
            ("TradesDura[h]", {"func":func_TradesDura}), ("val_TradesDura[h]", {"alias":"TradesDura[h]Val", "func":func_TradesDura}),
                ("TradesDura[h]_avg", {"alias":"TradesDura[h]Avg", "func":func_TradesDura}),
                ("val_TradesDura[h]_avg", {"alias":"TradesDura[h]ValAvg", "func":func_TradesDura}),
        ]
        self._backTests_metrics:dict[str, str] = {
            "Avg. Trade [%]":"Trades%", "Avg. Trade Duration":"TradesDura[h]", "Win Rate [%]":"WinRate%",
        }
        "the values to track when back testing (and the ascociated alias, the name of the metric)"

        self._current_checkpoint_nb:int = 0
        
        self._checkpoints_directory:Path = \
            AI_SENTI2_CHECKPOINTS_DIRECTORY.joinpath(
                _get_unique_name(
                    AI_SENTI2_CHECKPOINTS_DIRECTORY, onlyNumbers=False, nbCharacters=8,
                    randomChoice=True, guidlike=True, allowResize=True,
                    filter_dirnames=True, filter_filename=False,
                )
            )

        self._trainCalls_history:"list[tuple[str, dict[str, Any], bool]]" = []


    def compile(self, newLossFunc:"bool|None"=None, loss_func_kwargs:"None|dict[str, Any]"=None)->None:
        #self.network.build((None, self.nbPeriodes, self.number_series_inputs))
        if loss_func_kwargs is None:
            loss_func_kwargs = self._custom_loss_func_kwarg
        elif newLossFunc is None: # => (loss_func_kwargs given) and (newLossFunc = auto)
                newLossFunc = True
        if newLossFunc is None: # => (loss_func_kwargs not given) and (newLossFunc = auto)
            newLossFunc = False

        if newLossFunc is False:
            self.network.compile(loss='mean_squared_error', optimizer=self.optimizer)
        else:
            self.network.compile(
                loss=get_custom_loss_funcs(self.custom_loss_func)(**loss_func_kwargs),
                optimizer=self.optimizer,
            )
            self._custom_loss_func_kwarg = loss_func_kwargs



    def _calcNewLearningRate(self, loss:float)->float:
        loss_to_lr_table:"list[tuple[float, float]]" = [
            (0.08, 0.000_40), (0.04, 0.000_30), (0.03, 0.000_20), (0.02, 0.000_15),
            (0.015, 0.000_10), (0.01, 0.000_075), (0.0075, 0.000_05), (0.005, 0.000_025)
        ]
        return _getLr(loss, loss_to_lr_table)


    def _create_layers_of_network(self, inputs,
            use_fastMode:bool, use_more_parameters:"tuple[bool, float|int]",
            use_normalization_layers:bool, network_dropout:"float|None",
            use_hardSig:bool):
        """`inputs` the tensor of the inputs for the layers\n
        `use_fastMode` whether to use CUDA implementation for the layers (faster but way more memory)\n
        `use_more_parameters`:(_more_parameters is True, moreParamMult)\n
        `network_dropout` the dropout to use, /!\\ None -> disabled\n
        `use_normalization_layers` whether to use normalization layers (not significant)\n
        `use_hardSig` whether to use hard sigmoid (values clip) insted of normal sigmoid (values not clipping)"""

        (_more_parameters, moreParamMult) = use_more_parameters
        LSTM_layer = (keras.layers.CuDNNLSTM if use_fastMode is True else keras.layers.LSTM)

        def create_Dropout_and_conditional_Normalization(inputs):
            nonlocal use_normalization_layers
            # DROPOUT
            outputs = keras.layers.Dropout(network_dropout)(inputs)
            # NORMALIZATION
            if use_normalization_layers is True:
                outputs = keras.layers.Normalization()(outputs)
            return outputs

        ### operations on the timeSerie of datas
        # LSTM
        outputs = LSTM_layer(
            units=(300 if _more_parameters is False else int(600 * moreParamMult)),
            # input_shape=(self.nbPeriodes, self.number_series_inputs),
            return_sequences=True,  name='LSTM_in',
        )(inputs)

        # LSTM
        outputs = LSTM_layer(
            units=(95 if _more_parameters is False else int(200 * moreParamMult)),
            return_sequences=True,  name='LSTM_process',
        )(outputs)
        # DROPOUT & ?NORMALIZATION
        outputs = create_Dropout_and_conditional_Normalization(outputs)

        ### operations on the serie of data
        # FEED FORWARD
        outputs = keras.layers.Dense(
            units=(3*self.number_series_inputs if _more_parameters is False
                    else int(10*self.number_series_inputs * moreParamMult)),
            activation="relu", name='sequence_process_layer_0',
        )(outputs)

        # FEED FORWARD
        outputs = keras.layers.Dense(
            units=(self.number_series_inputs if _more_parameters is False
                    else int(5*self.number_series_inputs * moreParamMult)),
            activation="relu", name='sequence_process_layer_1',
        )(outputs)
        # DROPOUT & ?NORMALIZATION
        outputs = create_Dropout_and_conditional_Normalization(outputs)

        # FLATTEN
        outputs = keras.layers.Flatten()(outputs)

        ### operations on the flattened data
        # FEED FORWARD
        outputs = keras.layers.Dense(
            units=self.number_series_inputs, # has a lot of inputs
            activation="tanh", name='merged_process_layer_1',
        )(outputs)

        # FEED FORWARD
        outputs = keras.layers.Dense(
            units=self.number_series_outputs, name='Output',
            activation=("hard_sigmoid" if use_hardSig is True else "sigmoid"),
        )(outputs)

        return outputs





    def _compute_X_datas_iteration(self,
            X_train:"_3dArray_Float", X_test:"_3dArray_Float",
            Y_train:"_2dArray_Float|None", Y_test:"_2dArray_Float|None",
            batchSizePredict:int, numberOfIterations:int, verbose:int,
            senti2C_noiseToConvStart:"float|bool", errorToStableConv:"float",
            dataFrame_for_backtests:"pd.DataFrame|None",
            skipNbFirst:"int|None", skipNbLast:"int|None", track_train_backTest:bool,
            track_val_backTest:bool, verboseIterCalc_train:int,
            verboseIterCalc_test:int, returnMetrics:bool=False,
            get_Y_train_predict:"Pointer[_Serie_Float]|None"=None,
            get_Y_test_predict:"Pointer[_Serie_Float]|None"=None,
            )->"tuple[_3dArray_Float, _3dArray_Float, dict[str, tuple[float, int]]]":
        """return the computed X_train and X_test, and a dict like: {metricName:(metricValue, nbSamples), ...}"""

        metricsDict:"dict[str, Any]" = {}
        metrics_toReturn:"dict[str, tuple[float, int]]" = {}
        if verboseIterCalc_train != 0: print("computing X_train")
        grab_Y_train_result:bool = ((get_Y_train_predict is not None) or (track_train_backTest is True))
        X_train = ... # REMOVED
        if returnMetrics is True:
            metrics_toReturn["nbIters"] = (metricsDict["nbIterations"], len(X_train))
        else: self._add_trainigHistory({"nbIters":[metricsDict["nbIterations"]]})

        ## backtest_train metrics
        Y_result:"_Serie_Float" = metricsDict["Y_result"]
        if get_Y_train_predict is not None:
            get_Y_train_predict.value = Y_result
        if (track_train_backTest is True) and (self._backTests_metrics is not None):
            if dataFrame_for_backtests is None: raise ValueError("the dataFrame_for_backtests was expected, None was given")
            if (skipNbFirst is None) or (skipNbLast is None): raise ValueError("the skipNbFirst and skipNbLast were expected to be ints, None was given")
            backTest_datas = strategies_simulations2.create_DataFrameBacktest(
                dataFrame_for_backtests, Y_result, self, skipNbFirst, skipNbLast+len(X_test),
            )
            btest_session = backtests.backTest(
                backTest_datas, backtests.Senti2_Strategie, cash=1e9, fees=backtests.FEES,
            )
            btest_runResults:"pd.Series" = btest_session.run().loc[list(self._backTests_metrics.keys())]
            metricsVerbose = StringIO()
            for name in btest_runResults.index:
                metric_name = f"{self._backTests_metrics[name]}"
                raw_value:"Any" = btest_runResults[str(name)]
                if isinstance(raw_value, pd.Timedelta):
                    value = float(raw_value.total_seconds()/3600)
                else: value = float(raw_value)
                if str(value) == "nan": value = 0.0
                metricsVerbose.write(f"{metric_name}: {round(value, 4)}" + ('\n' if verbose == 1 else ', '))
                if returnMetrics is True:
                    metrics_toReturn[metric_name] = (value, len(X_train))
                else: self._add_trainigHistory({metric_name: [value]})
                del name, value, metric_name
            if verbose != 0: print(metricsVerbose.getvalue())
            del backTest_datas, btest_runResults, btest_session, metricsVerbose
        del Y_result


        if verboseIterCalc_test != 0: print("computing X_test")
        grab_Y_test_result:bool = ((get_Y_test_predict is not None) or (track_val_backTest is True))
        X_test = ... # REMOVED
        if returnMetrics is True:
            metrics_toReturn["val_nbIters"] = (metricsDict["nbIterations"], len(X_test))
        else: self._add_trainigHistory({"val_nbIters":[metricsDict["nbIterations"]]})
        ## backtest_val metrics
        Y_result:"_Serie_Float" = metricsDict["Y_result"]
        if get_Y_test_predict is not None:
            get_Y_test_predict.value = Y_result
        if (track_val_backTest is True) and (self._backTests_metrics is not None):
            if dataFrame_for_backtests is None: raise ValueError("the dataFrame_for_backtests was expected, None was given")
            if (skipNbFirst is None) or (skipNbLast is None): raise ValueError("the skipNbFirst and skipNbLast were expected to be ints, None was given")
            backTest_datas = strategies_simulations2.create_DataFrameBacktest(
                    dataFrame_for_backtests, Y_result, self, skipNbFirst+len(X_train), skipNbLast,
                )
            btest_session = backtests.backTest(
                    backTest_datas, backtests.Senti2_Strategie, cash=1e9, fees=backtests.FEES,
                )
            btest_runResults:"pd.Series" = btest_session.run().loc[list(self._backTests_metrics.keys())]
            metricsVerbose = StringIO()
            for name in btest_runResults.index:
                metric_name = f"val_{self._backTests_metrics[name]}"
                raw_value:"Any" = btest_runResults[str(name)]
                if isinstance(raw_value, pd.Timedelta):
                    value = float(raw_value.total_seconds()/3600)
                else: value = float(raw_value)
                if str(value) == "nan": value = 0.0
                metricsVerbose.write(f"{metric_name}: {round(value, 4)}" + ('\n' if verbose == 1 else ', '))
                if returnMetrics is True:
                    metrics_toReturn[metric_name] = (value, len(X_test))
                else: self._add_trainigHistory({metric_name: [value]})
                del name, value, metric_name
            if verbose != 0: print(metricsVerbose.getvalue())
            del backTest_datas, btest_runResults, btest_session, metricsVerbose
        del Y_result
        return X_train, X_test, metrics_toReturn

    def __compute_loss(self,
            X_data:"_3dArray_Float", Y_data:"_2dArray_Float", batchSize:int, verbose:int, metricName:str)->"tuple[str, float]":
        if verbose == 1: print(f"computing {metricName}, ", end="", flush=True)
        metric_value:float = self._base_evaluate(X_data, Y_data, batchSize, verbose, metricName)
        if verbose != 0: print(f"{metricName}: {metric_value :.4f}")
        return (metricName, metric_value)

    def _compute_new_Lr(self, verbose:int, learningRateFactor:float, _use_loss:"None|float"=None)->float:
        currentTrainLoss:float
        if _use_loss is None:
            # get the last TRAINING loss (not val)
            currentTrainLoss = self._trainingHistory["loss"][-1]
        else: currentTrainLoss = _use_loss
        # calc and set the new learning rate
        newLr:float = learningRateFactor * self._calcNewLearningRate(currentTrainLoss)
        prevLr:float = self.get_learning_rate()
        if verbose != 0:
            print(f"learning rate updated from {prevLr:.3e} to {newLr:.3e}")
        return newLr

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, batchSizePredict:int, numberOfIterations:int,
            verbose:int=1, verboseIterCalc:"int|tuple[int, int]"=1,
            senti2C_noiseToConvStart:"float|bool"=False,
            normalTrainingProportion:float=0.0, errorToStableConv:float=1e-4,
            learningRateFactor:float=1.0, dynamicLearningRate:bool=True,
            track_train_oldLoss:bool=True, track_val_oldLoss:bool=True,
            dataFrame_for_backtests:"None|pd.DataFrame"=None,
            skipNbFirst:"int|None"=None, skipNbLast:"int|None"=None, # needed for backtests
            track_train_backTest:bool=False, track_val_backTest:bool=False,
            _shuffle_train_datas:bool=True)->None:
        """train the ai by first calculating the convergence serie, then train the ai on it\n
        /!\\ warning the iter calc is only for the X_train, the X_test stay the same\n
        `verboseIterCalc` 0->no infos, 2->infos on single line, 1-> full infos and info's hist\
        when a tuple is give: (vebose for train conv, verbose for test conv)\n
        `noiseToConvStart` the noise to add from X_train when starting the convergence\n
        `normalTrainingProportion` the proportion of normal training (with no convergence)\n
        `track_(train|val)_backTest` is a list of the metrics from the backtests to track\
            they are evaluated before appling the DL to the network (to avoid loosing time)\n
        """
        __kwargs:"dict[str, Any]" = {
            "nbEpoches":nbEpoches, "batchSize":batchSize, "batchSizePredict":batchSizePredict, "numberOfIterations":numberOfIterations,
            "senti2C_noiseToConvStart":senti2C_noiseToConvStart, "normalTrainingProportion":normalTrainingProportion,
            "errorToStableConv":errorToStableConv, "learningRateFactor":learningRateFactor, "dynamicLearningRate":dynamicLearningRate,
            "skipNbFirst":skipNbFirst, "skipNbLast":skipNbLast, "_shuffle_train_datas":_shuffle_train_datas,
        }
        self._trainCalls_history.append(("train", __kwargs, False))
        _index_trainCall:int = len(self._trainCalls_history) -1

        if verbose == 0:
            verboseIterCalc = 0
            print(r"/!\ train verbose is OFF")

        verboseIterCalc_train:int
        verboseIterCalc_test:int
        if isinstance(verboseIterCalc, int):
            verboseIterCalc_train = verboseIterCalc
            verboseIterCalc_test = verboseIterCalc
        else :
            verboseIterCalc_train = verboseIterCalc[0]
            verboseIterCalc_test = verboseIterCalc[1]

        X_train_repare:_2dArray_Float = X_train[:, :, -1].copy()
        X_test_repare:_2dArray_Float = X_test[:, :, -1].copy()

        for epoche in range(1, nbEpoches+1):
            print(" "*5 + f"* Epoche {epoche}/{nbEpoches}")

            if normalTrainingProportion < 1.00:
                X_train, X_test, _ = self._compute_X_datas_iteration(
                    X_train, X_test, Y_train, Y_test, batchSizePredict, numberOfIterations,
                    verbose, senti2C_noiseToConvStart, errorToStableConv, dataFrame_for_backtests,
                    skipNbFirst, skipNbLast, track_train_backTest, track_val_backTest,
                    verboseIterCalc_train, verboseIterCalc_test, returnMetrics=False,
                )

            if normalTrainingProportion > 0.0:
                X_train = _randomRepares_X_train(
                    X_train, X_train_repare, normalTrainingProportion,
                )

            # train, the ai on the calculated X_train
            self._add_trainigHistory(
                self._base_train(
                    X_train=X_train, Y_train=Y_train,
                    X_test=X_test, Y_test=Y_test,
                    nbEpoches=1, batchSize=batchSize, verbose=verbose,
                    shuffle_train=_shuffle_train_datas,
                )
            )

            # repare X_... so it is just like before the iter calc
            X_train[:, :, -1] = X_train_repare
            X_test[:, :, -1] = X_test_repare


            # oldLoss computation
            if track_train_oldLoss is True:
                self._add_trainigMetricValue(*self.__compute_loss(X_train, Y_train, batchSizePredict, verbose, "oldLoss"))

            # val_oldLoss computation
            if track_val_oldLoss is True:
                self._add_trainigMetricValue(*self.__compute_loss(X_test, Y_test, batchSizePredict, verbose, "val_oldLoss"))

            # dynamic Lr calculation
            if dynamicLearningRate is True:
                self.set_learning_rate(self._compute_new_Lr(verbose, learningRateFactor))

        # set as finished
        self._trainCalls_history[_index_trainCall] = (*self._trainCalls_history[_index_trainCall][ :2], True)




    def __compute_XY_datas_saveDicts(self,
            dfs_to_train:"list[str]",
            all_dataframes_regularized_series:"SaveDictModule[str, dict[str, _Serie_Float]]",
            skipNbFirst:int, skipNbLast:int, testSamplesProportion:float,
            applieGlobalNoise:"float|None", applieSenti2Noise:"float|None",
            batchSize:int, verbose:int)->"tuple[SaveDictModule[str, dict[str, numpy.ndarray]], int, int]":
        # calc tout les X|Y dans un meme SaveDict
        # et applie les randoms in place au moment voulu
        all_dataframes_XY_saved:"SaveDictModule[str, dict[str, numpy.ndarray]]" = \
            SaveDictModule({}, greedySaves=False) # NO greedySaves=True: it would crash at del
        ### precompute the nb of steps for the whole epoche
        steps_per_train_epoch:int = 0
        steps_per_test_epoch:int = 0

        end_line:str = "\n"
        if verbose >= 2: end_line = "\r"
        if verbose >= 1: print("computing the XY train/test", end=end_line, flush=True)
        for df_key in dfs_to_train:
            if verbose >= 1: print(f"computing XY train/test for {df_key}", end=end_line, flush=True)
            ((X_train, Y_train), (X_test, Y_test)) = \
                self.split_datas(
                    all_dataframes_regularized_series[df_key],
                    skipNbFirst=skipNbFirst, skipNbLast=skipNbLast,
                    testSamplesProportion=testSamplesProportion,
                    chooseTestSamplesRandomly=False, randomizeSamples=False,
                    applieGlobalNoise=applieGlobalNoise, applieSenti2Noise=applieSenti2Noise,
                )
            all_dataframes_XY_saved[df_key] = {
                "X_train":X_train, "Y_train":Y_train, "X_test":X_test, "Y_test":Y_test
            }
            steps_per_train_epoch += ceil(len(X_train) / batchSize)
            steps_per_test_epoch += ceil(len(X_test) / batchSize)
            del X_train, Y_train, X_test, Y_test
            all_dataframes_XY_saved.save(df_key)
        if verbose >= 2: print()

        return all_dataframes_XY_saved, steps_per_train_epoch, steps_per_test_epoch

    def __train2_internal__load_XY_datas(self,
            verbose:int, all_dataframes_XY_saved:"SaveDictModule[str, dict[str, numpy.ndarray]]",
            df_key:str)->"tuple[_3dArray_Float, _3dArray_Float, _2dArray_Float, _2dArray_Float]":
        tload = time.perf_counter()
        print(f" -- dataframe: {df_key}")

        dict_XY_saved:"SaveObjModule[dict[str, numpy.ndarray]]" = \
            all_dataframes_XY_saved.getRawSaveModuleObject(df_key)
        dict_XY_saved.load()
        X_train:"_3dArray_Float" = dict_XY_saved.value["X_train"]
        X_test:"_3dArray_Float" = dict_XY_saved.value["X_test"]
        Y_train:"_2dArray_Float" = dict_XY_saved.value["Y_train"]
        Y_test:"_2dArray_Float" = dict_XY_saved.value["Y_test"]
        if verbose == 1: print(f"loaded in {time.perf_counter() - tload:.3f} sec")

        print(f"shapes -> X_train:{X_train.shape}, Y_train:{Y_train.shape}")
        print(f"shapes -> X_test:{X_test.shape}, Y_test:{Y_test.shape}")

        dict_XY_saved.unLoad_noSave()
        return X_train, X_test, Y_train, Y_test


    def __train2_internal__increment_metric(self,
            metrics_dict:"dict[str, tuple[float, int]]",
            metricName:"str", metricValue:float, nbSamples:int)->None:
        if metricName in metrics_dict:
            (old_metric_sum, old_tt_nbSamples) = metrics_dict[metricName]
            metrics_dict[metricName] = \
                (old_metric_sum + (metricValue * nbSamples), old_tt_nbSamples + nbSamples)
        else: metrics_dict[metricName] = (metricValue * nbSamples, nbSamples)

    def __train2_internal__increment_multiple_metrics(self,
            metrics_dict:"dict[str, tuple[float, int]]",
            metrics_to_add:"dict[str, tuple[float, int]]")->None:

        for metricName, (metric_val, nbSamples) in metrics_to_add.items():
            self.__train2_internal__increment_metric(metrics_dict, metricName, metric_val, nbSamples)

    def __train2_internal__metrics_toHist(self,
            metrics_dict:"dict[str, tuple[float, int]]",
            names_suffix:str="", applie_averages:bool=False)->"dict[str, list[float]]":
        """if applie_averages is True: consider the values are sums and compute the average\\
        else: consider the values are the raw metric's value, and ignore the nbSamples"""
        metrics_hist:"dict[str, list[float]]" = {}
        for metricName, (metric_val, nbSamples) in metrics_dict.items():
            metrics_hist[metricName + names_suffix] = [metric_val / (1.0 if applie_averages is False else nbSamples)]
        return metrics_hist

    def __train2_internal__compute_loss_and_update_metrics(self,
            X_data:"_3dArray_Float", Y_data:"_2dArray_Float", batchSizeIteration:int, add_metrics_per_dfs:bool,
            metric_name:str, verbose:int, all_dfs_metrics:"dict[str, tuple[float, int]]")->None:
        metric_name, metric_value = self.__compute_loss(X_data, Y_data, batchSizeIteration, verbose, metric_name)
        if add_metrics_per_dfs: self._add_trainigMetricValue(metric_name, metric_value)
        self.__train2_internal__increment_metric(all_dfs_metrics, metric_name, metric_value, len(X_data))

    def train2(self,
            # data creation
            dfs_to_train:"list[str]|None",
            all_dataframes:"SaveDictModule[str, pd.DataFrame]",
            all_dataframes_regularized_series:"SaveDictModule[str, dict[str, _Serie_Float]]",
            skipNbFirst:int, skipNbLast:int, # needed for backtests and data calc
            testSamplesProportion:float, applieGlobalNoise:"float|None", applieSenti2Noise:"float|None",
            shuffle_train_datas:bool,
            # training
            nbEpoches:int, batchSize:int,
            batchSizeIteration:"int|None"=None, numberOfIterations:int=60,
            verbose:int=1, verboseIterCalc:"int|tuple[int, int]"=1,
            senti2C_noiseToConvStart:"float|bool"=False,
            normalTrainingProportion:float=0.0, errorToStableConv:float=1e-4,
            learningRateFactor:float=1.0, dynamicLearningRate:bool=True,
            track_train_oldLoss:bool=True, track_val_oldLoss:bool=True,
            track_dfs_train_loss:bool=True, track_dfs_val_loss:bool=True,
            track_train_backTest:bool=False, track_val_backTest:bool=False,
            regroup_dfs:bool=True, add_metrics_per_dfs=False)->None:
        __kwargs:"dict[str, Any]" = {
            "dfs_to_train":dfs_to_train, "regroup_dfs":regroup_dfs, "nbEpoches":nbEpoches, "normalTrainingProportion":normalTrainingProportion,
            "testSamplesProportion":testSamplesProportion, "skipNbFirst":skipNbFirst, "skipNbLast":skipNbLast,
            "applieGlobalNoise":applieGlobalNoise, "applieSenti2Noise":applieSenti2Noise,
            "learningRateFactor":learningRateFactor, "dynamicLearningRate":dynamicLearningRate,
            "numberOfIterations":numberOfIterations, "senti2C_noiseToConvStart":senti2C_noiseToConvStart, "errorToStableConv":errorToStableConv,
            "shuffle_train_datas":shuffle_train_datas, "batchSize":batchSize, "batchSizeIteration":batchSizeIteration,
        }
        self._trainCalls_history.append(("train", __kwargs, False))
        _index_trainCall:int = len(self._trainCalls_history) -1


        if verbose == 0:
            verboseIterCalc = 0
            print(r"/!\ train verbose is OFF")
        verboseIterCalc_train:int; verboseIterCalc_test:int
        if isinstance(verboseIterCalc, int):
            verboseIterCalc_train = verboseIterCalc; verboseIterCalc_test = verboseIterCalc
        else: verboseIterCalc_train = verboseIterCalc[0]; verboseIterCalc_test = verboseIterCalc[1]

        if dfs_to_train is None: dfs_to_train = all_dataframes_regularized_series.keys()
        if batchSizeIteration is None: batchSizeIteration = batchSize

        # pré-compute the XY_datas in a saveDict
        all_dataframes_XY_saved, steps_per_train_epoch, steps_per_test_epoch = \
            self.__compute_XY_datas_saveDicts(
                dfs_to_train, all_dataframes_regularized_series, skipNbFirst, skipNbLast,
                testSamplesProportion, applieGlobalNoise, applieSenti2Noise,
                batchSize, verbose,
            )

        print(f" ** steps_per_train_epoch= {steps_per_train_epoch}, ", )

        all_XY_train_computed:"SaveDictModule[str, dict[str, numpy.ndarray]]" = \
            SaveDictModule({}, greedySaves=False)
        all_XY_test_computed:"SaveDictModule[str, dict[str, numpy.ndarray]]" = \
            SaveDictModule({}, greedySaves=False)

        X_train:"_3dArray_Float"; X_test:"_3dArray_Float"
        Y_train:"_2dArray_Float"; Y_test:"_2dArray_Float"
        try:
            for epoche in range(1, nbEpoches+1):
                print("*"*5 + f"* Epoche {epoche}/{nbEpoches}")

                all_XY_train_computed.clear()
                all_XY_test_computed.clear()

                all_dfs_metrics:"dict[str, tuple[float, int]]" = {}

                for df_key in dfs_to_train:

                    X_train, X_test, Y_train, Y_test = \
                        self.__train2_internal__load_XY_datas(verbose, all_dataframes_XY_saved, df_key)

                    X_train_repare:"_2dArray_Float" = X_train[:, :, -1].copy()
                    X_test_repare:"_2dArray_Float" = X_test[:, :, -1].copy()
                    self._applie_random_noise_on_datas(X_train, None, applieSenti2Noise, applieSenti2Noise)


                    df_for_backtests:"None|pd.DataFrame" = None
                    if (track_train_backTest is True) or (track_val_backTest is True):
                        df_for_backtests = all_dataframes[df_key]

                    if normalTrainingProportion < 1.00:
                        X_train, X_test, df_metrics = self._compute_X_datas_iteration(
                            X_train, X_test, Y_train, Y_test, batchSizeIteration, numberOfIterations,
                            verbose, senti2C_noiseToConvStart, errorToStableConv, df_for_backtests,
                            skipNbFirst, skipNbLast, track_train_backTest, track_val_backTest,
                            verboseIterCalc_train, verboseIterCalc_test, returnMetrics=True,
                        )

                        if normalTrainingProportion > 0.0:
                            X_train = _randomRepares_X_train(
                                X_train, X_train_repare, normalTrainingProportion,
                            )

                        if add_metrics_per_dfs is True:
                            self._add_trainigHistory(self.__train2_internal__metrics_toHist(df_metrics, applie_averages=False))
                        self.__train2_internal__increment_multiple_metrics(all_dfs_metrics, df_metrics)
                        del df_metrics

                    if regroup_dfs is False:
                        # train, the ai on the single Df
                        self._add_trainigHistory(
                            self._base_train(
                                X_train=X_train, Y_train=Y_train,
                                X_test=X_test, Y_test=Y_test,
                                nbEpoches=1, batchSize=batchSize, verbose=verbose,
                                shuffle_train=shuffle_train_datas,
                            )
                        )
                        self.__train2_internal__increment_metric(
                            all_dfs_metrics, "loss", self._trainingHistory["loss"][-1], len(X_train))
                    else:
                        # set the calculated XY_datas for the later train and test
                        all_XY_train_computed[df_key] = {"X_datas":X_train, "Y_datas":Y_train}
                        all_XY_train_computed.save(df_key)
                        all_XY_test_computed[df_key] = {"X_datas":X_test, "Y_datas":Y_test}
                        all_XY_test_computed.save(df_key)


                    if (regroup_dfs is True) and (add_metrics_per_dfs is True):
                        # loss computation
                        if track_dfs_train_loss is True:
                            self.__train2_internal__compute_loss_and_update_metrics(
                                X_train, Y_train, batchSizeIteration, add_metrics_per_dfs, "loss", verbose, all_dfs_metrics)
                        # val_oldLoss computation
                        if track_dfs_val_loss is True:
                            self.__train2_internal__compute_loss_and_update_metrics(
                                X_test, Y_test, batchSizeIteration, add_metrics_per_dfs, "val_loss", verbose, all_dfs_metrics)

                    # repare X_... so it is just like before the iter calc
                    X_train[:, :, -1] = X_train_repare
                    X_test[:, :, -1] = X_test_repare

                    # oldLoss computation
                    if track_train_oldLoss is True:
                        self.__train2_internal__compute_loss_and_update_metrics(
                            X_train, Y_train, batchSizeIteration, add_metrics_per_dfs, "oldLoss", verbose, all_dfs_metrics)
                    # val_oldLoss computation
                    if track_val_oldLoss is True:
                        self.__train2_internal__compute_loss_and_update_metrics(
                            X_test, Y_test, batchSizeIteration, add_metrics_per_dfs, "val_oldLoss", verbose, all_dfs_metrics)


                    all_dataframes.unLoad_noSave(df_key)
                    del X_train, Y_train, X_test, Y_test, df_for_backtests # free the RAM asap

                    print()


                ## run the trainng  update the Lr
                if regroup_dfs is True:
                    # train on the generator
                    train_history = self._base_train_generator(
                        _generator_all_XY_datas(all_XY_train_computed, batchSize, shuffle_train_datas, 1),
                        _generator_all_XY_datas(all_XY_test_computed, batchSize, False, 1),
                        steps_per_train_epoch=steps_per_train_epoch, steps_per_test_epoch=steps_per_test_epoch,
                        nbEpoches=1, verbose=verbose,
                    )
                    if add_metrics_per_dfs is False:
                        self._add_trainigHistory(train_history)
                    else:
                        prefix:str = ("train_" if add_metrics_per_dfs is True else "")
                        self._add_trainigHistory({
                            (prefix + key):val
                            for (key, val) in train_history.history.items()
                        })
                        del prefix

                    # get the last loss for the dynamic Lr calculation
                    if dynamicLearningRate is True:
                        dict_metrics:"dict[str, list[float]]" = train_history.history
                        lastLoss = dict_metrics["loss"][-1]
                        self.set_learning_rate(self._compute_new_Lr(verbose, learningRateFactor, _use_loss=lastLoss))
                        del dict_metrics, lastLoss
                    del train_history

                    ## compute all the averages metrics
                    suffix:str = ("_avg" if add_metrics_per_dfs is True else "")
                    self._add_trainigHistory(
                        self.__train2_internal__metrics_toHist(
                            all_dfs_metrics, names_suffix=suffix, applie_averages=True)
                        )
                    del suffix

                # else: => (regroup_dfs is False) => Lr update alredy applied

                print("\n\n")

        finally:
            all_XY_train_computed.clear()
            all_XY_test_computed.clear()
            all_dataframes_XY_saved.clear()

        # set as finished
        self._trainCalls_history[_index_trainCall] = (*self._trainCalls_history[_index_trainCall][ :2], True)

    def __train3_internal__load_XY_datas(self,
            verbose:int, all_dataframes_X_saved:"SaveDictModule[str, dict[str, _3dArray_Float]]",
            df_key:str)->"tuple[_3dArray_Float, _3dArray_Float]":
        tload = time.perf_counter()
        print(f" -- dataframe: {df_key}")

        dict_XY_saved:"SaveObjModule[dict[str, numpy.ndarray]]" = \
            all_dataframes_X_saved.getRawSaveModuleObject(df_key)
        dict_XY_saved.load()
        X_train:"_3dArray_Float" = dict_XY_saved.value["X_train"]
        X_test:"_3dArray_Float" = dict_XY_saved.value["X_test"]
        if verbose == 1: print(f"loaded in {time.perf_counter() - tload:.3f} sec")

        print(f"shapes -> X_train:{X_train.shape}, X_test:{X_test.shape}")

        dict_XY_saved.unLoad_noSave()
        return X_train, X_test
        
    
    def train3(self,
            # data creation
            dfs_to_train:"list[str]|None",
            all_dataframes:"SaveDictModule[str, pd.DataFrame]",
            all_dataframes_regularized_series:"SaveDictModule[str, dict[str, _Serie_Float]]",
            skipNbFirst:int, skipNbLast:int, # needed for backtests and data calc
            testSamplesProportion:float, applieGlobalNoise:"float|None", applieSenti2Noise:"float|None",
            shuffle_train_datas:bool,
            # training
            nbEpoches:int, batchSize:int, correction_function_version:int,
            #correction_function:"Callable[Concatenate[_Serie_Float, _Serie_Float, _P], _2dArray_Float]",
            correction_function_parameters:"dict[str, Any]",
            batchSizeIteration:"int|None"=None, numberOfIterations:int=60,
            verbose:int=1, verboseIterCalc:"int|tuple[int, int]"=1,
            senti2C_noiseToConvStart:"float|bool"=False, errorToStableConv:float=1e-4,
            learningRateFactor:float=1.0, dynamicLearningRate:bool=True, add_metrics_per_dfs=False,
            track_dfs_train_loss:bool=True, track_dfs_val_loss:bool=True,
            track_train_backTest:bool=False, track_val_backTest:bool=False,
            saveCheckpointEvery_CorrectionStep:bool=False, miniEpoches_betwinCheckpoints:"int|None"=None,
            correctionStepEveryLoss:"None|bool"=None,
            #*correction_function_args:_P.args, **correction_function_kwargs:_P.kwargs
            )->None:
        """`correctionStepEveryLoss` is whether to use a new correctioon every step (None), \
            or to use a new correction each time the loss is under `correctionStepEveryLoss`
        `saveCheckpointEvery_CorrectionStep` when true, it will only save a cp when """
        self.__train3_internal__check_correctionFunc_parameters(
            correction_function_version, correction_function_parameters
        ) # done first to avoid a mistake while making a call
        
        __kwargs:"dict[str, Any]" = {
            "dfs_to_train":dfs_to_train, "skipNbFirst":skipNbFirst, "skipNbLast":skipNbLast, "testSamplesProportion":testSamplesProportion,
            "applieGlobalNoise":applieGlobalNoise, "applieSenti2Noise":applieSenti2Noise, "shuffle_train_datas":shuffle_train_datas,
            "nbEpoches":nbEpoches, "batchSize":batchSize, "batchSizeIteration":batchSizeIteration, "numberOfIterations":numberOfIterations,
            "correction_function_version":correction_function_version, "correction_function_parameters":correction_function_parameters,
            "senti2C_noiseToConvStart":senti2C_noiseToConvStart, "correctionStepEveryLoss":correctionStepEveryLoss,
            "saveCheckpointEvery_CorrectionStep":saveCheckpointEvery_CorrectionStep, "errorToStableConv":errorToStableConv,
            "learningRateFactor":learningRateFactor, "dynamicLearningRate":dynamicLearningRate, "miniEpoches_betwinCheckpoints":miniEpoches_betwinCheckpoints,
        }
        self._trainCalls_history.append(("train", __kwargs, False))
        _index_trainCall:int = len(self._trainCalls_history) -1

        if verbose == 0:
            verboseIterCalc = 0
            print(r"/!\ train verbose is OFF")
        verboseIterCalc_train:int; verboseIterCalc_test:int
        if isinstance(verboseIterCalc, int):
            verboseIterCalc_train = verboseIterCalc; verboseIterCalc_test = verboseIterCalc
        else: verboseIterCalc_train = verboseIterCalc[0]; verboseIterCalc_test = verboseIterCalc[1]

        if dfs_to_train is None: dfs_to_train = all_dataframes_regularized_series.keys()
        if batchSizeIteration is None: batchSizeIteration = batchSize


        # pré-compute the XY_datas in a saveDict
        all_dataframes_XY_saved, steps_per_train_epoch, steps_per_test_epoch = \
            self.__compute_XY_datas_saveDicts(
                dfs_to_train, all_dataframes_regularized_series, skipNbFirst, skipNbLast,
                testSamplesProportion, applieGlobalNoise, applieSenti2Noise,
                batchSize, verbose,
            )
        print(f" ** steps_per_train_epoch= {steps_per_train_epoch}")


        all_X_train_computed:"SaveDictModule[str, _3dArray_Float]" = SaveDictModule({}, greedySaves=False)
        all_X_test_computed:"SaveDictModule[str, _3dArray_Float]" = SaveDictModule({}, greedySaves=False)
        all_Y_train_computed:"SaveDictModule[str, _2dArray_Float]" = SaveDictModule({}, greedySaves=False)
        all_Y_test_computed:"SaveDictModule[str, _2dArray_Float]" = SaveDictModule({}, greedySaves=False)
        computeNewCorrection:bool = True
        _miniEpoches_betwinCP:int = (0 if miniEpoches_betwinCheckpoints is None else miniEpoches_betwinCheckpoints)
        epoches_sinceLastCp:int = _miniEpoches_betwinCP # => it allow to save cp directly after the first epoche

        X_train:"_3dArray_Float"; X_test:"_3dArray_Float"
        Y_train:"_2dArray_Float"; Y_test:"_2dArray_Float"
        try:
            for epoche in range(1, nbEpoches+1):
                print("*"*5 + f"* Epoche {epoche}/{nbEpoches}")

                all_X_train_computed.clear()
                all_X_test_computed.clear()
                if (correctionStepEveryLoss is None) or (computeNewCorrection is True):
                    all_Y_train_computed.clear()
                    all_Y_test_computed.clear()


                all_dfs_metrics:"dict[str, tuple[float, int]]" = {}

                for df_key in dfs_to_train:

                    X_train, X_test = \
                        self.__train3_internal__load_XY_datas(verbose, all_dataframes_XY_saved, df_key)

                    self._applie_random_noise_on_datas(X_train, None, applieSenti2Noise, applieSenti2Noise)

                    df_for_backtests:"None|pd.DataFrame" = None
                    if (track_train_backTest is True) or (track_val_backTest is True):
                        df_for_backtests = all_dataframes[df_key]

                    get_Y_train_predict:Pointer[_Serie_Float] = Pointer()
                    get_Y_test_predict:Pointer[_Serie_Float] = Pointer()
                    X_train, X_test, df_metrics = self._compute_X_datas_iteration(
                        X_train, X_test, None, None, batchSizeIteration, numberOfIterations, verbose,
                        senti2C_noiseToConvStart, errorToStableConv, df_for_backtests,
                        skipNbFirst, skipNbLast, track_train_backTest, track_val_backTest,
                        verboseIterCalc_train, verboseIterCalc_test, returnMetrics=True,
                        get_Y_train_predict=get_Y_train_predict, get_Y_test_predict=get_Y_test_predict,
                    )
                    # dont do random reparations, not appropriate

                    #### compute the Y_correction for the train
                    if (correctionStepEveryLoss is None) or (computeNewCorrection is True):
                        full_close_array:"_Serie_Float" = all_dataframes[df_key]["Close"].to_numpy()
                        aligned_close_array:"_Serie_Float" = self.synconize_array(
                            full_close_array, skipNbFirst, skipNbLast + len(X_test), lag=-1, shiftNbPeriodes=True)
                        Y_train:"_2dArray_Float" = self.__train3_internal__compute_Y_correction(
                            get_Y_train_predict.value, aligned_close_array,
                            correction_function_version, correction_function_parameters,
                        )
                        #Y_train:"_2dArray_Float" = correction_function(
                        #    get_Y_train_predict.value, aligned_close_array,
                        #    *correction_function_args, **correction_function_kwargs,
                        #)
                        del aligned_close_array

                        aligned_close_array:"_Serie_Float" = self.synconize_array(
                            full_close_array, skipNbFirst + len(X_train), skipNbLast, lag=-1, shiftNbPeriodes=True)
                        #Y_test:"_2dArray_Float" = correction_function(
                        #    get_Y_test_predict.value, aligned_close_array,
                        #    *correction_function_args, **correction_function_kwargs,
                        #)
                        Y_test:"_2dArray_Float" = self.__train3_internal__compute_Y_correction(
                            get_Y_test_predict.value, aligned_close_array,
                            correction_function_version, correction_function_parameters,
                        )
                        del aligned_close_array, full_close_array


                    if add_metrics_per_dfs is True:
                        self._add_trainigHistory(self.__train2_internal__metrics_toHist(df_metrics, applie_averages=False))
                    self.__train2_internal__increment_multiple_metrics(all_dfs_metrics, df_metrics)
                    del df_metrics



                    # set the calculated XY_datas for the later train and test
                    all_X_train_computed[df_key] = X_train; all_X_train_computed.save(df_key)
                    all_X_test_computed[df_key] = X_test; all_X_test_computed.save(df_key)

                    all_Y_train_computed[df_key] = Y_train; all_Y_train_computed.save(df_key)
                    all_Y_test_computed[df_key] = Y_test; all_Y_test_computed.save(df_key)


                    if add_metrics_per_dfs is True:
                        # loss computation
                        if track_dfs_train_loss is True:
                            self.__train2_internal__compute_loss_and_update_metrics(
                                X_train, Y_train, batchSizeIteration, add_metrics_per_dfs, "loss", verbose, all_dfs_metrics)
                        # val_oldLoss computation
                        if track_dfs_val_loss is True:
                            self.__train2_internal__compute_loss_and_update_metrics(
                                X_test, Y_test, batchSizeIteration, add_metrics_per_dfs, "val_loss", verbose, all_dfs_metrics)

                    # dont repare X_... no needs to

                    all_dataframes.unLoad_noSave(df_key)
                    del X_train, Y_train, X_test, Y_test, df_for_backtests # free the RAM asap

                    print()


                ## run the trainng & update the Lr
                # train on the generator
                train_history = self._base_train_generator(
                    _generator_all_XY_datas((all_X_train_computed, all_Y_train_computed), batchSize, shuffle_train_datas, 1),
                    _generator_all_XY_datas((all_X_test_computed, all_Y_test_computed), batchSize, False, 1),
                    steps_per_train_epoch=steps_per_train_epoch, steps_per_test_epoch=steps_per_test_epoch,
                    nbEpoches=1, verbose=verbose,
                )
                if add_metrics_per_dfs is False:
                    self._add_trainigHistory(train_history)
                else:
                    prefix:str = ("train_" if add_metrics_per_dfs is True else "")
                    self._add_trainigHistory({
                        (prefix + key):val
                        for (key, val) in train_history.history.items()
                    })
                    del prefix

                # get the last loss for the dynamic Lr calculation, and Y_correction
                dict_metrics:"dict[str, list[float]]" = train_history.history
                lastLoss = dict_metrics["loss"][-1]
                # checkpoints
                if correctionStepEveryLoss is not None:
                    computeNewCorrection = (lastLoss <= correctionStepEveryLoss)
                    epoches_sinceLastCp += 1
                    if (saveCheckpointEvery_CorrectionStep is not False) and (epoches_sinceLastCp >= _miniEpoches_betwinCP):
                        self.save(f"checkpoint_{self._current_checkpoint_nb}", directory=self._checkpoints_directory)
                        self._current_checkpoint_nb += 1
                        epoches_sinceLastCp = 1

                if dynamicLearningRate is True:
                    self.set_learning_rate(self._compute_new_Lr(verbose, learningRateFactor, _use_loss=lastLoss))
                del train_history, lastLoss, dict_metrics

                ## compute all the averages metrics
                suffix:str = ("_avg" if add_metrics_per_dfs is True else "")
                self._add_trainigHistory(
                    self.__train2_internal__metrics_toHist(
                        all_dfs_metrics, names_suffix=suffix, applie_averages=True
                ))
                del suffix

                print("\n\n")

        finally:
            all_X_train_computed.clear()
            all_X_test_computed.clear()
            all_dataframes_XY_saved.clear()
            all_Y_train_computed.clear()
            all_Y_test_computed.clear()

        # set as finished
        self._trainCalls_history[_index_trainCall] = (*self._trainCalls_history[_index_trainCall][ :2], True)

#######
#######
#######
#######
#######
#######
#######
#######


class Model_senti2_V7_GPT(Model_senti2_V7):
    """inheritage of the V7, but using a GTP model"""
    VERSION = "V7_GPT"
    # other class parameters are the ones form the V7

    def __init__(self,
            _nbTransformerBlocks:"int|None"=None, _nbAttentionHeads:"int|None"=None,
            _headsNbDimensions:"int|None"=None, _use_V7_transformer:bool=False,
            _network:"None|keras.Model"=None, _network_dropout:"float|None"=None,
            _more_parameters:"bool|int|float"=False, _normalization_layers:bool=False,
            fastMode:bool=True, _more_periodes:"int|bool"=False, _use_hardSig:bool=True)->None:
        """`_use_V7_transformer` select what to use at the end of the GPT transformers:\n
                True -> use the senti2_V7 model, False -> use a simple Bidirectional-LSTM\n"""
        self._popNoneSeriesSelection()
        if get_RELIABILITY_MODE() is False:
            raise ValueError("incorrect reliability mode: use enable_reliable_mode()")

        self.network:"keras.Model"
        if _network is not None:
            super().__init__(_network=_network)
        else: # network not given => create the network
            ### parameters
            network_dropout:"float|None"
            if _network_dropout is None: network_dropout = 0.10
            elif _network_dropout <= 0.0: network_dropout = None
            else: network_dropout = _network_dropout

            if _more_periodes is True:
                self.nbPeriodes *= 2
            elif _more_periodes is not False:
                self.nbPeriodes = _more_periodes

            moreParamMult:"float|int" = 1
            if isinstance(_more_parameters, (int, float)):
                moreParamMult = _more_parameters

            _T = TypeVar("_T")
            def choose(value:"_T|None", default:"_T")->"_T":
                return value if isinstance(value, type(default)) else default

            nbTransformerBlocks:int = choose(_nbTransformerBlocks, 8)
            nbAttentionHeads:int = choose(_nbAttentionHeads, 8)
            headsNbDimensions:int = choose(_headsNbDimensions, 32)
            ff_linear_size:int = 4*self.number_series_inputs

            ### create the model
            inputs = keras.Input(shape=(self.nbPeriodes, self.number_series_inputs))

            outputs = self._create_layers_of_GPT_network(
                inputs, nbTransformerBlocks, nbAttentionHeads,
                headsNbDimensions, ff_linear_size, network_dropout,
            )

            outputs = self._create_layers_GPT_to_output(
                outputs, use_fastMode=fastMode, network_dropout=network_dropout,
                use_more_parameters=(_more_parameters is not False, moreParamMult),
                use_normalization_layers=_normalization_layers, use_hardSig=_use_hardSig,
                use_V7_transformer=_use_V7_transformer,
            )

            # creat the model and init the V7 class
            super().__init__(_network=keras.Model(inputs=inputs, outputs=outputs))



    def _create_layers_of_GPT_network(self,
            inputs, nbTransformerBlocks:int, nbAttentionHeads:int, headsNbDimensions:int,
            ff_linear_size:int, network_dropout:"float|None"):
        # DROPOUT
        outputs = keras.layers.Dropout(0.1)(inputs)

        # all the transformers blocks
        outputs = _Transformers_GPT(
            nb_trasformers=nbTransformerBlocks,
            nb_heads=nbAttentionHeads,
            embeded_dims=self.number_series_inputs,
            nbDims_heads=headsNbDimensions,
            feedForward_size=ff_linear_size,
            dropoutRate=network_dropout,
        )(outputs, None)

        # NORMALIZATION (mandatory)
        outputs = keras.layers.LayerNormalization()(outputs)
        return outputs

    def _create_layers_GPT_to_output(self,
            inputs, use_fastMode:bool, use_more_parameters:"tuple[bool, float|int]",
            use_normalization_layers:bool, network_dropout:"float|None",
            use_hardSig:bool, use_V7_transformer:bool, ):
        """
        `use_V7_transformer` select what to use at the end of the GPT transformers\n
        The following parameters applie to both V7 and V7_GPT choice
        `inputs` the tensor of the inputs for the layers\n
        `use_fastMode` whether to use CUDA implementation for the layers (faster but way more memory)\n
        `use_more_parameters`:(_more_parameters is True, moreParamMult)\n
        `network_dropout` the dropout to use, /!\\ None -> disabled\n
        `use_normalization_layers` whether to use normalization layers (not significant)\n
        `use_hardSig` whether to use hard sigmoid (values clip) insted of normal sigmoid (values not clipping)"""

        # NOTE, there is 2 options:
        # option 1:
        #   use a classic reduction with lstm that only output final value (faster)
        # option 2:
        #   use a model like the senti2V7 normal (reproduced) to transform to the final value (slower)
        # -> implement both, tooglable with a parameter

        if use_V7_transformer is True:
            return super()._create_layers_of_network(
                inputs, use_fastMode=use_fastMode, use_more_parameters=use_more_parameters,
                use_normalization_layers=use_normalization_layers,
                network_dropout=network_dropout, use_hardSig=use_hardSig,
            )

        else :
            (_more_parameters, moreParamMult) = use_more_parameters

            # BIDIRECTIONAL-LSTM -> flatten the sequence
            LSTM_layer = (keras.layers.CuDNNLSTM if use_fastMode is True else keras.layers.LSTM)
            outputs = keras.layers.Bidirectional(LSTM_layer(
                units=(self.number_series_inputs // 2 if _more_parameters is False
                        else int(self.number_series_inputs * moreParamMult)),
            ), name="Bidirectional_LSTM_flattener")(inputs)
            # DROPOUT
            outputs = keras.layers.Dropout(network_dropout)(outputs)
            # NORMALIZATION
            if use_normalization_layers is True:
                outputs = keras.layers.Normalization()(outputs)

            # FEED FORWARD x2
            outputs = keras.layers.Dense(
                self.number_series_inputs, activation="linear", name="sequence_process_layer",
            )(outputs)
            outputs = keras.layers.Dense(1, activation=("hard_sigmoid" if use_hardSig is True else "sigmoid"))(outputs)

            return outputs