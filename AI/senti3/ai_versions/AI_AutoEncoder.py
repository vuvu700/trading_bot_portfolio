if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

import tensorflow as tf
import keras
import keras.layers
import keras.optimizers
import keras.callbacks
import keras.activations
import keras.backend
import keras.utils
from keras import Model as keras_model

import os
from io import StringIO
import numpy
import json
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from math import ceil


from calculationLib import (
    _SeriesGeneratorConfig, toJsonSeriesGeneratorConfig, fromJsonSeriesGeneratorConfig,
    get_default_series_generator, generate_calculationSeries, _SerieName)
from save_formats import (
    AsJson_Model_IO_config, AsJson_BaseAI,
    toJsonMultiple, fromJsonMultiple, JSON_SEMI_COMPACT_ARGS)
from paths_cfg import AI_SAVE_DIRECTORY, AI_CHECKPOINTS_DIRECTORY

import AI
from .baseAI import BaseAI, _MetricNameSingle, _MetricNameDfs
from ..datas.ai_datas_operations import add_noise, compressDatasArray
from ..datas.datas_types import (
    Datas_dataset, Datas_series_raw, Datas_series_regularized,
    Datas_array)
from ..datas.types_config import (
    ResumeConfig, _MetricName, _GlobalPlotConfig,
    PlotConfig, CustomLossFuncConfig, Aliased_BackTestField,
    TrainCallHist, LossToLrTuple, Model_IO_config,
    _resume_func_TradesDura, _resume_func_nbIter)
from ..training.lossFunctions import get_custom_loss_funcs_creator, _NormalLossFuncName
from ..optimizerFactorie import (
    keras_Optimizer,
    OptimizerFactoryConfig_Adam, AdamKwargs, 
    OptimizerFactoryConfig_SGD, SgdKwargs)
from ..modelsConfigs.layersFactory import (
    _ActivationName, AutoEncoderFactoryConfig, AutoEncodeurModel)
from ..modelsConfigs.autoencoderModels import (
    get_dense_autoencoder_V1_config, get_dense_autoencoder_V1_generic_config,
    Config_DenseAutoencoder_V1_generic, get_LSTM_autoencoder_config)
from ..utilsKeras import KerasConfig
from ..datas.ai_datas_operations import (
    compressMatrixToArray, _RegroupMethode)


from holo.types_ext import _Serie_Float, _3dArray_Float, _2dArray_Float
from holo.__typing import (
    Any, Callable, Generator, Iterable, 
    Protocol, TypeGuard, Literal, NamedTuple, 
    TypedDict, NotRequired, LiteralString,
    Concatenate, Self, TypeAlias, TypeVar,
    Generic, JsonTypeAlias, overload,
    assertIsinstance, NoReturn, cast,
    DefaultDict,
)
from holo.prettyFormats import (
    getCurrentFuncCode, prettyPrint, prettyPrintToJSON,
)
from holo.protocols import _T, _P
from holo.pointers import Pointer


DEFAULT_INPUT_SERIES: "list[_SerieName]" = [
    ... # REMOVED
    ]
DEFAULT_OUTPUT_SERIES: "list[_SerieName]" = DEFAULT_INPUT_SERIES

DEFAULT_DATAS_DTYPE = numpy.float32

DEFAULT_RESUME_CFG: "list[ResumeConfig]" = [
    ResumeConfig(metricName="train_loss", alias="lossAvgTrain"),
    ResumeConfig(metricName="train_val_loss", alias="valAvgTrain"),
        ResumeConfig(metricName="loss"),
        ResumeConfig(metricName="val_loss", alias="val"),
        ResumeConfig(metricName="loss_avg", alias="lossAvg"),
        ResumeConfig(metricName="val_loss_avg", alias="valAvg"),
    ResumeConfig("oldLoss"),
    ResumeConfig("val_oldLoss", alias="oldLossVal"),
        ResumeConfig("oldLoss_avg", alias="oldLossAvg"),
        ResumeConfig("val_oldLoss_avg", alias="val_oldLoss_avg"),
    ResumeConfig("nbIters", func=_resume_func_nbIter),
    ResumeConfig("val_nbIters", alias="nbItersVal", func=_resume_func_nbIter),
        ResumeConfig("nbIters_avg", alias="nbItersAvg", func=_resume_func_nbIter),
        ResumeConfig("val_nbIters_avg", alias="nbItersValAvg", func=_resume_func_nbIter),
    ResumeConfig("Trades%"),
    ResumeConfig("val_Trades%", alias="TradesVal%"),
        ResumeConfig("Trades%_avg", alias="Trades%Avg"),
        ResumeConfig("val_Trades%_avg", alias="TradesVal%Avg"),
    ResumeConfig("WinRate%"),
    ResumeConfig("val_WinRate%", alias="WinRateVal%"),
        ResumeConfig("WinRate%_avg", alias="WinRate%Avg"),
        ResumeConfig("val_WinRate%_avg", alias="WinRateVal%Avg"),
    ResumeConfig("TradesDura[h]", func=_resume_func_TradesDura),
    ResumeConfig("val_TradesDura[h]", alias="TradesDura[h]Val", func=_resume_func_TradesDura),
        ResumeConfig("TradesDura[h]_avg", alias="TradesDura[h]Avg", func=_resume_func_TradesDura),
        ResumeConfig("val_TradesDura[h]_avg", alias="TradesDura[h]ValAvg", func=_resume_func_TradesDura)]

DEFAULT_PLOT_CFG = PlotConfig(
        globalCfg=_GlobalPlotConfig(
            nbSubPlots=5, tight_layout=True,
            hlines=[(50.0, 3), (65.0, 3), (0.0, 2), (0.5, 2), (1.0, 2), (0.0900, 0),
                    (0.1000, 0), (0.0100, 0), (0.0200, 0), (10.0, 1), (20.0, 1)],
            yLims={2: (1.5, -1.0), 4: (10.0, 0.0)},
        ),
        metricsCfgs={
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
        })

DEFAULT_REGULARIZE_ARGS = AI.RegularizeConfig(
    preferReRangePlus=True, rescale=True, valuesRange=AI.ValuesRange(0.0, 1.0))


DEFAULT_BACKTEST_CONFIG: "list[Aliased_BackTestField]" = [
        Aliased_BackTestField("Avg. Trade [%]", "Trades%"),
        Aliased_BackTestField("Avg. Trade Duration", "TradesDura[h]"),
        Aliased_BackTestField("Win Rate [%]", "WinRate%")]

DEFAULT_LOSS_LR_TABLE: "list[LossToLrTuple]" = [
    LossToLrTuple(loss=0.08, lr=0.000_40), LossToLrTuple(loss=0.04, lr=0.000_30),
    LossToLrTuple(loss=0.03, lr=0.000_20), LossToLrTuple(loss=0.02, lr=0.000_15),
    LossToLrTuple(loss=0.015, lr=0.000_10), LossToLrTuple(loss=0.01, lr=0.000_075),
    LossToLrTuple(loss=0.0075, lr=0.000_05), LossToLrTuple(loss=0.005, lr=0.000_025)]


DEFAULT_BATCH_SIZE: int = 1024

DEFAULT_OPTIMIZER_FACTORY_CONFIG = \
    OptimizerFactoryConfig_Adam(AdamKwargs(learning_rate=0.000_20))


########################################################################


class _Implement_AutoEncoder_base(BaseAI[BaseAI._requiredSingleMetrics, BaseAI._requiredDfsMetrics]):
    """base class for auto encoder sub classes, that implement the base methodes"""
    modelFactoryConfig: "AutoEncoderFactoryConfig"
    
    def predict_senti(self, inputDF:"Datas_dataset", nbIters:int=1,
                      regroupSentiOutput:"_RegroupMethode"='first',
                      randomSentiNoise:"bool|float"=True)->"_Serie_Float":
        """predict senti serie from exact datas except for "SENTI" that will be iterated\n
        the "SENTI" serie will start as random values or """
        if self.ioConfig.nbPeriodes_inputs != 1:
            raise NotImplementedError(f"the support for multi periodes isn't done yet")
        ### TODO: adapt this function to multiple periodes
        # => the convergence don't change (this is still repeating the same In Out)
        # => the gathered final senti serie needs to be assembled
        ### setup
        inputDatas_Array: "Datas_array" = self.transformeToArrayDatas(
            fromDatas=self.transformeToRegularizedDatas(inputDF, selection="inputs"), 
            datasKind="inputs", filterSeries=True)
        inputSentiIndex: int = inputDatas_Array.getSeriesIndexes({"SENTI"})["SENTI"]
        if randomSentiNoise is not False:
            noiseShape = (inputDatas_Array.nbSamples, inputDatas_Array.nbPeriodes, )
            if randomSentiNoise is True: 
                # full random start
                inputDatas_Array.datas[:, :, inputSentiIndex] = numpy.random.random(noiseShape)
            else: # => add noise
                inputDatas_Array.datas[:, :, inputSentiIndex] += \
                    numpy.random.normal(loc=0.0, scale=randomSentiNoise, size=noiseShape)
                numpy.clip(inputDatas_Array.datas[:, :, inputSentiIndex], 
                           0.0, 1.0, out=inputDatas_Array.datas[:, :, inputSentiIndex])
            if self.ioConfig.valuesRange != AI.ValuesRange(0, 1):
                AI.moveRange(inputDatas_Array.datas[:, :, inputSentiIndex], 
                             valuesRange=self.ioConfig.valuesRange)                
            
        ### do the iterations
        for iterStep in range(1, nbIters+1):
            # predict the output
            print(f"\riteration n°{iterStep}", end="")
            outputDatas_array = self.predict(inputDatas=inputDatas_Array, verbose=0)
            outputSentiIndex: int = outputDatas_array.getSeriesIndexes({"SENTI"})["SENTI"]
            # transfert the "SENTI" serie, back to the inputs
            inputDatas_Array.datas[:, :, inputSentiIndex] = \
                outputDatas_array.datas[:, :, outputSentiIndex]
        else: print()
        # note: only single periode datas
        return compressMatrixToArray(
            datasMatrix=inputDatas_Array.datas[:, :, inputSentiIndex],
            compressionMethode=regroupSentiOutput)


class AI_Dense_AutoEncoder(_Implement_AutoEncoder_base):
    """an autoencoder for the AI"""
    __prettyAttrs__ = BaseAI.__prettyAttrs__
    
    def __init__(self,
            seriesGeneratorConfig:"_SeriesGeneratorConfig",
            config: "Config_DenseAutoencoder_V1_generic", 
            batchSizes:"int|None"=None) -> None:
        self.model: "AutoEncodeurModel"
        # finalize the io config for the model
        model_io_config = Model_IO_config(
            nbPeriodes_inputs=1, nbPeriodes_outputs=1,
            outputPeriodesShift=0,
            input_series=DEFAULT_INPUT_SERIES,
            output_series=DEFAULT_OUTPUT_SERIES,
            datasDtype=DEFAULT_DATAS_DTYPE, 
            regularizeSeriesConfig=DEFAULT_REGULARIZE_ARGS)
        # get the model to use 
        assert model_io_config.valuesRange == AI.ValuesRange(0, 1)
        modelFactoryConfig: "AutoEncoderFactoryConfig" = \
            get_dense_autoencoder_V1_generic_config(
                models_io=model_io_config, outputRange="hard_sigmoid", 
                modelDtype="float32", **config)
        # use the correct batchSizes
        if batchSizes is None:
            batchSizes = DEFAULT_BATCH_SIZE
        # init the ai
        super().__init__(
            seriesGeneratorConfig=seriesGeneratorConfig,
            ioConfig=model_io_config,
            resumeCfg=DEFAULT_RESUME_CFG,
            plotCfg=DEFAULT_PLOT_CFG,
            loss_func_config="mean_squared_error",
            modelFactoryConfig=modelFactoryConfig,
            backtests_fields_config=DEFAULT_BACKTEST_CONFIG,
            lossToLearningRate_table=DEFAULT_LOSS_LR_TABLE,
            optimizerFactoryConfig=DEFAULT_OPTIMIZER_FACTORY_CONFIG,
            preTrainedAIs={},
            batch_size=batchSizes, 
            kerasConfig=KerasConfig(tf_v1_compat=False, memoryGrowth=True, toRevert=False),
            metricsHistory=BaseAI.getEmptyRequiredMetrics(checkDfKey=None))





DEFAULT_NB_PERIODES: int = 6

class AI_LSTM_AutoEncoder(_Implement_AutoEncoder_base):
    """an autoencoder for the AI"""
    __prettyAttrs__ = BaseAI.__prettyAttrs__
    
    modelFactoryConfig: "AutoEncoderFactoryConfig"
    def __init__(self,
            seriesGeneratorConfig:"_SeriesGeneratorConfig",
             nbPeriodes:"int|None", batchSizes:"int|None"=None) -> None:
        self.model: "AutoEncodeurModel"
        # finalize the io config for the model
        if nbPeriodes is None:
            nbPeriodes = DEFAULT_NB_PERIODES
        
        model_io_config = Model_IO_config(
            nbPeriodes_inputs=nbPeriodes,
            nbPeriodes_outputs=nbPeriodes,
            outputPeriodesShift=0,
            input_series=DEFAULT_INPUT_SERIES,
            output_series=DEFAULT_OUTPUT_SERIES,
            datasDtype=DEFAULT_DATAS_DTYPE,
            regularizeSeriesConfig=DEFAULT_REGULARIZE_ARGS)
        # get the model to use 
        assert model_io_config.valuesRange == AI.ValuesRange(0, 1)
        modelFactoryConfig: "AutoEncoderFactoryConfig" = get_LSTM_autoencoder_config(
                models_io=model_io_config, outputRange="hard_sigmoid",
                modelDtype="float32")
        # use the correct batchSizes
        if batchSizes is None:
            batchSizes = DEFAULT_BATCH_SIZE
        # init the ai
        super().__init__(
            seriesGeneratorConfig=seriesGeneratorConfig,
            ioConfig=model_io_config,
            resumeCfg=DEFAULT_RESUME_CFG,
            plotCfg=DEFAULT_PLOT_CFG,
            loss_func_config="mean_squared_error",
            modelFactoryConfig=modelFactoryConfig,
            backtests_fields_config=DEFAULT_BACKTEST_CONFIG,
            lossToLearningRate_table=DEFAULT_LOSS_LR_TABLE,
            optimizerFactoryConfig=DEFAULT_OPTIMIZER_FACTORY_CONFIG,
            preTrainedAIs={},
            batch_size=batchSizes, 
            kerasConfig=KerasConfig(tf_v1_compat=False, memoryGrowth=True, toRevert=False),
            metricsHistory=BaseAI.getEmptyRequiredMetrics(checkDfKey=None))