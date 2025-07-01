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
    get_default_series_generator, generate_calculationSeries, _SerieName, )
from save_formats import (
    AsJson_Model_IO_config, AsJson_BaseAI,
    toJsonMultiple, fromJsonMultiple, JSON_SEMI_COMPACT_ARGS, )
from paths_cfg import AI_SAVE_DIRECTORY, AI_CHECKPOINTS_DIRECTORY

import AI
from .baseAI import PreTrainedModelArgs
from .baseExtends import AI_SupportsConvergenceTrain
from .AI_AutoEncoder import AI_Dense_AutoEncoder, AI_LSTM_AutoEncoder
from ..datas.ai_datas_operations import add_noise, compressDatasArray
from ..datas.datas_types import (
    Datas_dataset, Datas_series_raw, Datas_series_regularized,
    Datas_array, )
from ..datas.types_config import (
    ResumeConfig, _MetricName, CustomLossFuncConfig, BacktestsResultsStorage,
    TrainCallHist, LossToLr, LossToLr_table, Model_IO_config,
    _resume_func_TradesDura, _resume_func_nbIter, )
from ..training.lossFunctions import get_custom_loss_funcs_creator, _NormalLossFuncName
from ..optimizerFactorie import (
    keras_Optimizer,
    OptimizerFactoryConfig_Adam, AdamKwargs, 
    OptimizerFactoryConfig_SGD, SgdKwargs, )
from ..modelsConfigs.layersFactory import _ActivationName, ModelFactoryConfig
from ..modelsConfigs.senti3Models import (
    get_ModelV1_config, get_GPT_config, 
    get_OLD_config, get_ModelV2_config,
    get_ModelV3_config, get_ModelV3fixed_config,
    get_TemporalConv_config, )
from ..utilsKeras import KerasConfig


from holo.types_ext import _Serie_Float, _3dArray_Float, _2dArray_Float
from holo.__typing import (
    Any, Callable, Generator, Iterable, 
    Protocol, TypeGuard, Literal, NamedTuple, 
    TypedDict, NotRequired, LiteralString,
    Concatenate, Self, TypeAlias, TypeVar,
    Generic, JsonTypeAlias, overload, override,
    assertIsinstance, NoReturn, cast,
    DefaultDict, Union, Never)
from holo.prettyFormats import (
    getCurrentFuncCode, prettyPrint, prettyPrintToJSON, )
from holo.protocols import _T, _P


DEFAULT_INPUT_SERIES: "list[_SerieName]" = [
    ... # REMOVED
    ]
DEFAULT_OUTPUT_SERIES: "list[_SerieName]" = ["SENTI", "RSI", "CLOSE_n"]

DEFAULT_NB_PERIODES_INPUTS: int = 32
DEFAULT_NB_PERIODES_OUTPUTS: int = 8

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


DEFAULT_BACKTESTS_CONFIG: "BacktestsResultsStorage" = BacktestsResultsStorage(
    fieldsAliases={Aliased_BackTestField("avgTradeReturn", "TradesReturn"),
                   Aliased_BackTestField("avgTradeDuration", "TradesDura[periodes]"),
                   Aliased_BackTestField("winRate", "WinRate%")})

DEFAULT_LOSS_LR_TABLE: "LossToLr_table" = LossToLr_table([
    LossToLr(loss=0.08, lr=0.000_40), LossToLr(loss=0.04, lr=0.000_30),
    LossToLr(loss=0.03, lr=0.000_20), LossToLr(loss=0.02, lr=0.000_15),
    LossToLr(loss=0.015, lr=0.000_10), LossToLr(loss=0.01, lr=0.000_075),
    LossToLr(loss=0.0075, lr=0.000_05), LossToLr(loss=0.005, lr=0.000_025)])


DEFAULT_BATCH_SIZE: int = 1024

DEFAULT_OPTIMIZER_FACTORY_CONFIG = \
    OptimizerFactoryConfig_Adam(AdamKwargs(learning_rate=0.000_20))


########################################################################


class AI_TESTING(AI_SupportsConvergenceTrain[Never, Never]):
    """testing version of the AI meant to test some functionalities"""
    
    modelFactoryConfig: "ModelFactoryConfig"
    def __init__(self,
            seriesGeneratorConfig:"_SeriesGeneratorConfig", 
            nbPeriodes_inputs:"int|None", nbPeriodes_outputs:"int|None", batchSizes:"int|None"=None,
            lossFunc:"_NormalLossFuncName|CustomLossFuncConfig|None"=None) -> None:
        # use the correct nb of input periodes
        if nbPeriodes_inputs is None:
            nbPeriodes_inputs = DEFAULT_NB_PERIODES_INPUTS
        # use the correct nb of output periodes
        if nbPeriodes_outputs is None:
            nbPeriodes_outputs = DEFAULT_NB_PERIODES_OUTPUTS
        # finalize the io config for the model    
        model_io_config = Model_IO_config(
            nbPeriodes_inputs=nbPeriodes_inputs,
            nbPeriodes_outputs=nbPeriodes_outputs,
            outputPeriodesShift=nbPeriodes_inputs,
            input_series=DEFAULT_INPUT_SERIES,
            output_series=DEFAULT_OUTPUT_SERIES,
            datasDtype=DEFAULT_DATAS_DTYPE, 
            regularizeSeriesConfig=DEFAULT_REGULARIZE_ARGS)
        # select the loss function to use
        if lossFunc is None:
            if model_io_config.nbPeriodes_inputs == 1:
                lossFunc = "mean_squared_error"
            else: # => more than one periode
                lossFunc = CustomLossFuncConfig("fixPeriodesDistances", {})
        
        # get the model to use 
        assert model_io_config.valuesRange == AI.ValuesRange(0, 1)
        modelFactoryConfig: "ModelFactoryConfig" = get_ModelV3_config(
                models_io=model_io_config, outputRange="hard_sigmoid", 
                nbTransformers=8, modelDtype="float32")
        # use the correct batchSizes
        if batchSizes is None:
            batchSizes = DEFAULT_BATCH_SIZE
        # init the ai
        super().__init__(
            seriesGeneratorConfig=seriesGeneratorConfig,
            ioConfig=model_io_config,
            resumeCfg=DEFAULT_RESUME_CFG,
            plotCfg=DEFAULT_PLOT_CFG,
            loss_func_config=lossFunc,
            modelFactoryConfig=modelFactoryConfig,
            backtestsResults=DEFAULT_BACKTESTS_CONFIG,
            lossToLearningRate_table=DEFAULT_LOSS_LR_TABLE,
            optimizerFactoryConfig=DEFAULT_OPTIMIZER_FACTORY_CONFIG,
            batch_size=batchSizes, 
            kerasConfig=KerasConfig(tf_v1_compat=False, memoryGrowth=True, toRevert=False),
            preTrainedAIs={},
            metricsHistory=self.getEmptyRequiredMetrics(checkDfKey=None))



class AI_TESTING2(AI_SupportsConvergenceTrain[Never, Never]):
    """testing version of the AI meant to test some functionalities"""
    
    modelFactoryConfig: "ModelFactoryConfig"
    def __init__(self,
            seriesGeneratorConfig:"_SeriesGeneratorConfig", 
            autoencodeurAI:"PreTrainedModelArgs[AI_Dense_AutoEncoder]", nbTransformersUnits:int, 
            nbPeriodes_inputs:"int|None", nbPeriodes_outputs:"int|None", batchSizes:"int|None"=None,
            lossFunc:"_NormalLossFuncName|CustomLossFuncConfig|None"=None) -> None:
        # use the correct nb of input periodes
        if nbPeriodes_inputs is None:
            nbPeriodes_inputs = DEFAULT_NB_PERIODES_INPUTS
        # use the correct nb of output periodes
        if nbPeriodes_outputs is None:
            nbPeriodes_outputs = DEFAULT_NB_PERIODES_OUTPUTS
        # finalize the io config for the model    
        model_io_config = Model_IO_config(
            nbPeriodes_inputs=nbPeriodes_inputs,
            nbPeriodes_outputs=nbPeriodes_outputs,
            outputPeriodesShift=nbPeriodes_inputs,
            input_series=DEFAULT_INPUT_SERIES,
            output_series=DEFAULT_OUTPUT_SERIES,
            datasDtype=DEFAULT_DATAS_DTYPE, 
            regularizeSeriesConfig=DEFAULT_REGULARIZE_ARGS)
        # select the loss function to use
        if lossFunc is None:
            if model_io_config.nbPeriodes_inputs == 1:
                lossFunc = "mean_squared_error"
            else: # => more than one periode
                lossFunc = CustomLossFuncConfig("fixPeriodesDistances", {})
        
        # get the model to use 
        assert model_io_config.valuesRange == AI.ValuesRange(0, 1)
        modelFactoryConfig: "ModelFactoryConfig" = get_ModelV3fixed_config(
                models_io=model_io_config, nbTransformersUnits=nbTransformersUnits, 
                preTrainedAutoencodeurKey="autoencodeur", nbTransformers=4,
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
            loss_func_config=lossFunc,
            modelFactoryConfig=modelFactoryConfig,
            backtestsResults=DEFAULT_BACKTESTS_CONFIG,
            lossToLearningRate_table=DEFAULT_LOSS_LR_TABLE,
            optimizerFactoryConfig=DEFAULT_OPTIMIZER_FACTORY_CONFIG,
            batch_size=batchSizes, 
            kerasConfig=KerasConfig(
                tf_v1_compat=False, memoryGrowth=True, toRevert=False),
            preTrainedAIs={"autoencodeur": autoencodeurAI},
            metricsHistory=self.getEmptyRequiredMetrics(checkDfKey=None))