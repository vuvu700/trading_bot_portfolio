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

import attrs
import os
from io import StringIO
import numpy
import json
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


from calculationLib import (
    _SeriesGeneratorConfig, toJsonSeriesGeneratorConfig, fromJsonSeriesGeneratorConfig,
    get_default_series_generator, generate_calculationSeries, _SerieName, )
import AI
from paths_cfg import AI_SAVE_DIRECTORY, AI_CHECKPOINTS_DIRECTORY
from ..datas.ai_datas_operations import compressDatasArray

from ..datas.datas_types import (
    _T_DfKey,
    Datas_dataset, Datas_series_raw, Datas_series_regularized,
    Datas_array, Datas_InputOutput, Datas_training_Generator, )
from ..datas.types_config import (
    Metrics, MetricsDfs, ResumeConfig, FittingHistory,
    CustomLossFuncConfig, BacktestsResultsStorage,
    TrainCallHist, LossToLr, LossToLr_table, Model_IO_config, AllMetrics, 
    _MetricName, _MetricNameSingle, _MetricNameDfs, _DfKey, )
from ..training.lossFunctions import get_custom_loss_funcs_creator, _NormalLossFuncName
from ..optimizerFactorie import _OptimizerFactoryConfig, keras_Optimizer
from ..modelsConfigs.layersFactory import GenericModelFactoryConfig
from ..utilsKeras import (
    KerasConfig, load_keras_model, save_keras_model, get_saveModel_fullPath, )
from save_formats import (
    AsJson_BaseAI, AsJson_CustomLossFuncConfig, AsJson_LoadModelArgs,
    toJsonMultiple, fromJsonMultiple, JSON_SEMI_COMPACT_ARGS, )
from ..training.callbacks import MesureFittingTimes
from ..datas.convergence import ConvergeConfig, ConvergeParams, computeConvergence
from metricsPloter.config import PlotConfig

from holo.types_ext import _Serie_Float, _3dArray_Float, _2dArray_Float
from holo.__typing import (
    Any, Callable, Generator, Iterable, 
    Protocol, TypeGuard, Literal,
    TypedDict, NotRequired, LiteralString,
    Concatenate, Self, TypeAlias, TypeVar,
    Generic, JsonTypeAlias, getLiteralArgs, Union,
    assertIsinstance, NoReturn, cast, assertListSubType,
    DefaultDict, get_args, ClassFactory, )
from holo.prettyFormats import (
    getCurrentFuncCode, prettyPrintToJSON, print_exception,
    prettyPrint, PrettyfyClass, PrettyPrint_CompactArgs, )
from holo.protocols import _T, _P
from holo.pointers import Pointer


############################################


_AI_Verbose = Literal['auto', 'disable', 'oneLine', 'progressBar']
_T_AI = TypeVar("_T_AI", bound="BaseAI")


def _getNotImplementedError(unsupportedType:"type")->NotImplementedError:
    return NotImplementedError(f"no {getCurrentFuncCode(depth=2).co_name} procedures implemented for {unsupportedType}")

def convertVerbose(verbose:"_AI_Verbose")->"Literal['auto', 0, 1, 2]":
    if verbose == "auto": return "auto"
    elif verbose == "disable": return 0
    elif verbose == "progressBar": return 1
    elif verbose == "oneLine": return 2
    else: raise ValueError(f"invalide keras verbosity: {repr(verbose)}")

@attrs.frozen
class _LoadedAI_initArgs_model():
    model: "keras.models.Model"
    optimizer: "keras_Optimizer"

@attrs.frozen
class _LoadedAI_initArgs():
    modelArgs: "_LoadedAI_initArgs_model|None"
    trainingCallsHistory: "list[TrainCallHist]"
    comments: "list[str]"
    fittingHistory: "FittingHistory"
    nbEpochesFinished: "int"

@attrs.frozen
class PreTrainedModelArgs(Generic[_T_AI]):
    ai: "_T_AI"
    savedAtPath: "Path"
    
    def toJson(self)->"AsJson_LoadModelArgs[_T_AI]":
        return AsJson_LoadModelArgs(
            cls=self.__class__.__name__,
            ai_cls=self.ai.__class__.__name__,
            configFilePath=self.savedAtPath.as_posix())
    
    @classmethod
    def fromJson(cls, datas:"AsJson_LoadModelArgs[_T_AI]")->"PreTrainedModelArgs[_T_AI]":
        assert cls.__name__ == datas["cls"]
        preTrainedModelPath = Path(datas["configFilePath"])
        loadedAI = BaseAI.load_dispatch(
            clsName=datas["ai_cls"], configFilePath=preTrainedModelPath)
        preTrainedModelArgs = PreTrainedModelArgs.__new__(cls)
        PreTrainedModelArgs.__init__(
            self=preTrainedModelArgs,
            ai=loadedAI, savedAtPath=preTrainedModelPath)
        return preTrainedModelArgs




class SupportsMetrics(Protocol[_MetricNameSingle, _MetricNameDfs]):
    metricsHistory: "AllMetrics[_MetricNameSingle, _MetricNameDfs]"
    
    @classmethod
    def getRequiredSingleMetrics(cls)->"set[_MetricNameSingle]":
        ... # return set(get_args(BaseAI._requiredSingleMetrics))
    @classmethod
    def getRequiredDfsMetrics(cls)->"set[_MetricNameDfs]":
        ... # return set(get_args(BaseAI._requiredDfsMetrics))

    @classmethod
    def getEmptyRequiredMetrics(
            cls, *, checkDfKey:"Callable[[str], bool]|None")->"AllMetrics[_MetricNameSingle, _MetricNameDfs]":
        return AllMetrics(
            singleMetricNames=cls.getRequiredSingleMetrics(),
            dfsMetricNames=cls.getRequiredDfsMetrics(), 
            checkDfKey=checkDfKey)


class BaseAI(SupportsMetrics[_MetricNameSingle, _MetricNameDfs], PrettyfyClass):
    # __slots__ = ( # currently commented because it makes the type hinting bug ....
    #     "seriesGeneratorConfig", "ioConfig", "resumeCfg", "plotCfg",
    #     "loss_func_config", "backtests_config", "backtests_fields_config", 
    #     "lossToLearningRate_table", "modelFactoryConfig", "model", "optimizerFactoryConfig", 
    #     "optimizer", "trainingCallsHistory", "comments", "kerasConfig", "metricsHistory", )
    
    __SUB_CLASSES: "dict[str, type[BaseAI]]" = {}
    
    def __init_subclass__(cls, **kwargs)->None:
        super().__init_subclass__(**kwargs)
        ClassFactory._ClassFactory__registerFactoryUser(cls)
        BaseAI.__SUB_CLASSES[cls.__name__] = cls
    
    def __init__(self,
            ioConfig:"Model_IO_config", 
            seriesGeneratorConfig:"_SeriesGeneratorConfig",  
            plotCfg:"PlotConfig", resumeCfg:"list[ResumeConfig]", 
            loss_func_config:"CustomLossFuncConfig|_NormalLossFuncName",
            backtestsResults:"BacktestsResultsStorage",
            lossToLearningRate_table:"LossToLr_table",
            modelFactoryConfig:"GenericModelFactoryConfig",
            optimizerFactoryConfig:"_OptimizerFactoryConfig",
            kerasConfig:"KerasConfig", batch_size:int,
            preTrainedAIs:"dict[str, PreTrainedModelArgs]",
            metricsHistory:"AllMetrics[_MetricNameSingle, _MetricNameDfs]",
            *, # use the following kwargs when loading a config
            _loadedAI_args:"_LoadedAI_initArgs|None"=None)->None:
        self.kerasConfig: "KerasConfig" = kerasConfig
        self.kerasConfig.setConfig()
        # set the parameters of the AI
        self.seriesGeneratorConfig: "_SeriesGeneratorConfig" = seriesGeneratorConfig
        self.ioConfig: "Model_IO_config" = ioConfig
        self.resumeCfg: "list[ResumeConfig]" = resumeCfg
        self.plotCfg: "PlotConfig" = plotCfg
        self.loss_func_config: "CustomLossFuncConfig|_NormalLossFuncName" = loss_func_config
        self.backtestsResults: "BacktestsResultsStorage" = backtestsResults
        self.lossToLearningRate_table: "LossToLr_table" = lossToLearningRate_table
        self.modelFactoryConfig: "GenericModelFactoryConfig" = modelFactoryConfig
        self.optimizerFactoryConfig: "_OptimizerFactoryConfig" = optimizerFactoryConfig
        self.batch_size: int = batch_size
        
        # ensure the seriesGeneratorConfig can generate the inputs
        _missingSeriesGenerators: "set[_SerieName]" = \
            set(self.ioConfig.input_series).difference(self.seriesGeneratorConfig.keys())
        if len(_missingSeriesGenerators) != 0:
            raise KeyError("the following series gernerator are missing in order to "
                           f"generate the input series: {_missingSeriesGenerators}")
        del _missingSeriesGenerators
        
        # create what needs to be created
        modelsArgs: "_LoadedAI_initArgs_model|None" = \
            (None if _loadedAI_args is None else _loadedAI_args.modelArgs)
        
        
        self.preTrainedAIs: "dict[str, PreTrainedModelArgs]" = preTrainedAIs
        self.model: "keras.models.Model" = \
            (modelFactoryConfig.build(preTrainedModels={
                name: args.ai.model for name, args in preTrainedAIs.items()})
             if modelsArgs is None else modelsArgs.model)
        self.optimizer: "keras_Optimizer" = \
            (optimizerFactoryConfig.createOptimizer() if modelsArgs is None
             else modelsArgs.optimizer)
        self.__isCompiled: bool = (modelsArgs is not None)

        # datas stored
        self.trainingCallsHistory: "list[TrainCallHist]" = \
            ([] if _loadedAI_args is None else _loadedAI_args.trainingCallsHistory)
        self.fittingHistory: "FittingHistory" = \
            (FittingHistory() if _loadedAI_args is None else _loadedAI_args.fittingHistory)
        self.metricsHistory: "AllMetrics[_MetricNameSingle, _MetricNameDfs]"
        self.metricsHistory = metricsHistory
        self.comments: "list[str]" = \
            ([] if _loadedAI_args is None else _loadedAI_args.comments)
        self.nbEpochesFinished: int = \
            (0 if _loadedAI_args is None else _loadedAI_args.nbEpochesFinished)

        # compile the ai so it is easyer to use
        self.compile(build=True)
        # revert the keras config
        self.kerasConfig.revert()
    
    
    @property
    def learningRate(self)->float:
        return self.model.optimizer.learning_rate.numpy()
    
    @learningRate.setter
    def learningRate(self, value:float)->None:
        keras.backend.set_value(self.model.optimizer.learning_rate, value)
    
    
    def toJson(self, *, _saveModel_fullPath:"Path|None"=None)->"AsJson_BaseAI":
        """convert the ai's config to a json format\n
        `_saveModel_fullPath` is an internal parameter that refer to where the model has been saved"""
        saveModel_fullPath: "str|None" = \
            (None if _saveModel_fullPath is None
             else _saveModel_fullPath.as_posix())
        loss_func_config: "str|AsJson_CustomLossFuncConfig" = \
            (self.loss_func_config if isinstance(self.loss_func_config, str) 
             else self.loss_func_config.toJson())
        return AsJson_BaseAI(
            cls=self.__class__.__name__,
            seriesGeneratorConfig=toJsonSeriesGeneratorConfig(self.seriesGeneratorConfig),
            ioConfig=self.ioConfig.toJson(),
            preTrainedModels= \
                {modelName: preTrainedModelArgs.toJson()
                 for modelName, preTrainedModelArgs in self.preTrainedAIs.items()},
            resumeCfg=toJsonMultiple(self.resumeCfg),
            plotCfg=self.plotCfg.toJson(),
            loss_func_config=loss_func_config,
            backtestsResults=self.backtestsResults.toJson(),
            lossToLearningRate_table=self.lossToLearningRate_table.toJson(),
            modelFactoryConfig=self.modelFactoryConfig.toJson(),
            modelFullPath=saveModel_fullPath,
            optimizerFactoryConfig=self.optimizerFactoryConfig.toJson(),
            #"optimizer": is saved with the model,
            trainingCallsHistory=toJsonMultiple(self.trainingCallsHistory),
            comments=list(self.comments),
            kerasConfig=self.kerasConfig.toJson(),
            batch_size=self.batch_size, 
            metricsHistory=self.metricsHistory.toJson(), 
            fittingHistory=self.fittingHistory.toJson(), 
            nbEpochesFinished=self.nbEpochesFinished)
    
    @classmethod
    def fromJson(cls, datas:"AsJson_BaseAI", 
                 preTrainedModels:"dict[str, PreTrainedModelArgs]|None"=None)->"Self":
        """return an ai initialized with the BaseAI init, \
            further inits needs to be done by the sub class"""
        assert cls.__name__ == datas["cls"]

        loss_func_config: "_NormalLossFuncName|CustomLossFuncConfig"
        if isinstance(datas["loss_func_config"], str):
            loss_func_config = datas["loss_func_config"]
        else: loss_func_config = CustomLossFuncConfig.fromJson(datas["loss_func_config"])

        modelsToLoad: "dict[str, AsJson_LoadModelArgs]" = {}
        if preTrainedModels is None:
            # => use the models of the datas
            
            modelsToLoad.update()
            preTrainedModels = {}
        else: # => some models have been given
            # check that all the models given are needed
            notNeededModels: "set[str]" = \
                set().difference(datas["preTrainedModels"].keys())
            if len(notNeededModels) != 0: 
                raise ValueError(f"the following pretrained models aren't needed: {notNeededModels}")
            # => all the models in preTrainedModels are needed
            for modelName, loadModelArgs in datas["preTrainedModels"].items():
                if modelName in preTrainedModels.keys():
                    # => pre-trained model alredy given
                    assert (type(preTrainedModels[modelName]).__name__ == loadModelArgs["ai_cls"]), \
                        TypeError(f"the type of teh pre trained model: {modelName} don't correspond: "
                                  f"got {type(preTrainedModels[modelName]).__name__} but expected {loadModelArgs['ai_cls']}")
                    assert (preTrainedModels[modelName].savedAtPath.as_posix() == loadModelArgs["configFilePath"]), \
                        ValueError(f"the paths don't correspond for the model: {modelName}, "
                                   f"got {preTrainedModels[modelName].savedAtPath.as_posix()} but expected {loadModelArgs['configFilePath']}")
                    continue # => the model don't needs to be loaded
                # => the model needs to be loaded
                modelsToLoad[modelName] = loadModelArgs
            # => added all the models that arn't given in `preTrainedModels`
            
        # load the pre trained models
        preTrainedModels = preTrainedModels.copy()
        # => use a copy to keep the dict of the arg untouched
        preTrainedModels = {
            modelName: PreTrainedModelArgs.fromJson(preTrainedModelArgs)
            for modelName, preTrainedModelArgs in modelsToLoad.items()}
        
        ### TODO: 
        #   - load the preTrained models to create the model
        #   - addapt the asJson of BaseAI
        #   - try to add some dropout on the autoencoder to improve the stats ? 

        modelArgs: "_LoadedAI_initArgs_model|None"
        model: "keras.Model|None"
        optimizer: "keras_Optimizer|None"
        kerasConfig: "KerasConfig|None"
        if datas["modelFullPath"] is None:
            # => model wasn't saved
            modelArgs = None
            kerasConfig = None
        else: # => model(and optimizer) was saved
            # load keras config
            kerasConfig = KerasConfig.fromJson(datas["kerasConfig"])
            kerasConfig.setConfig()
            # load the model
            customObjects: "dict[str, Any]" = {}
            saveModelFullPath:Path = Path(datas["modelFullPath"])
            if not isinstance(loss_func_config, str):
                # => custom loss func
                customObjects[loss_func_config.functionName] = \
                    loss_func_config.createLossFunction()
            model = load_keras_model(
                saveFile_fullPath=saveModelFullPath,
                customObjects=customObjects) # no kwnon custom objects for now
            optimizer = assertIsinstance(keras_Optimizer, model.optimizer)
            # => optimizer is packed with => the model has been compiled
            modelArgs = _LoadedAI_initArgs_model(model=model, optimizer=optimizer)

        ai = BaseAI.__new__(cls)
        BaseAI.__init__(
            self=ai,
            seriesGeneratorConfig=fromJsonSeriesGeneratorConfig(datas["seriesGeneratorConfig"]),
            ioConfig=Model_IO_config.fromJson(datas["ioConfig"]),
            resumeCfg=fromJsonMultiple(datas["resumeCfg"], ResumeConfig, ResumeConfig),
            plotCfg=PlotConfig.fromJson(datas["plotCfg"]),
            loss_func_config=loss_func_config,
            backtestsResults=BacktestsResultsStorage.fromJson(datas["backtestsResults"]),
            lossToLearningRate_table=LossToLr_table.fromJson(datas["lossToLearningRate_table"]),
            modelFactoryConfig=GenericModelFactoryConfig.dispatchFromJson(datas["modelFactoryConfig"]),
            optimizerFactoryConfig=_OptimizerFactoryConfig.fromJson(datas["optimizerFactoryConfig"]),
            kerasConfig=(kerasConfig or KerasConfig.fromJson(datas["kerasConfig"])),
            metricsHistory=AllMetrics.fromJson(datas["metricsHistory"]),
            batch_size=datas["batch_size"],
            preTrainedAIs=preTrainedModels,
            _loadedAI_args=_LoadedAI_initArgs(
                modelArgs=modelArgs,
                trainingCallsHistory=fromJsonMultiple(datas["trainingCallsHistory"], TrainCallHist, TrainCallHist),
                comments=datas["comments"],
                fittingHistory=FittingHistory.fromJson(datas["fittingHistory"]),
                nbEpochesFinished=datas["nbEpochesFinished"]))
        return ai
    
    def compile(self, build:bool=True)->None:
        lossFunc:"Callable|_NormalLossFuncName"
        if isinstance(self.loss_func_config, str):
            lossFunc = self.loss_func_config
        else: lossFunc = self.loss_func_config.createLossFunction()
        self.model.compile(optimizer=self.optimizer, loss=lossFunc)
        if build is True:
            self.model.build((None, self.ioConfig.nbPeriodes_inputs, self.ioConfig.nbFeatures_inputs))
        self.__isCompiled = True
        
    
    
    def save(self, configFilePath:Path, directory:"Path|None")->None:
        """save the model and the configuration at `directory`/`configFilePath`\n
        if the `directory` is not given, use the default directory from path_cfg.py\n
        `configFilePath` will automaticaly add '.json' at the end\n
        the model will be saved in the same directory as `configFilePath` \
            and use the same name (without the .json)"""
        if self.__isCompiled is False:
            raise RuntimeError("the ai must be compiled before being saved "
                               +"( in order to save the optimizer and the lossfunc)")
        if directory is None: 
            directory = AI_SAVE_DIRECTORY
        if configFilePath.name.endswith(".json") is False:
            # => don't ends with '.json' => add '.json'
            configFilePath = configFilePath.with_name(
                configFilePath.name + ".json")
        # => configFilePath ends with '.json'
        saveModelFileName = configFilePath.name[: -len(".json")]
        saveModelDirectory = configFilePath.parent
        
        saveModel_fullPath = save_keras_model(
            model=self.model, filename=saveModelFileName, 
            directory=saveModelDirectory)
        
        jsonDatas: "AsJson_BaseAI" = self.toJson(
            _saveModel_fullPath=saveModel_fullPath)

        with open(configFilePath, mode="w") as configFile:
            prettyPrintToJSON(
                jsonDatas, stream=configFile, end=None,
                indentSequence=" "*2, 
                compact=JSON_SEMI_COMPACT_ARGS)
    
    @staticmethod
    def load_dispatch(clsName:str, configFilePath:"Path")->"BaseAI":
        return BaseAI.__SUB_CLASSES[clsName].load(configFilePath=configFilePath)
    
    @classmethod
    def load(cls, configFilePath:Path)->"Self":
        """load the model\n
        `configFilePath` is the path of the config file of the model\n
        assert that the version of the model is compatible with the classe choosen"""
        if configFilePath.name.endswith(".json") is False:
            # => add '.json'
            configFilePath = configFilePath.with_name(
                configFilePath.name + ".json")
        # => configFilePath's name endswith '.json'

        # read the datas
        with open(configFilePath, mode="r") as configFile:
            datas: "AsJson_BaseAI" = json.load(configFile)
        
        # load the AI from the config's datas
        return cls.fromJson(datas)
    
    def copy(self, copyModel:bool=True)->"Self":
        """create a new instance of self, and if asked, copy the exact model in the new ai, \
            otherwise it will have a new model with the same achitecture"""
        newAi: "Self" = self.__class__.fromJson(self.toJson())
        if copyModel is True:
            newAi.model = self.model
            newAi.optimizer = self.optimizer
        return newAi
    
    def show_inputs_outputs_series(self)->None:
        inputsSeries:"list[_SerieName]" = self.ioConfig.input_series
        print(f"inputsSeries({len(inputsSeries)}):", inputsSeries)
        outputsSeries:"list[_SerieName]" = self.ioConfig.output_series
        print(f"outputsSeries({len(outputsSeries)}):", outputsSeries)

    def show_network_layers_infos(
            self, verbose:Literal["summary", "layers shapes", "layers config", "all"])->None:
        if verbose in ("summary", "all"):
            self.model.summary(expand_nested=True, line_length=106)
        if verbose in ("layers shapes", "all"):
            print("`name`: `input shape` -> `output shape`")
            for layer in self.model.layers:
                print(f"{layer.name}: {layer.input_shape} -> {layer.output_shape} ({layer.count_params():_d} params)")
        if verbose in ("layers config", "all"):
            self.printModelArchitecture()
    
    def printModelArchitecture(self)->None:
            prettyPrintToJSON(
                self.modelFactoryConfig.toJson(), specificCompact={dict}, 
                compact=PrettyPrint_CompactArgs(keepReccursiveCompact=False))
    
    def plot_model(self,
            to_file:"str|Literal[False]", show_shapes:bool=True, show_dtype:bool=True,
            show_layer_names:bool=True, expand_nested:bool=True,
            show_layer_activations:bool=True, dpi:int=100):
        filname:str = ("~toDelModelImage.png" if to_file is False else to_file)
        image_IPython = keras.utils.plot_model(
            model=self.model, to_file=filname, show_shapes=show_shapes, show_dtype=show_dtype,
            show_layer_names=show_layer_names, expand_nested=expand_nested,
            show_layer_activations=show_layer_activations, dpi=dpi)
        if to_file is False: # => del the crated image
            os.remove(filname)
        return image_IPython


    def getSeriesFilter(
            self, selection:"Literal['inputs', 'outputs', 'both']")->"set[_SerieName]":
        series: "set[_SerieName]" = set()
        if selection in ("inputs", "both"):
            series.update(self.ioConfig.input_series)
        if selection in ("outputs", "both"):
            series.update(self.ioConfig.output_series)
        return series
    
        

    def transformeToRegularizedDatas(
            self, fromDatas:"Datas_dataset[_T_DfKey]|Datas_series_raw[_T_DfKey]",
            selection:"Literal['inputs', 'outputs', 'both']")->"Datas_series_regularized[_T_DfKey]":
        """trasnforme the datas to regularized datas with all the series that the AI might needs"""
        if isinstance(fromDatas, Datas_dataset):
            fromDatas = fromDatas.toDatas_series_raw(
                seriesGeneratorConfig=self.seriesGeneratorConfig,
                dtype=self.ioConfig.datasDtype)
        # => fromDatas is a "Datas_series_raw[_T_DfKey]"
        # select the series to keep
        fromDatas = fromDatas.filterSeries(self.getSeriesFilter(selection))
        return fromDatas.regularize(self.ioConfig.regularizeSeriesConfig)
    
    def transformeToArrayDatas(
            self, fromDatas:"Datas_series_regularized[_T_DfKey]", 
            datasKind:"Literal['inputs', 'outputs']", filterSeries:bool=True)->"Datas_array[_T_DfKey]":
        # determine the series and the nb of periodes to use
        nbPeriodes: int; seriesOrder: "list[_SerieName]"
        if datasKind == "inputs":
            nbPeriodes = self.ioConfig.nbPeriodes_inputs
            seriesOrder = self.ioConfig.input_series
        elif datasKind == "outputs":
            nbPeriodes = self.ioConfig.nbPeriodes_outputs
            seriesOrder = self.ioConfig.output_series
        else: raise ValueError(f"invalide datasKind: {repr(datasKind)}")
        # filter the series if asked
        if filterSeries is True:
            fromDatas = fromDatas.filterSeries(
                selectedSeries=set(seriesOrder))
        # create the datas
        return fromDatas.toData_array(
            nbPeriodes=nbPeriodes, seriesOrder=seriesOrder,
            dtype=self.ioConfig.datasDtype)
    
    def transformeToInputOutputDatas(
            self, fromDatas:"Datas_series_regularized[_T_DfKey]")->"Datas_InputOutput[_T_DfKey]":
        d: int = self.ioConfig.outputPeriodesShift
        n2: int = self.ioConfig.nbPeriodes_outputs
        n: int = self.ioConfig.nbPeriodes_inputs
        # fromDatas: [T-(s-1) -> T+(n2-1)+d]
        # T+(n2-1)+d - (d + n2 - n) = T+(n-1)
        inputsEndIndex: "int|None" = -(d + n2 - n)
        if inputsEndIndex == 0: inputsEndIndex = None
        elif inputsEndIndex > 0:
            raise ValueError(f"there might be something wrong: inputsEndIndex={inputsEndIndex}")
        inputsRegularized: "Datas_series_regularized[_T_DfKey]" = \
            fromDatas.slice(startIndex=(+0), endIndex=(inputsEndIndex))
        # inputsRegularized: [T-(s-1) -> T+(n-1)]
        outputRegularized: "Datas_series_regularized[_T_DfKey]" = \
            fromDatas.slice(startIndex=(+d), endIndex=(None))
        # outputRegularized: [T-(s-1)+d -> T+(n2-1)+d]
        return Datas_InputOutput(
            inputs_datas=self.transformeToArrayDatas(
                inputsRegularized, datasKind="inputs", filterSeries=True),
            outputs_datas=self.transformeToArrayDatas(
                outputRegularized, datasKind="outputs", filterSeries=True))


    def predict(self, inputDatas:"Datas_array[_T_DfKey]", verbose:"_AI_Verbose")->"Datas_array[_T_DfKey]":
        assert inputDatas.seriesNames == self.ioConfig.input_series
        assert inputDatas.nbPeriodes == self.ioConfig.nbPeriodes_inputs
        
        outputArray: "_3dArray_Float"
        if (inputDatas.nbSamples <= self.batch_size):
                #and (KerasConfig.getCurrentReliableMode() is False):
                # dont use it when in reliable mode => it don't use the GPU
            outputArray = numpy.asarray(self.model(inputDatas.datas, training=False))
        else: outputArray = self.model.predict(
            inputDatas.datas, batch_size=self.batch_size, 
            verbose=cast(str, convertVerbose(verbose)))
        
        output_nbPeriodes: int = self.ioConfig.nbPeriodes_outputs
        input_nbPeriodes: int = self.ioConfig.nbPeriodes_inputs
        shift_periodes: int = self.ioConfig.outputPeriodesShift
        periodeDuration: timedelta = inputDatas.periodeDuration
        lastLayerTotalShift: int = (shift_periodes + output_nbPeriodes - input_nbPeriodes)
        output_datas = Datas_array(
            fromKey=inputDatas.fromKey, datas=outputArray,
            seriesNames=self.ioConfig.output_series.copy(),
            trueStartDate=inputDatas.trueStartDate+periodeDuration*shift_periodes,
            startDate=inputDatas.startDate+periodeDuration*lastLayerTotalShift,
            endDate=inputDatas.endDate+periodeDuration*lastLayerTotalShift,
            valuesRange=self.ioConfig.valuesRange)
        assert output_datas.nbPeriodes == output_nbPeriodes
        return output_datas

    def evaluate_single(self, inputs:"Datas_InputOutput", verbose:"_AI_Verbose")->"float":
        """eval a single datas_array"""
        return assertIsinstance(float, self.model.evaluate(
            x=inputs.inputs_datas.datas,
            y=inputs.outputs_datas.datas,
            batch_size=self.batch_size,
            use_multiprocessing=False, workers=1,
            verbose=cast(str, convertVerbose(verbose)))) # beacuse bad typing from keras

    



