if __name__ == "__main__":
    raise ImportError(f"the {__file__} must be imported from main dir")


import numpy
import pickle
import attrs
import keras.callbacks
from collections import Collection


from calculationLib import _SerieName
from AI import ValuesRange, RegularizeConfig
from ..training.lossFunctions import get_custom_loss_funcs_creator, _CustomLossFuncName
from .datas_types import (
    Datas_dataset, Datas_series_raw, Datas_series_regularized, 
    Datas_array, _T_DfKey, DatasetInfos, )
from save_formats import (
    AsJson_Model_IO_config, AsJson_TrainCallHist,
    AsJson_ResumeConfig, AsJson_CustomLossFuncConfig,
    AsJson_LossToLr, AsJson_LossToLrTable,
    AsJson_FittingHistory, AsJson_MetricsHistory, AsJson_dtype,
    AsJson_Metrics, AsJson_MetricsDfs, AsJson_ConvergenceOutputResume,
    AsJson_BacktestResult, AsJson_BacktestsResultsStorage, AsJson_StrategieResults,
    fromJson_dtype, toJson_dtype, toJsonMultiple, fromJsonMultiple,
    toJsonFunc, fromJsonFunc, toJsonTuple, 
    fromJsonTuple, registerFunction, )
from backtests.senti3Strategie import _ResultKargs as _Backtests_fields, StrategieResults

from holo import assertIsinstance
from holo.types_ext import _Serie_Float, _3dArray_Float, _2dArray_Float
from holo.__typing import (
    Any, Callable, ClassFactory, Generator, Iterable, 
    Protocol, TypeGuard, Literal, Union,
    TypedDict, NotRequired, LiteralString, Dict,
    Concatenate, Self, TypeAlias, TYPE_CHECKING,
    runtime_checkable, Union, Generic, TypeVar,
    isNamedTuple, cast, JsonTypeAlias, Sequence,
    get_args, FinalClass, overload, NoReturn, )
from holo.protocols import _T, _P
from holo.pointers import Pointer
from holo.prettyFormats import _ObjectRepr, PrettyfyClass
from holo.linkedObjects import SkipList

_MetricName = TypeVar("_MetricName", bound=LiteralString)
_MetricName2 = TypeVar("_MetricName2", bound=LiteralString)
_MetricValue = Union[float, int]
_T_MetricValue = TypeVar("_T_MetricValue", bound=_MetricValue)
# use separated typeVar
_MetricNameSingle = TypeVar("_MetricNameSingle", bound=LiteralString)
_MetricNameDfs = TypeVar("_MetricNameDfs", bound=LiteralString)
_DfKey = str
_EpochID = int


### train call history

@attrs.frozen
class TrainCallHist():
    trainFunc: "Callable[..., Any]"
    funcsKwargs: "dict[str, Any]"
    nbEpochesDone: Pointer[int] # pointers makes them "mutable"
    hasFinished: Pointer[bool]
    
    @classmethod
    def create_fromDir(cls, trainFunc:"Callable[..., Any]", 
                       funcs_dir:"dict[str, Any]",
                       names_filter:"set[str]|None"=None)->"TrainCallHist":
        """create an unfinished TrainCallHist with 0 epochesDone\n
        give `names_filter` to gather specifics names, \
            default get all names in the funcs def (except self/cls)"""
        if names_filter is None:
            # get all args except self/cls
            names_filter = set(trainFunc.__code__.co_varnames)
            if "self" in names_filter: names_filter.remove("self")
            elif "cls" in names_filter: names_filter.remove("cls")
        # => names_filter is a set[str]
        
        return cls.create(
            trainFunc=trainFunc,
            funcsKwargs={argName: funcs_dir[argName] for argName in names_filter},
            check=False) # alredy done 
    
    @classmethod
    def create(cls, trainFunc:"Callable[..., Any]",
               funcsKwargs:"dict[str, Any]", check:bool=True)->"TrainCallHist":
        """create an unfinished TrainCallHist with 0 epochesDone\n
        `check` will check that all arg's name from `funcsKwargs`\
            are valide with trainFunc's definition \
            won't allow names given with **kwargs"""
        if check is True:
            # assert the values
            assert callable(trainFunc)
            kwargs_names:"set[str]" = set(trainFunc.__code__.co_varnames)
            for kw in funcsKwargs.keys():
                if kw not in kwargs_names:
                    raise KeyError(f"argument: {repr(kw)} isn't in the definition of {trainFunc}")
        return TrainCallHist(
            trainFunc=trainFunc, funcsKwargs=funcsKwargs, 
            nbEpochesDone=Pointer(0), hasFinished=Pointer(False))

    def marksAsFinished(self)->None:
        if self.hasFinished.value is True:
            raise RuntimeError("has allredy finished")
        self.hasFinished.value = True
    
    def addEpoches(self, nbNewEpochesDone:int=1)->None:
        if nbNewEpochesDone < 0:
            raise ValueError(f"nbNewEpochesDone must be positive value")
        self.nbEpochesDone.value = self.nbEpochesDone.value + nbNewEpochesDone

    def toJson(self)->"AsJson_TrainCallHist":
        return AsJson_TrainCallHist(
            cls=self.__class__.__name__,
            trainFunc=toJsonFunc(self.trainFunc),
            funcsKwargs=self.funcsKwargs,
            nbEpochesDone=self.nbEpochesDone.value,
            hasFinished=self.hasFinished.value)
    
    @classmethod
    def fromJson(cls, datas:"AsJson_TrainCallHist")->"TrainCallHist":
        assert datas["cls"] == cls.__name__
        trainCallHist = TrainCallHist.__new__(cls)
        TrainCallHist.__init__(
            self=trainCallHist,
            trainFunc=fromJsonFunc(datas["trainFunc"]),
            funcsKwargs=datas["funcsKwargs"],
            nbEpochesDone=Pointer(datas["nbEpochesDone"]),
            hasFinished=Pointer(datas["hasFinished"]))
        return trainCallHist


class FittingHistory():
    __slots__ = ("fittCallsParams", )
    
    def __init__(self, *, _defaults:"list[dict[str, int|str]]|None"=None) -> None:
        self.fittCallsParams: "list[dict[str, int|str]]" = \
            ([] if _defaults is None else _defaults)
        """list all the fit calls done and their parametes"""
    
    def addCall(self, hist:"keras.callbacks.History")->None:
        parameters: "dict[str, int|str]" = {}
        # safely add the parameters (check all types)
        for arg, value in assertIsinstance(dict, hist.params).items():
            arg = assertIsinstance(str, arg)
            if not isinstance(value, (str, int)):
                raise TypeError(f"unsupported type: {type(value)} for the parameter: {arg}")
            if arg == "epochs": 
                # what is given is the ending epoch 
                # replace it with the nb of epoches done
                value = len(hist.epoch)
            parameters[arg] = value
        self.fittCallsParams.append(parameters)
    
    def toJson(self)->"AsJson_FittingHistory":
        return AsJson_FittingHistory(
            cls=self.__class__.__name__,
            fittCallsParams=self.fittCallsParams)
        
    @classmethod
    def fromJson(cls, datas:"AsJson_FittingHistory")->"Self":
        assert datas["cls"] == cls.__name__
        fitHist = FittingHistory.__new__(cls)
        FittingHistory.__init__(
            self=fitHist, _defaults=datas["fittCallsParams"])
        return fitHist



class Metrics(Generic[_MetricName], FinalClass, PrettyfyClass):
    """handle the metrics for the AI (one value per epochID for each metric)"""
    __slots__ = ("values", )
    
    def __init__(self, metrics:"set[_MetricName]") -> None:
        self.values: "dict[_MetricName, dict[_EpochID, _MetricValue]]"
        """dict[metric -> (epochID -> value)] # keys are all setted"""
        self.values = {name: {} for name in metrics}
    
    def __init_fromValues__(self, values:"dict[_MetricName, dict[_EpochID, _MetricValue]]")->None:
        self.values = values
    
    @property
    def metrics(self)->"Collection[_MetricName]":
        return self.values.keys()

    @property
    def maxEpochID(self)->"_EpochID|None":
        return max((epochID for metricValues in self.values.values()
                    for epochID in metricValues.keys()), 
                   default=None)

    def validateMetric(self, metricName:str)->"TypeGuard[_MetricName]":
        return (metricName in self.metrics)
    
    def assertIsMetric(self, metricName:str)->"_MetricName|NoReturn":
        if not self.validateMetric(metricName):
            raise KeyError(f"invalide name for a metric: {repr(metricName)}, isn't registered")
        return metricName

    def ensureMetrics(self, requiredMetrics:"set[_MetricName2]")->"Metrics[_MetricName2]|NoReturn":
        """ensure that the required metrics are in self (return self)"""
        for metric in requiredMetrics:
            self.assertIsMetric(metricName=metric)
        # => have all the required metrics
        return cast("Metrics[_MetricName2]", self)

    def _getMetricEpochesDatas(self, matricName:"_MetricName")->"dict[_EpochID, _MetricValue]":
        metricDatas: "dict[_EpochID, _MetricValue]|None" = self.values.get(matricName, None)
        if metricDatas is None:
            raise KeyError(f"invalide name for a metric: {repr(matricName)}, isn't registered")
        return metricDatas

    def addSingle(self, epochID:"_EpochID", metricName:"_MetricName", value:"_MetricValue")->None:
        """add the value of a metric for the given epochID"""
        metricDatas = self._getMetricEpochesDatas(matricName=metricName)
        # check the value isn't alredy setted
        if epochID in metricDatas.keys():
            raise KeyError(f"there is alredy a value for the metric: {metricName} at epochID: {epochID}")
        # => new value
        metricDatas[epochID] = value
        
    def getValue(self, epochID:"_EpochID", metricName:"_MetricName", 
                 assertValueType:"type[_T_MetricValue]"=_MetricValue)->"_T_MetricValue":
        metricDatas = self._getMetricEpochesDatas(matricName=metricName)
        # check the value isn't alredy setted
        value: "_MetricValue|None" = metricDatas.get(epochID, None)
        if value is None:
            raise KeyError(f"there is no a value for the metric: {metricName} at epochID: {epochID}")
        return assertIsinstance(assertValueType, value)
    
    def addMultiple(self, values:"dict[tuple[_MetricName, _EpochID], _MetricValue]")->None:
        for (metricName, epochID), value in values.items():
            self.addSingle(epochID=epochID, metricName=metricName, value=value)
    
    def getPlotDatas(self, metricName:"_MetricName", epochRange:"range|None",
                     missing:"_T"=None)->"tuple[range, list[_MetricValue|_T]]":
        metricDatas = self._getMetricEpochesDatas(matricName=metricName)
        if epochRange is None:
            epochsSorted = sorted(metricDatas.keys())
            if len(epochsSorted) != 0:
                epochRange = range(epochsSorted[0], epochsSorted[-1]+1)
            else: epochRange = range(0, 0) # => empty
        # => epochRange
        return (epochRange, [metricDatas.get(epochID, missing) for epochID in epochRange])
    
    @classmethod
    def fromJson(cls, datas:"AsJson_Metrics[_MetricName]")->"Metrics[_MetricName]":
        assert datas["cls"] == cls.__name__
        assert issubclass(cls, Metrics)
        metrics: "Metrics[_MetricName]" = Metrics.__new__(cls)
        metrics.__init_fromValues__(values=datas["values"])
        return metrics

    def toJson(self)->"AsJson_Metrics[_MetricName]":
        return AsJson_Metrics(cls=self.__class__.__name__, values=self.values)
    


class MetricsDfs(FinalClass, Generic[_MetricName], PrettyfyClass):
    __slots__ = ("metrics", "checkDfKey", "values", )
    
    metrics: "set[_MetricName]"
    checkDfKey: "Callable[[_DfKey], bool]|None"
    """None -> all keys are valide | func -> returns True when a key is valide"""
    values: "dict[_DfKey, Metrics[_MetricName]]"
    """dict[_DfKey -> (metric -> (epochID -> value))]"""
    
    def __init__(self, metrics:"set[_MetricName]", checkDfKey:"Callable[[str], bool]|None")->None:
        self.metrics = metrics
        self.checkDfKey = checkDfKey
        self.values = {}
    
    @property
    def knownDfsKeys(self)->"set[_DfKey]":
        return set(self.values.keys())
    
    @property
    def maxEpochID(self)->"_EpochID|None":
        maxi: "_EpochID|None" = None
        for dfMetrics in self.values.values():
            dfMaxEpochID: "int|None" = dfMetrics.maxEpochID
            if dfMaxEpochID is None: continue
            elif maxi is None: maxi = dfMaxEpochID
            else: maxi = max(maxi, dfMaxEpochID)
        return maxi
    
    def validateMetric(self, metricName:str)->"TypeGuard[_MetricName]":
        return (metricName in self.metrics)
    
    def assertIsMetric(self, metricName:str)->"_MetricName":
        if not self.validateMetric(metricName):
            raise KeyError(f"invalide name for a metric: {repr(metricName)}, isn't registered")
        return metricName
    
    def ensureMetrics(self, requiredMetrics:"set[_MetricName2]")->"MetricsDfs[_MetricName2]|NoReturn":
        """ensure that the required metrics are in self (return self)"""
        for metric in requiredMetrics:
            self.assertIsMetric(metricName=metric)
        # => have all the required metrics
        return cast("MetricsDfs[_MetricName2]", self)
    
    def assertDfKeyIsValide(self, dfKey:"_DfKey")->"None|NoReturn":
        if (self.checkDfKey is not None) and (self.checkDfKey(dfKey) is False):
            # => this key is not allowed
            raise KeyError(f"invalide dfKey: {dfKey} can't add metrics for this dfKey")
    
    def _getDfMetrics(self, dfKey:"_DfKey", createIfMissing:bool)->"Metrics[_MetricName]|NoReturn":
        self.assertDfKeyIsValide(dfKey=dfKey)
        dfMetrics: "Metrics[_MetricName]|None" = self.values.get(dfKey, None)
        if dfMetrics is None:
            if createIfMissing is True:
                # => this key needs to be added
                self.values[dfKey] = dfMetrics = Metrics(metrics=self.metrics)
            else: raise KeyError(f"the dfkey: {dfKey} currently has no metrics")
        return dfMetrics
     
    def addSingle(self, epochID:"_EpochID", dfKey:"_DfKey", name:"_MetricName", value:"_MetricValue")->None:
        """add the value of a metric for the given epochID and dataframeKey"""
        dfMetrics: "Metrics[_MetricName]" = self._getDfMetrics(dfKey=dfKey, createIfMissing=True)
        try: dfMetrics.addSingle(epochID=epochID, metricName=name, value=value)
        except KeyError as err: 
            raise KeyError(f"failed to add the metric for dfkey {repr(dfKey)}: {str(err)}")
    
    def getValue(self, epochID:"_EpochID", dfKey:"_DfKey", name:"_MetricName", 
                 assertValueType:"type[_T_MetricValue]"=_MetricValue)->"_T_MetricValue":
        dfMetrics: "Metrics[_MetricName]" = self._getDfMetrics(dfKey=dfKey, createIfMissing=False)
        try: return dfMetrics.getValue(epochID=epochID, metricName=name, assertValueType=assertValueType)
        except KeyError as err: 
            raise KeyError(f"failed to get the metric for dfkey {repr(dfKey)}: {str(err)}")
    
    def addMultiple(
            self, values:"dict[tuple[_DfKey, _MetricName, _EpochID], _MetricValue]")->None:
        for (dfKey, metric, epochID), val in values.items():
            self.addSingle(epochID=epochID, dfKey=dfKey, name=metric, value=val)
    
    def getPlotDatas(self, metric:"_MetricName", dfKeys:"set[_DfKey]|None",
                     epochRange:"range|None", missing:"_T"=None,
                     )->"dict[_DfKey, tuple[range, list[_MetricValue|_T]]]":
        if dfKeys is None:
            # => auto df keys
            dfKeys = set(self.values.keys())
        # => dfKeys are setted => get the datas
        results: "dict[_DfKey, tuple[range, list[_MetricValue|_T]]]" = {}
        for dfKey in dfKeys:
            dfMetrics: "Metrics[_MetricName]" = \
                self._getDfMetrics(dfKey=dfKey, createIfMissing=False)
            results[dfKey] = dfMetrics.getPlotDatas(
                metricName=metric, epochRange=epochRange, missing=missing)
        return results

    @classmethod
    def fromJson(cls, datas:"AsJson_MetricsDfs[_MetricName]")->"MetricsDfs[_MetricName]":
        assert datas["cls"] == cls.__name__
        assert issubclass(cls, MetricsDfs)
        metrics: "MetricsDfs[_MetricName]" = MetricsDfs.__new__(cls)
        MetricsDfs.__init__(
            self=metrics, 
            metrics=set(datas["metrics"]),
            checkDfKey=(None if datas["checkDfKey"] is None
                        else fromJsonFunc(datas["checkDfKey"])))
        ### add the values
        for dfKey, metricsDatas in datas["values"].items():
            metrics.values[dfKey] = Metrics.fromJson(metricsDatas)
        return metrics
    
    def toJson(self)->"AsJson_MetricsDfs[_MetricName]":
        return AsJson_MetricsDfs(
            cls=self.__class__.__name__,
            metrics=list(self.metrics),
            checkDfKey=(None if self.checkDfKey is None else toJsonFunc(self.checkDfKey)),
            values={dfKey: metricsDatas.toJson()
                    for dfKey, metricsDatas in self.values.items()})


class AllMetrics(Generic[_MetricNameSingle, _MetricNameDfs], FinalClass, PrettyfyClass):
    __slots__ = ("singleMetrics", "dfsMetrics", )
    
    singleMetrics: "Metrics[_MetricNameSingle]"
    dfsMetrics: "MetricsDfs[_MetricNameDfs]"
    
    def __init__(
            self, singleMetricNames:"set[_MetricNameSingle]", 
            dfsMetricNames:"set[_MetricNameDfs]", checkDfKey:"Callable[[str], bool]|None")->None:
        self.singleMetrics = Metrics(metrics=singleMetricNames)
        self.dfsMetrics = MetricsDfs(metrics=dfsMetricNames, checkDfKey=checkDfKey)
    
    @property
    def maxEpochID(self)->"int|None":
        singleMaxEpoch = self.singleMetrics.maxEpochID
        dfsMaxEpoch = self.dfsMetrics.maxEpochID
        if singleMaxEpoch is None:
            return dfsMaxEpoch
        elif dfsMaxEpoch is None:
            return singleMaxEpoch
        else: return max(singleMaxEpoch, dfsMaxEpoch)
    
    @classmethod
    def fromJson(cls, datas:"AsJson_MetricsHistory")->"Self":
        assert datas["cls"] == cls.__name__
        allMetrics = AllMetrics.__new__(cls)
        allMetrics.singleMetrics = Metrics.fromJson(datas["singleMetrics"])
        allMetrics.dfsMetrics = MetricsDfs.fromJson(datas["dfsMetrics"])
        return allMetrics

    def toJson(self)->"AsJson_MetricsHistory":
        return AsJson_MetricsHistory(
            cls=self.__class__.__name__,
            singleMetrics=self.singleMetrics.toJson(),
            dfsMetrics=self.dfsMetrics.toJson())


### AI's io config

@attrs.frozen
class Model_IO_config():
    nbPeriodes_inputs: int
    nbPeriodes_outputs: int
    outputPeriodesShift: int
    input_series: "list[_SerieName]"
    output_series: "list[_SerieName]"
    datasDtype: "type[numpy.float16|numpy.float32|numpy.float64]"
    regularizeSeriesConfig: "RegularizeConfig"
    
    # for an arbitrary sample at the time T:
    # for an arbitrary feature, the values of the inputs are:
    #   [T-(nbPeriodes_inputs-1), ..., T+0]
    # and the associated outputs values for an arbitrary feature:
    #   [T-(nbPeriodes_inputs-1)+shift, ..., T+shift]
    
    # shape of the inputs: (nbSamples, nbFeatures_inputs, nbPeriodes_inputs)
    # indexes (for a given feature): periodes (0 -> n-1) / samples (0 -> s-1)
    # [  T-(s-1)+0  , ... ,  T-0+0  ] # trueStartDate = T-(s-1)+0
    # [    ...      ,     ,   ...   ] # startDate =T-(s-1)+(n-1)
    # [T-(s-1)+(n-2), ... ,T-0+(n-2)] # endDate = T-0+(n-1)
    # [T-(s-1)+(n-1), ... ,T-0+(n-1)]
    #
    # shape of the outputs: (nbSamples, nbFeatures_outputs, nbPeriodes_outputs)
    # indexes (for a given feature): periodes (0 -> n2-1) / samples (0 -> s-1)
    # [  T-(s-1)+0+d   , ... ,   T-0+0+d  ] # trueStartDate = T-(s-1)+0+d
    # [      ...       ,     ,     ...    ] # startDate = T-(s-1)+(n2-1)+d
    # [T-(s-1)+(n2-2)+d, ... ,T-0+(n2-2)+d] # endDate = T-0+(n2-1)+d
    # [T-(s-1)+(n2-1)+d, ... ,T-0+(n2-1)+d]
    
    @property
    def nbFeatures_inputs(self)->int:
        return len(self.input_series)
    
    @property
    def nbFeatures_outputs(self)->int:
        return len(self.output_series)
    
    @property
    def valuesRange(self)->ValuesRange:
        return self.regularizeSeriesConfig.valuesRange
    
    @classmethod
    def fromJson(cls, datas:"AsJson_Model_IO_config")->"Self":
        assert datas["cls"] == cls.__name__
        io_config = Model_IO_config.__new__(cls)
        Model_IO_config.__init__(
            self=io_config,
            nbPeriodes_inputs=datas["nbPeriodes_inputs"],
            nbPeriodes_outputs=datas["nbPeriodes_outputs"],
            outputPeriodesShift=datas["outputPeriodesShift"],
            input_series=datas["input_series"],
            output_series=datas["output_series"],
            datasDtype=fromJson_dtype(datas["datasDtype"]), 
            regularizeSeriesConfig=RegularizeConfig.fromJson(
                datas["regularizeSeriesConfig"]))
        return io_config

    def toJson(self)->"AsJson_Model_IO_config":
        return AsJson_Model_IO_config(
            cls=self.__class__.__name__,
            nbPeriodes_inputs=self.nbPeriodes_inputs,
            nbPeriodes_outputs=self.nbPeriodes_outputs,
            outputPeriodesShift=self.outputPeriodesShift,
            input_series=self.input_series,
            output_series=self.output_series,
            datasDtype=toJson_dtype(self.datasDtype), 
            regularizeSeriesConfig=self.regularizeSeriesConfig.toJson())




### resume config

@registerFunction
def _resume_func_nbIter(v:float): return f"{v:>4.1f}"
@registerFunction
def _resume_func_TradesDura(v:float): return f"{v:.2f}"

@attrs.frozen
class ResumeConfig(Generic[_MetricName]):
    metricName: "_MetricName"
    alias: "str|None" = None
    func: "Callable[[float], str]|None" = None

    def toJson(self)->"AsJson_ResumeConfig":
        return AsJson_ResumeConfig(
            cls=self.__class__.__name__,
            metricName=self.metricName,
            alias=self.alias,
            func=(None if self.func is None 
                  else toJsonFunc(self.func)))
    
    @classmethod
    def fromJson(cls, datas:"AsJson_ResumeConfig")->Self:
        assert datas["cls"] == cls.__name__
        resumeConfig = ResumeConfig.__new__(cls)
        ResumeConfig.__init__(
            self=resumeConfig,
            metricName=datas["metricName"],
            alias=datas["alias"], 
            func=(None if datas["func"] is None
                  else fromJsonFunc(datas["func"])))
        return resumeConfig


### loss function config

@attrs.frozen
class CustomLossFuncConfig():
    functionName: "_CustomLossFuncName"
    default_kwargs: "dict[str, Any]"

    def createLossFunction(self): 
        # out is not typed, infer the type from get_custom_loss_funcs_creator
        return get_custom_loss_funcs_creator(
            funcName=self.functionName)(**self.default_kwargs)

    def toJson(self)->"AsJson_CustomLossFuncConfig":
        return AsJson_CustomLossFuncConfig(
            cls=self.__class__.__name__,
            functionName=self.functionName,
            kwargs=self.default_kwargs)
    
    @classmethod
    def fromJson(cls, datas:"AsJson_CustomLossFuncConfig")->Self:
        assert datas["cls"] == cls.__name__
        customLossConfig = CustomLossFuncConfig.__new__(cls)
        CustomLossFuncConfig.__init__(
            self=customLossConfig,
            functionName=datas["functionName"],
            default_kwargs=datas["kwargs"])
        return customLossConfig

 
### loss -> Lr tuple

class LossToLr_table(FinalClass):
    __slots__ = ("table", )
    
    table: "SkipList[LossToLr, float]"
    """sorted table of lossToLr (from lowLoss to maxLoss)"""
    
    def __init__(self, entrys:"Iterable[tuple[float, float]|LossToLr]") -> None:
        self.table = SkipList(
            [(elt if isinstance(elt, LossToLr) else LossToLr(*elt)) for elt in entrys],
            eltToKey=lambda elt: elt.loss, probability=1/2)
    
    def getLr(self, loss:float)->float:
        """return the learning rate computed from the table and the loss\n
        linearly interpolates the losses inside de table and retun the bounds of the table if outside"""
        ### handle the upper and lower bounds of losses
        if loss >= self.table.getLast().loss:
            # => loss is over the max of the table => return its lr
            return self.table.getLast().lr
        elif loss <= self.table.getFirst().loss:
            # => loss is under the min of the table => return its lr
            return self.table.getFirst().lr
        # => loss is strictly inside the table
        try: return self.table.get(loss).lr # exact
        except KeyError: pass # => not exact loss match
        ### find the correct interval for the `loss` and linearly interpolate the learning rate
        return self.__lerp(
            loss=loss, lowLossLr=self.table.getBefore(loss), 
            highLossLr=self.table.getAfter(loss))
        
    def __lerp(self, loss:float, lowLossLr:"LossToLr", highLossLr:"LossToLr")->float:
        lrRange: float = (highLossLr.lr - lowLossLr.lr)
        lossRange: float = (highLossLr.loss - lowLossLr.loss)
        rangePosition: float = (loss - lowLossLr.loss) / lossRange
        """betwin [0, 1], where loss is, relatively to [low, high]"""
        return lowLossLr.lr + lrRange * rangePosition
    
    def __pretty__(self, *_, **__)->"_ObjectRepr":
        return _ObjectRepr(
            className=self.__class__.__name__,
            args=(), kwargs={"table": list(self.table)})
    
    
    def toJson(self)->"AsJson_LossToLrTable":
        return AsJson_LossToLrTable(
            cls=self.__class__.__name__, 
            table=toJsonMultiple(self.table))

    @classmethod
    def fromJson(cls, datas:"AsJson_LossToLrTable")->"Self":
        assert datas["cls"] == cls.__name__
        table = LossToLr_table.__new__(cls)
        LossToLr_table.__init__(
            self=table, entrys=fromJsonMultiple(datas["table"], LossToLr, LossToLr))
        return table

@attrs.frozen
class LossToLr():
    loss: float
    lr: float

    def toJson(self)->"AsJson_LossToLr":
        return AsJson_LossToLr(
            cls=self.__class__.__name__,
            loss=self.loss, lr=self.lr)

    @classmethod
    def fromJson(cls, datas:"AsJson_LossToLr")->"Self":
        assert datas["cls"] == cls.__name__
        lossToLrTuple = LossToLr.__new__(cls)
        LossToLr.__init__(
            self=lossToLrTuple, loss=datas["loss"], lr=datas["lr"])
        return lossToLrTuple



### convergence computation types



@attrs.frozen
class ConvergenceOutput(Generic[_T_DfKey]):
    converged_inputs: "Datas_series_regularized[_T_DfKey]"
    last_output: "Datas_array[_T_DfKey]"
    deltas_perStep: "list[dict[_SerieName, float]]"
    hasConverged: bool
    """True if when it stoped it had converged"""
    averageTimePerStep: float
    """the average time it tooke to compleate a convergence step"""
    
    @property
    def seriesNames(self)->"set[_SerieName]":
        if self.nbIterationsDone == 0:
            raise ValueError(f"no convergence step done")
        # => deltas_perStep isn't empty
        return set(self.deltas_perStep[0].keys())
    
    @property
    def nbIterationsDone(self)->int:
        return len(self.deltas_perStep)
    
    def finalDelta(self, serie:"_SerieName")->float:
        if self.nbIterationsDone == 0:
            raise ValueError(f"no convergence step done")
        # => deltas_perStep isn't empty
        return self.deltas_perStep[-1][serie]
    
    def getResume(self)->"ConvergenceOutputResume":
        return ConvergenceOutputResume(
            hasConverged=self.hasConverged, deltas_perStep=self.deltas_perStep,
            nbIterationsDone=self.nbIterationsDone, averageTimePerStep=self.averageTimePerStep,
            inputedDataframeInfos=self.converged_inputs.getInfos())

@attrs.frozen
class ConvergenceOutputResume():
    inputedDataframeInfos: "DatasetInfos"
    nbIterationsDone: int
    deltas_perStep: "list[dict[_SerieName, float]]"
    hasConverged: bool
    averageTimePerStep: float
    
    def toJson(self)->"AsJson_ConvergenceOutputResume":
        return AsJson_ConvergenceOutputResume(
            cls=self.__class__.__name__, inputedDataframeInfos=self.inputedDataframeInfos.toJson(),
            hasConverged=self.hasConverged, averageTimePerStep=self.averageTimePerStep,
            deltas_perStep=self.deltas_perStep, nbIterationsDone=self.nbIterationsDone)
    @classmethod
    def fromJson(cls, datas:"AsJson_ConvergenceOutputResume")->"Self":
        assert datas["cls"] == cls.__name__
        resume = ConvergenceOutputResume.__new__(cls)
        ConvergenceOutputResume.__init__(
            self=resume, inputedDataframeInfos=DatasetInfos.fromJson(datas["inputedDataframeInfos"]),
            hasConverged=datas["hasConverged"], averageTimePerStep=datas["averageTimePerStep"],
            deltas_perStep=datas["deltas_perStep"], nbIterationsDone=datas["nbIterationsDone"])
        return resume



### backtests related
@attrs.frozen()
class BacktestResult(PrettyfyClass):
    datasetInfos: "DatasetInfos"
    result: "StrategieResults"

    def toJson(self)->"AsJson_BacktestResult":
        return AsJson_BacktestResult(
            cls=self.__class__.__name__,
            datasetInfos=self.datasetInfos.toJson(),
            result=self.result.toJson())
    @classmethod
    def fromJson(cls, datas:"AsJson_BacktestResult")->"Self":
        assert datas["cls"] == cls.__name__
        result = BacktestResult.__new__(cls)
        BacktestResult.__init__(
            self=result, datasetInfos=DatasetInfos.fromJson(datas["datasetInfos"]),
            result=StrategieResults.fromJson(datas["result"]))
        return result

@attrs.frozen()
class BacktestResumeConfig(PrettyfyClass):
    ... # TODO ? -> integrate to BacktestsResultsStorage


class BacktestsResultsStorage(FinalClass, PrettyfyClass):
    __slots__ = ("perEpochsResults", "otherResults", )
    
    def __init__(self) -> None:
        self.perEpochsResults: "dict[_DfKey, dict[int, BacktestResult]]" = {}
        self.otherResults: "list[BacktestResult]" = []
    
    def __init_fromValues__(
            self, otherResults:"list[BacktestResult]", 
            perEpochsResults:"dict[_DfKey, dict[int, BacktestResult]]")->None:
        self.perEpochsResults = perEpochsResults
        self.otherResults = otherResults
    
    
    def addPerEpochResult(self, epoch:int, dfKey:"_DfKey", result:"BacktestResult")->None:
        """add the result of the backtest the given epoch and dataframeKey"""
        epochsResults: "dict[int, BacktestResult]|None" = self.perEpochsResults.get(dfKey, None)
        if epochsResults is None:
            self.perEpochsResults[dfKey] = epochsResults = {}
        # check the value isn't alredy setted
        if epoch in epochsResults.keys():
            raise KeyError(f"there is alredy a result for the dfKey: {dfKey} at epoch: {epoch}")
        # => new value
        epochsResults[epoch] = result
    
    def toJson(self)->"AsJson_BacktestsResultsStorage":
        return AsJson_BacktestsResultsStorage(
            cls=self.__class__.__name__,
            perEpochsResults={
                dfKey: {epoch: res.toJson() for epoch, res in dfData.items()}
                for dfKey, dfData in self.perEpochsResults.items()},
            otherResults=toJsonMultiple(self.otherResults))
    @classmethod
    def fromJson(cls, datas:"AsJson_BacktestsResultsStorage")->"Self":
        assert datas["cls"] == cls.__name__
        storage = BacktestsResultsStorage.__new__(cls)
        BacktestsResultsStorage.__init_fromValues__(
            self=storage, 
            perEpochsResults={
                dfkey: {epoch: BacktestResult.fromJson(data) for epoch, data in dfData.items()}
                for dfkey, dfData in datas["perEpochsResults"].items()},
            otherResults=fromJsonMultiple(datas["otherResults"], BacktestResult, BacktestResult))
        return storage