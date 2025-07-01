import attrs
import numpy
import copy
from io import BufferedReader
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as plt_Figure

from AI.senti3.datas.types_config import (
    AllMetrics, _MetricNameSingle, _MetricNameDfs, )

from save_formats import (
    fromJsonMultiple, toJsonMultiple,
    toJsonTuple, fromJsonTuple, manualReparesWrapperGUI,
    AsJson_AxisConfig, AsJson_PlotConfig, AsJson_FigureConfig, 
    AsJson_Limits, AsJson_LineConfig, AsJson_AxisMetricsConfigs,
    AsJson_DfsMetricConfig, AsJson_SingleMetricConfig, SupportJson, )



from holo import MapDictValues2, MapDictValues
from holo.__typing import (
    Generic, Self, Literal, Iterable,
    Callable, TypeVar, TypedDict, 
    Any, cast, get_args_LiteralString, )
from holo.protocols import _T, _T2, _T3
from holo.types_ext import _Serie_Float
from holo.prettyFormats import PrettyfyClass, ClassFactory

_DfKey = str
_LineStyle = Literal["solid", "dotted", "dashed", "dashdot"]
_PointStyle = Literal["None", "point", "circle", "square", "triangle_down"]
_MarkerStyle = Literal["None", ".", "o", "s", "v"]
_PointStyle_to_MarkerStyle: "dict[_PointStyle, _MarkerStyle]" = {
    "None": "None", "point": ".", "circle": "o", 
    "square": "s", "triangle_down": "v", }

_Scales = Literal['linear', 'log', 'symlog', 'logit']

########## utils

class Map(Generic[_T, _T2]):
    def __init__(self, func:"Callable[[_T], _T2]")->None:
        self.func: "Callable[[_T], _T2]" = func
    
    def __call__(self, args:"Iterable[_T]")->"Iterable[_T2]":
        return (self.func(arg) for arg in args)

class Compose(Generic[_T, _T2, _T3]):
    def __init__(self, func1:"Callable[[_T], _T2]", 
                 func2:"Callable[[_T2], _T3]")->None:
        self.func1: "Callable[[_T], _T2]" = func1
        self.func2: "Callable[[_T2], _T3]" = func2
    
    def __call__(self, arg:"_T")->"_T3":
        return self.func2(self.func1(arg))


def extractTextList(text:str)->"list[str]":
    """extract the elements of a list and tuple like: [a, b, c, ...]\n
    works with `[...]`, `(...)`, or even just `a, b, c, ...` """
    splits = [v.strip() for v in text.strip("[()]").split(",")]
    if (len(splits) == 1) and (splits[0] == ""):
        # => empty
        return []
    return splits

def extractIndexs(text:str)->"int|tuple[int, int]":
    try:
        splits = text.strip("[()]").split(",", maxsplit=1)
        if len(splits) == 1:
            return int(splits[0])
        elif len(splits) == 2:
            return (int(splits[0]), int(splits[1]))
    except: pass # => will raise a generic error
    raise ValueError(f"the indexs of the new axis is invalide: {text}")
        

class NoneIfy(Generic[_T, _T2]):
    def __init__(self, func:"Callable[[_T], _T2]")->None:
        self.func: "Callable[[_T], _T2]" = func
    
    def __call__(self, arg:"None|_T")->"None|_T2":
        return (None if arg is None else self.func(arg))


########## config


@attrs.frozen(kw_only=True)
class Limits(PrettyfyClass):
    mini: float
    maxi: float
    
    @classmethod
    def fromText(cls, text:str)->"Limits|None":
        """convert a limit of format: `mini -> maxi`"""
        text = text.strip()
        if text == "auto":
            return None
        parts: "list[str]" = text.split("->")
        assert len(parts) == 2
        return Limits(mini=float(parts[0]), maxi=float(parts[1]))
    
    def __str__(self) -> str:
        return f"{self.mini} -> {self.maxi}"
    
    def toJson(self)->"AsJson_Limits":
        return AsJson_Limits(
            cls=self.__class__.__name__, 
            mini=self.mini, maxi=self.maxi)
    
    @classmethod
    @manualReparesWrapperGUI
    def fromJson(cls, datas:"AsJson_Limits")->"Self":
        assert cls.__name__ == datas["cls"]
        lim = Limits.__new__(cls)
        Limits.__init__(self=lim, mini=datas["mini"], maxi=datas["maxi"])
        return lim

    def copy(self)->"Self":
        lim = Limits.__new__(type(self))
        Limits.__init__(self=lim, mini=self.mini, maxi=self.maxi)
        return lim
    

_LineConfigFields = Literal["color", "lineStyle", "pointStyle", "lineWidth", "emaConfig"]

class LineKwargs(TypedDict):
    color: "str|None"
    linestyle: "_LineStyle"
    linewidth: float
    marker: "_MarkerStyle"


@attrs.frozen()
class LineConfig(PrettyfyClass, SupportJson[AsJson_LineConfig]):
    enabled: bool
    color: "None|str" # None -> auto | colorName | colorHex
    lineStyle: "_LineStyle"
    pointStyle: "_PointStyle"
    lineWidth: float
    emaCoeff: "None|float"
    """None -> no ema | float -> ema factor"""
    
    @property
    def markerStyle(self)->"_MarkerStyle":
        return _PointStyle_to_MarkerStyle[self.pointStyle]
    
    @staticmethod
    def default()->"LineConfig":
        return LineConfig(
            enabled=True,
            color=None, lineStyle="solid", 
            pointStyle="None", 
            lineWidth=1.0, emaCoeff=None)
    
    def toKwargs(self)->"LineKwargs":
        return LineKwargs(
            color=self.color,
            linestyle=self.lineStyle,
            linewidth=self.lineWidth,
            marker=self.markerStyle)
    
    def plotLine(self, ax:"plt.Axes", xRange:range, 
                 values:"list[float|int]", label:str)->None:
        if self.enabled is False:
            return # => disabled => don't plot
        if self.emaCoeff is not None:
            label = f"{label} - ema{self.emaCoeff:.4g}"
        ax.plot(xRange, self.applieEma(values), 
                label=label, **self.toKwargs())
    
    def applieEma(self, values:"list[float|int]")->_Serie_Float:
        ema_serie: "_Serie_Float" = numpy.asarray(values, dtype=numpy.float32)
        if self.emaCoeff is None:
            return ema_serie # => don't applie ema
        for index in range(1, len(ema_serie)):
            prev = ema_serie[index-1]
            if prev == numpy.nan:
                continue # => no continuity
            # => applie ema 
            # (/!\ nan will stay nan /!\)
            ema_serie[index] = \
                prev * (1 - self.emaCoeff) + ema_serie[index] * self.emaCoeff
        return ema_serie
    
    def toJson(self)->"AsJson_LineConfig":
        return AsJson_LineConfig(
            cls=self.__class__.__name__, 
            enabled=self.enabled, color=self.color,
            lineStyle=self.lineStyle, pointStyle=self.pointStyle,
            width=self.lineWidth, emaCoeff=self.emaCoeff)
    
    @classmethod
    @manualReparesWrapperGUI
    def fromJson(cls, datas:"AsJson_LineConfig")->"Self":
        assert datas["cls"] == cls.__name__
        cfg =  LineConfig.__new__(cls)
        LineConfig.__init__(
            self=cfg, enabled=datas["enabled"],
            color=datas["color"], 
            lineStyle=datas["lineStyle"],
            pointStyle=datas["pointStyle"],
            lineWidth=datas["width"],
            emaCoeff=datas["emaCoeff"])
        return cfg
        
@attrs.frozen()
class SingleMetricConfig(PrettyfyClass):
    configs: "list[LineConfig]"
    
    @staticmethod
    def default()->"SingleMetricConfig":
        return SingleMetricConfig(configs=[])
    
    def getLineConfig(self)->"list[LineConfig]":
        """get the config to use for this metric\n
        will always retrun at least one config (teh default one if it contain no configs)"""
        configs: "list[LineConfig]" = []
        configs.extend(self.configs)
        if len(configs) == 0:
            configs.append(LineConfig.default())
        return configs

    def toJson(self)->"AsJson_SingleMetricConfig":
        return AsJson_SingleMetricConfig(
            cls=self.__class__.__name__,
            configs=toJsonMultiple(self.configs))
    
    @classmethod
    @manualReparesWrapperGUI
    def fromJson(cls, datas:"AsJson_SingleMetricConfig")->"Self":
        assert cls.__name__ == datas["cls"]
        cfgs = SingleMetricConfig.__new__(cls)
        SingleMetricConfig.__init__(
            self=cfgs,
            configs=fromJsonMultiple(datas["configs"], LineConfig, LineConfig))
        return cfgs

    def getConfigs(self)->"list[LineConfig]":
        configs: "list[LineConfig]" = self.configs.copy()
        if len(configs) == 0:
            configs.append(LineConfig.default())
        return configs

    def plot(self, metric:"_MetricNameSingle", ax:"plt.Axes",
             datas:"AllMetrics[_MetricNameSingle, _MetricNameDfs]")->None:
        (xRange, values) = datas.singleMetrics.getPlotDatas(
            metricName=metric, epochRange=None, missing=numpy.nan)
        for cfg in self.getConfigs():
            cfg.plotLine(ax=ax, xRange=xRange, values=values, label=metric)

@attrs.frozen()
class DfsMetricConfig(PrettyfyClass):
    """a config without any lines will return a default line"""
    allKeysConfig: "list[LineConfig]"
    perKeyConfig: "dict[_DfKey, list[LineConfig]]"

    def toJson(self)->"AsJson_DfsMetricConfig":
        TO_LIST = Compose(Map(LineConfig.toJson), list)
        return AsJson_DfsMetricConfig(
            cls=self.__class__.__name__,
            allKeysConfig=TO_LIST(self.allKeysConfig),
            perKeyConfig=MapDictValues2(TO_LIST)(self.perKeyConfig))
    
    @classmethod
    @manualReparesWrapperGUI
    def fromJson(cls, datas:"AsJson_DfsMetricConfig")->"Self":
        assert cls.__name__ == datas["cls"]
        dfsCfg = DfsMetricConfig.__new__(cls)
        FROM_LIST = Compose(Map(LineConfig.fromJson), list)
        DfsMetricConfig.__init__(
            self=dfsCfg, 
            allKeysConfig=FROM_LIST(datas["allKeysConfig"]),
            perKeyConfig=MapDictValues2(FROM_LIST)(datas["perKeyConfig"]))
        return dfsCfg
    
    @staticmethod
    def default()->"DfsMetricConfig":
        return DfsMetricConfig(allKeysConfig=[], perKeyConfig={})
            
    def getLineConfigs(self, dfKey:"_DfKey")->"list[LineConfig]":
        """get the config to use for the given DfKey\n
        will always retrun at least one config (teh default one if it contain no configs)"""
        configs: "list[LineConfig]" = []
        configs.extend(self.allKeysConfig)
        configs.extend(self.perKeyConfig.get(dfKey, list()))
        if len(configs) == 0:
            configs.append(LineConfig.default())
        return configs
    
    def getSelectedDfsKeys(self)->"None|set[_DfKey]":
        if (len(self.allKeysConfig) != 0):
            # => will needs all dfs values
            return None    
        elif len(self.perKeyConfig) != 0:
            # => only needs some of the dfs
            return set(self.perKeyConfig.keys())
        else: return None # => empty config (defaul is for all => select all)
    
    def plot(self, metric:"_MetricNameDfs", ax:"plt.Axes",
             datas:"AllMetrics[_MetricNameSingle, _MetricNameDfs]")->None:
        metricSeries = datas.dfsMetrics.getPlotDatas(
            metric=metric, epochRange=None, missing=numpy.nan, 
            dfKeys=self.getSelectedDfsKeys())
        for dfKey, (xRange, values) in metricSeries.items():
            linesCfgs = self.getLineConfigs(dfKey=dfKey)
            for cfg in linesCfgs:
                cfg.plotLine(ax=ax, xRange=xRange, values=values, 
                             label=f"{metric}:{dfKey}")



@attrs.frozen()
class AxisMetricsConfigs(Generic[_MetricNameSingle, _MetricNameDfs], PrettyfyClass):
    disabledSingleMetrics: "set[_MetricNameSingle]"
    disabledDfsMetrics: "set[_MetricNameDfs]"
    singleLinesConfigs: "dict[_MetricNameSingle, SingleMetricConfig]"
    """all the lines config stored for each single metric"""
    dfsLinesConfigs: "dict[_MetricNameDfs, DfsMetricConfig]"
    """all the lines config stored for each single metric"""
    
    def copy(self)->"Self":
        return copy.deepcopy(self)
    
    @classmethod
    def empty(cls)->"AxisMetricsConfigs": # typeVar are unknown
        return AxisMetricsConfigs(
            disabledSingleMetrics=set(), disabledDfsMetrics=set(),
            singleLinesConfigs=dict(), dfsLinesConfigs=dict())
    
    def toJson(self)->"AsJson_AxisMetricsConfigs[_MetricNameSingle, _MetricNameDfs]":
        return AsJson_AxisMetricsConfigs(
            cls=self.__class__.__name__,
            disabledDfsMetrics=list(self.disabledDfsMetrics),
            disabledSingleMetrics=list(self.disabledSingleMetrics),
            singleLinesConfigs=MapDictValues2(SingleMetricConfig.toJson)(self.singleLinesConfigs),
            dfsLinesConfigs=MapDictValues2(DfsMetricConfig.toJson)(self.dfsLinesConfigs))
    
    @classmethod
    @manualReparesWrapperGUI
    def fromJson(cls, datas:"AsJson_AxisMetricsConfigs[_MetricNameSingle, _MetricNameDfs]")->"Self":
        assert cls.__name__ == datas["cls"]
        axConfig = AxisMetricsConfigs.__new__(cls)
        AxisMetricsConfigs.__init__(
            self=axConfig,
            disabledSingleMetrics=set(datas["disabledSingleMetrics"]),
            disabledDfsMetrics=set(datas["disabledDfsMetrics"]),
            singleLinesConfigs=MapDictValues2(SingleMetricConfig.fromJson)(datas["singleLinesConfigs"]),
            dfsLinesConfigs=MapDictValues2(DfsMetricConfig.fromJson)(datas["dfsLinesConfigs"]))
        return axConfig
    
    def plot(self, ax:"plt.Axes", datas:"AllMetrics[_MetricNameSingle, _MetricNameDfs]")->None:
        # plot all single metrics
        for singleMetric, singlePlotConfig in self.singleLinesConfigs.items():
            if singleMetric in self.disabledSingleMetrics:
                continue # => disabled => don't plot
            singlePlotConfig.plot(metric=singleMetric, ax=ax, datas=datas)
        # plot all dfs metrics
        print(f"to plot -> {set(self.dfsLinesConfigs.keys()).difference(self.disabledDfsMetrics)}")
        for dfMetric, dfPlotConfig in self.dfsLinesConfigs.items():
            if dfMetric in self.disabledDfsMetrics:
                continue # => disabled => don't plot
            dfPlotConfig.plot(metric=dfMetric, ax=ax, datas=datas)
    
    def getSelectedSingleMetrics(self)->"set[_MetricNameSingle]":
        return set(self.singleLinesConfigs.keys()).difference(self.disabledSingleMetrics)
    
    def getSelectedDfsMetrics(self)->"set[_MetricNameDfs]":
        return set(self.dfsLinesConfigs.keys()).difference(self.disabledDfsMetrics)
    
    def updateSelectedSingleMetrics(self, selectedMetrics:"set[_MetricNameSingle]")->None:
        # disable all current
        self.disabledSingleMetrics.update(self.singleLinesConfigs.keys())
        # => select some
        for metric in selectedMetrics:
            if metric in self.singleLinesConfigs.keys():
                # => alredy added, enable it
                self.disabledSingleMetrics.discard(metric)
            else: # => dont have a default config
                self.singleLinesConfigs[metric] = SingleMetricConfig.default()
            
            
    def updateSelectedDfsMetrics(self, selectedMetrics:"set[_MetricNameDfs]")->None:
        # disable all current
        self.disabledDfsMetrics.update(self.dfsLinesConfigs.keys())
        # => select some
        for metric in selectedMetrics:
            if metric in self.dfsLinesConfigs.keys():
                # => alredy added, enable it
                self.disabledDfsMetrics.discard(metric)
            else: # => dont have a default config
                self.dfsLinesConfigs[metric] = DfsMetricConfig.default()
    
    def clearSingleMetricsConfigs(self)->None:
        """clear all the configs of the singleMetrics (keeps the metrics selection)"""
        for metricLineCfgs in self.singleLinesConfigs.values():
            metricLineCfgs.configs.clear()
            
    def addSingleMetricsConfigs(self, configs:"list[tuple[_MetricNameSingle, LineConfig]]")->None:
        """add each config to the singleMetric associated"""
        for cfgName, cfg in configs:
            self.singleLinesConfigs[cfgName].configs.append(cfg)
    
    
    def clearDfsMetricsConfigs(self)->None:
        """clear all the configs of the dfsMetrics (keeps the metrics selection)"""
        for dfsConfigs in self.dfsLinesConfigs.values():
            dfsConfigs.allKeysConfig.clear()
            for dfkeyConfigs in dfsConfigs.perKeyConfig.values():
                dfkeyConfigs.clear()
            
    def addDfsMetricsConfigs(self, configs:"list[tuple[_MetricNameDfs, _DfKey|None, LineConfig]]")->None:
        """add each config to the dfsMetric and dfkey associated (dfkey is None => allDfs config)"""
        for metric, dfKey, cfg in configs:
            metricsConfigs = self.dfsLinesConfigs[metric]
            if dfKey is None: # => all dfs
                metricsConfigs.allKeysConfig.append(cfg)
            else: # => dfKey sepcific
                if dfKey not in metricsConfigs.perKeyConfig.keys():
                    metricsConfigs.perKeyConfig[dfKey] = []
                metricsConfigs.perKeyConfig[dfKey].append(cfg)
    
    

@attrs.frozen()
class AxisConfig(Generic[_MetricNameSingle, _MetricNameDfs], PrettyfyClass):
    name: str
    yLabel: str
    hlines: "set[float]"
    yLimits: "Limits|None"
    scale: "_Scales"
    indexs: "int|tuple[int, int]"
    """define the size and the position of the axis"""
    metricsConfigs: "AxisMetricsConfigs[_MetricNameSingle, _MetricNameDfs]"
    
    @classmethod
    def empty(cls, name:str, yLabel:str, 
              indexs:"int|tuple[int, int]")->"AxisConfig": # typeVar are unknown
        return AxisConfig(
            name=name, yLabel=yLabel, hlines=set(),
            yLimits=None, indexs=indexs, scale="linear",
            metricsConfigs=AxisMetricsConfigs.empty())

    def toJson(self)->"AsJson_AxisConfig[_MetricNameSingle, _MetricNameDfs]":
        return AsJson_AxisConfig(
            cls=self.__class__.__name__, name=self.name, yLabel=self.yLabel,
            hlines=list(self.hlines), scale=self.scale,
            yLimits=NoneIfy(Limits.toJson)(self.yLimits),
            indexs=(self.indexs if isinstance(self.indexs, int) 
                    else toJsonTuple(self.indexs)),
            metricsConfigs=self.metricsConfigs.toJson())
    
    @classmethod
    @manualReparesWrapperGUI
    def fromJson(cls, datas:"AsJson_AxisConfig[_MetricNameSingle, _MetricNameDfs]")->"Self":
        assert cls.__name__ == datas["cls"]
        axConfig = AxisConfig.__new__(cls)
        AxisConfig.__init__(
            self=axConfig, name=datas["name"], yLabel=datas["yLabel"],
            hlines=set(datas["hlines"]), scale=datas["scale"],
            yLimits=NoneIfy(Limits.fromJson)(datas["yLimits"]),
            indexs=(datas["indexs"] if isinstance(datas["indexs"], int) 
                    else fromJsonTuple(tuple, datas["indexs"])), 
            metricsConfigs=AxisMetricsConfigs.fromJson(datas["metricsConfigs"]))
        return axConfig
    
    @staticmethod
    def assertScale(text:str)->"_Scales":
        availableScales = get_args_LiteralString(_Scales)
        if text in availableScales:
            return text # => is valid
        # => assertion failed
        raise ValueError(f"the scale: {text}, isn't valid, "
                         f"please chose one of {', '.join(availableScales)}")
    
    def plot(self, ax:"plt.Axes", datas:"AllMetrics[_MetricNameSingle, _MetricNameDfs]")->None:
        self.metricsConfigs.plot(ax=ax, datas=datas)
        # other things
        ax.legend(loc='upper left')
    
    def setupAxis(self, ax:"plt.Axes")->"tuple[Self, plt.Axes]":
        ax.set_xlabel("epoches")
        ax.set_yscale(self.scale)
        if self.yLabel is not None: ax.set_ylabel(self.yLabel)
        if self.yLimits is not None:
            ax.set_ylim(ymin=self.yLimits.mini, ymax=self.yLimits.maxi)
        # add hlines
        if self.hlines is not None: 
            for hline in self.hlines:
                ax.axhline(hline, color="gray", linestyle="dashed", linewidth=0.75)
        return (self, ax)
        



_FigureConfigField = Literal["figureID", "nbRows", "nbCols", "plotSize"]

@attrs.frozen(kw_only=True)
class FigureConfig(PrettyfyClass):
    figureID: str
    nbRows: int
    nbCols: int
    plotSize: "tuple[float, float]"
    
    @classmethod
    def fromText(cls, datas:"dict[_FigureConfigField, str]")->"FigureConfig":
        # extract the plotSize
        plotSize = tuple(map(float, extractTextList(datas["plotSize"])))
        assert len(plotSize) == 2
        # contruct the new object
        return FigureConfig(
            figureID=datas["figureID"],
            nbRows=int(datas["nbRows"]),
            nbCols=int(datas["nbCols"]),
            plotSize=plotSize)
    
    def toJson(self)->"AsJson_FigureConfig":
        return AsJson_FigureConfig(
            cls=self.__class__.__name__, 
            figureID=self.figureID,
            nbRows=self.nbRows, nbCols=self.nbCols,
            plotSize=toJsonTuple(self.plotSize))
    
    @classmethod
    def fromJson(cls, datas:"AsJson_FigureConfig")->"Self":
        assert cls.__name__ == datas["cls"]
        plotConfig = FigureConfig.__new__(cls)
        FigureConfig.__init__(
            self=plotConfig, figureID=datas["figureID"], 
            nbRows=datas["nbRows"], nbCols=datas["nbCols"],
            plotSize=fromJsonTuple(tuple, datas["plotSize"]))
        return plotConfig

    def getFigure(self):
        fig = plt.figure(num=self.figureID, figsize=self.plotSize)
        fig.suptitle(f"Metrics plotter")
        return fig


@attrs.frozen(kw_only=True)
class PlotConfig(Generic[_MetricNameSingle, _MetricNameDfs], PrettyfyClass):
    axisConfigs: "dict[str, AxisConfig[_MetricNameSingle, _MetricNameDfs]]"
    figureConfig: FigureConfig
    
    def axisNameIsUsed(self, name:str)->bool:
        return name in self.axisConfigs.keys()
    
    def getCurrentAxisNames(self)->"set[str]":
        return set(self.axisConfigs.keys())
    
    def addAxisConfig(self, axisConfig:"AxisConfig[_MetricNameSingle, _MetricNameDfs]")->None:
        name: str = axisConfig.name
        if self.axisNameIsUsed(name) is True:
            raise KeyError(f"there is alredy an axis config with this name: {name}")
        # => can add
        self.axisConfigs[axisConfig.name] = axisConfig
    
    def getNewAxisName(self)->str:
        """return a valide new axis name like: 'axis n°...'"""
        # not optimized but works perfectly
        getName = lambda i: f"axis °{i}"
        currIndex: int = 0
        while self.axisNameIsUsed(getName(currIndex)):
            currIndex += 1
        return getName(currIndex)
    
    def removeAxisConfig(self, axisConfigName:str)->None:
        if self.axisNameIsUsed(axisConfigName) is False:
            raise KeyError(f"there is no axis named: {axisConfigName}")
        # => can remove
        self.axisConfigs.pop(axisConfigName)
    
    def replaceAxisConfig(self, oldAxisConfigName:str, newAxisConfig:"AxisConfig")->None:
        if oldAxisConfigName == newAxisConfig.name:
            # => same config edited (will not cause problems)
            self.removeAxisConfig(oldAxisConfigName)
            self.addAxisConfig(newAxisConfig)
        else: # => old config got renamed
            self.addAxisConfig(newAxisConfig) # add new (and check it is valid)
            self.removeAxisConfig(oldAxisConfigName) # remove old
        
    def getUsedMetrics(self)->"tuple[set[_MetricNameSingle], set[_MetricNameDfs]]":
        singleMetrics: "set[_MetricNameSingle]" = set()
        dfsMetrics: "set[_MetricNameDfs]" = set()
        for axisConfig in self.axisConfigs.values():
            singleMetrics.update(axisConfig.metricsConfigs.singleLinesConfigs.keys())
            dfsMetrics.update(axisConfig.metricsConfigs.dfsLinesConfigs.keys())
        return (singleMetrics, dfsMetrics)
    
    def replaceFigCfg(self, newfigCfg:"FigureConfig")->"PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        return PlotConfig(axisConfigs=self.axisConfigs, figureConfig=newfigCfg)
    
    @staticmethod
    def empty(figConf:"FigureConfig")->"PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        return PlotConfig(axisConfigs={}, figureConfig=copy.deepcopy(figConf))
        
    
    def emptyCopy(self)->"PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        """create an empty axis config, based on the figure config of self"""
        return PlotConfig.empty(self.figureConfig)
    

    def toJson(self)->"AsJson_PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        return AsJson_PlotConfig(
            cls=self.__class__.__name__,
            axisConfigs={cfg.name: cfg.toJson() for cfg in self.axisConfigs.values()},
            figureConfig=self.figureConfig.toJson())
    
    def __assertSupportsMetrics(self, allMetrics:"AllMetrics[_MetricNameSingle, _MetricNameDfs]")->None:
        singleMetrics, dfsMetrics = self.getUsedMetrics()
        missingSingleMetrics = singleMetrics.difference(allMetrics.singleMetrics.metrics)
        missingDfsMetrics = dfsMetrics.difference(allMetrics.dfsMetrics.metrics)
        if (len(missingSingleMetrics) != 0) or (len(missingDfsMetrics) != 0):
            raise KeyError(f"the loaded config has some metrics that "
                           f"arn't available in the metrics datas:\n"
                           f"\tbad single metrics: {missingSingleMetrics}"
                           f"\tbad dfs metrics: {missingDfsMetrics}")
    
    @classmethod
    def fromJson(cls, datas:"AsJson_PlotConfig[_MetricNameSingle, _MetricNameDfs]")->"Self":
        assert cls.__name__ == datas["cls"]
        plotConfig = PlotConfig.__new__(cls)
        PlotConfig.__init__(
            self=plotConfig,
            axisConfigs={name: AxisConfig.fromJson(cfg)
                         for name, cfg in datas["axisConfigs"].items()},
            figureConfig=FigureConfig.fromJson(datas["figureConfig"]))
        return plotConfig

    @staticmethod
    def fromFile(allMetrics:"AllMetrics[_MetricNameSingle, _MetricNameDfs]",
                 file:"BufferedReader")->"PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        # load from the file
        with file:
            config = PlotConfig.fromJson(json.load(file))
        # check that it can be used on thoses metrics
        PlotConfig.__assertSupportsMetrics(config, allMetrics)
        return config
    
    def saveToFile(self, filePath:Path)->None:
        with open(filePath, mode="w", encoding="utf-8") as file:
            json.dump(self.toJson(), file)

    
    def setupFigure(self, figure:"plt_Figure")->"list[tuple[AxisConfig[_MetricNameSingle, _MetricNameDfs], plt.Axes]]":
        nbRows: int = self.figureConfig.nbRows
        nbCols: int = self.figureConfig.nbCols
        axis: "list[tuple[AxisConfig, plt.Axes]]" = []
        for axisConfig in self.axisConfigs.values():
            ### create the axis
            ax = figure.add_subplot(nbRows, nbCols, axisConfig.indexs)
            axis.append(axisConfig.setupAxis(ax))
        # share x axis setup
        if len(axis) > 1:
            for (_, ax) in axis[1: ]:
                ax.sharex(axis[0][1])
        return axis