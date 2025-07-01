if __name__ == "__main__":
    raise RuntimeError("this file must be imported, not launched")

import pickle
import numpy
import numba
import pandas as pd
import scipy.stats
import itertools
import attrs


from .indicators_functions import (
    ... # REMOVED
)
### TODO: depreciated
from .entryPoints import (
    ... # REMOVED
)
from .entryPoints2 import (
    ... # REMOVED
)
from .ai_data import (
    ... # REMOVED
)
from modules.numbaJit import fastJitter
from save_formats import (
    toJsonFunc, fromJsonFunc, AsJson_Function,
    AsJson_SeriesGeneratorConfig, 
    AsJson_SingleSerieGeneratorConfig, 
    AsJson_MultiSerieGeneratorConfig,
)

from holo.__typing import (
    Any, Callable, Dict, Literal, Union, Tuple,
    JsonTypeAlias, assertListSubType, assertIsinstance,
    cast, get_args, Self, TypeGuard, 
)
from holo.types_ext import _Serie_Float, _Serie_Integer, _Serie_Boolean
from holo.profilers import Profiler
from holo.prettyFormats import prettyPrint, prettyTime
from holo import getDuplicated

_Kwargs = Dict[str, Any]
_Func_serie = Callable[..., _Serie_Float]
_Func_multiSerie = Callable[..., Tuple[_Serie_Float, ...]]

_SerieName = Literal[
    ... # REMOVED
    ]

validSerieNames: "set[_SerieName]" = set(get_args(_SerieName))
"""all the valide serie names"""


_SingleSerieKey = _SerieName
_MultiSeriesKey = Tuple[_SerieName, ...]

_SingleSerieValue = Tuple[_Func_serie, _Kwargs]
_MultiSerieValue = Tuple[_Func_multiSerie, _Kwargs]

_SingleSerieCfgItem = Tuple[_SingleSerieKey, _SingleSerieValue]
_MultiSerieCfgItem = Tuple[_MultiSeriesKey, _MultiSerieValue]
_ConfigItemGroup = Union[_SingleSerieCfgItem, _MultiSerieCfgItem]


_SingleSeriesGeneratorConfig = Dict[_SingleSerieKey, _SingleSerieValue]
_MultiSeriesGeneratorConfig = Dict[_MultiSeriesKey, _MultiSerieValue]

def _matchSingleCfgItem(item:"_ConfigItemGroup")->"TypeGuard[_SingleSerieCfgItem]":
    return isinstance(item[0], str)
def _matchMultiCfgItem(item:"_ConfigItemGroup")->"TypeGuard[_MultiSerieCfgItem]":
    return isinstance(item[0], tuple)


@attrs.frozen(kw_only=True)
class SeriesGeneratorConfig():
    singleSeriesCfg: _SingleSeriesGeneratorConfig = attrs.Factory(dict)
    multiSeriesCfg: _MultiSeriesGeneratorConfig = attrs.Factory(dict)
    
    
    def __internal_getAllSeries(self)->"list[_SerieName]":
        series: "list[_SerieName]" = [
            name for multiNames in self.multiSeriesCfg.keys() for name in multiNames]
        series.extend(self.singleSeriesCfg.keys())
        return series
    
    def getAllSeries(self)->"set[_SerieName]":
        """list all the series that can be computed with this config"""
        return set(self.__internal_getAllSeries())
    
    def assertUniquesSeries(self)->None:
        """check that all the series are only given once"""
        series: "list[_SerieName]" = self.__internal_getAllSeries()
        dups = getDuplicated(series)
        if len(dups) != 0:
            raise KeyError(f"there are duplicated series in the configs: {dups}")
    
    def getCfgItemForSerie(self, serieName:"_SerieName")->"_ConfigItemGroup":
        """get the config to generate the serie asked\n
        will assume the configs for each series are unique"""
        if serieName in self.singleSeriesCfg:
            return (serieName, self.singleSeriesCfg[serieName])
        # else try to get it from multi series
        for multiNames, cfg in self.multiSeriesCfg.items():
            if serieName in multiNames:
                return (multiNames, cfg)
        # => didn't find a config
        raise KeyError(f"couldn't find a config for the serie: {serieName}")
            
    def addOther(self, other: "SeriesGeneratorConfig")->None:
        self.addSingle(other.singleSeriesCfg)
        self.addMulti(other.multiSeriesCfg)
    
    def addSingle(self, other: _SingleSeriesGeneratorConfig)->None:
        """add the cfg of other to self\n
        will overwrite any config on the same serie"""
        for name, cfg in other.items():
            self.singleSeriesCfg[name] = cfg
        self.assertUniquesSeries()
            
    
    def addMulti(self, other: _MultiSeriesGeneratorConfig)->None:
        """add the cfg of other to self\n
        will overwrite any config on the same serie"""
        for names, cfg in other.items():
            self.multiSeriesCfg[names] = cfg
        self.assertUniquesSeries()
    
    
    def toJson(self)->"AsJson_SeriesGeneratorConfig":
        asJson = AsJson_SeriesGeneratorConfig(
            cls=self.__class__.__name__,
            singleSeriesGenerator=[], multiSeriesGenerator=[])
        # single series
        for serieName, (func, kwargs) in self.singleSeriesCfg.items():
            kwargs = kwargs.copy()
            for kwarg, value in kwargs.items():
                if isinstance(value, Callable):
                    kwargs[kwarg] = toJsonFunc(value)
            asJson["singleSeriesGenerator"].append(
                AsJson_SingleSerieGeneratorConfig(
                    serieName=serieName, func=toJsonFunc(func), kwargs=kwargs))
            del serieName, func, kwargs
        
        # multi series
        for seriesNames, (func, kwargs) in self.multiSeriesCfg.items():
            kwargs = kwargs.copy()
            for kwarg, value in kwargs.items():
                if isinstance(value, Callable):
                    kwargs[kwarg] = toJsonFunc(value)
            asJson["multiSeriesGenerator"].append(
                AsJson_MultiSerieGeneratorConfig(
                    seriesNames=list(seriesNames), func=toJsonFunc(func), kwargs=kwargs))
            del seriesNames, func, kwargs
        
        return asJson

    @staticmethod
    def __fromJsonSingle(datas:"list[AsJson_SingleSerieGeneratorConfig]")->_SingleSeriesGeneratorConfig:
        seriesGeneratorConfig: "_SingleSeriesGeneratorConfig" = {}
        for serieGenConfig in datas:
            kwargs: "_Kwargs" = serieGenConfig["kwargs"].copy()
            for attr, value in kwargs.items():
                if isinstance(value, dict) \
                        and (value.keys() == AsJson_Function.__annotations__.keys()):
                    # because all keys are required
                    kwargs[attr] = fromJsonFunc(cast(AsJson_Function, value))
            seriesGeneratorConfig[serieGenConfig["serieName"]] = (
                fromJsonFunc(serieGenConfig["func"]), kwargs)
        return seriesGeneratorConfig
    
    @staticmethod
    def __fromJsonMulti(datas:"list[AsJson_MultiSerieGeneratorConfig]")->_MultiSeriesGeneratorConfig:
        seriesGeneratorConfig: "_MultiSeriesGeneratorConfig" = {}
        for serieGenConfig in datas:
            kwargs: "_Kwargs" = serieGenConfig["kwargs"].copy()
            for attr, value in kwargs.items():
                if isinstance(value, dict) \
                        and (value.keys() == AsJson_Function.__annotations__.keys()):
                    # because all keys are required
                    kwargs[attr] = fromJsonFunc(cast(AsJson_Function, value))
            seriesGeneratorConfig[tuple(serieGenConfig["seriesNames"])] = (
                fromJsonFunc(serieGenConfig["func"]), kwargs)
        return seriesGeneratorConfig

    @classmethod
    def fromJson(cls, datas:"AsJson_SeriesGeneratorConfig")->"Self":
        assert datas["cls"] == cls.__name__
        self = SeriesGeneratorConfig.__new__(cls)
        SeriesGeneratorConfig.__init__(
            self=self, 
            singleSeriesCfg=self.__fromJsonSingle(datas["singleSeriesGenerator"]),
            multiSeriesCfg=self.__fromJsonMulti(datas["multiSeriesGenerator"]))
        self.assertUniquesSeries()
        return self
        
        
       

FEES_MAKER, FEES_TAKER = 0.0200/100, 0.0500/100
"""default fees on binance"""
def FEES_FUNC(fMaker:float, fTaker:float)->float:
    return 1- (1 - fTaker)*(1 - fMaker)
FEES:float = FEES_FUNC(FEES_MAKER, FEES_TAKER)
"""default fees on binance"""


_profilerSeriesCalculation = Profiler(list(validSerieNames), nbMesurements=None)
    


# important :
# when speaking of index of series inputed and outputed of the calculations functions:
# - 0 is the most recent data
# - len(array)-1 is the oldest data



def get_default_series_generator(nbPeriodes:int)->"SeriesGeneratorConfig":
    return SeriesGeneratorConfig(singleSeriesCfg={
    "SMA": (sma_numpy_serie, {"array":"COLUMN:Close", "nbPeriodes":nbPeriodes}),
    "VWMA": (vwma_numpy_serie, {"array":"COLUMN:Close", "volumeArray":"COLUMN:Volume", "nbPeriodes":nbPeriodes}),
    "SMMA": (smma_numpy_serie, {"array":"COLUMN:Close", "nbPeriodes":nbPeriodes}),
    "EMA": (ema_numpy_serie, {"array":"COLUMN:Close", "nbPeriodes":nbPeriodes}),
    "EMA_n":(rolling_rerange_serie, {"array":"SERIE:EMA", "nbPeriodes":nbPeriodes}),
    
    ... # REMOVED
    })

def get_senti_serie_rerange_generator(
        useHighLowPrices:bool, totalFees:float, targetedRatio:float,
        minTradeDuration:int, smoothingKwargs:"None|EmaSmoothKwargs", 
        rerange3Kwargs:"None|ReRange3Kwargs")->"SeriesGeneratorConfig":
    return SeriesGeneratorConfig(singleSeriesCfg={
    "SENTI": (compute_senti_serie_rerange, {
        "arrayHigh":("COLUMN:High" if useHighLowPrices else "COLUMN:Close"), 
        "arrayLow":("COLUMN:Low" if useHighLowPrices else "COLUMN:Close"), 
        "arrayPrice":"COLUMN:Close", "totalFees":totalFees, 
        "targetedRatio":targetedRatio, "minTradeDuration":minTradeDuration, 
        "smoothingKwargs":smoothingKwargs, "rerange3Kwargs":rerange3Kwargs})
    })

def get_senti_serie_spike_generator(
        useHighLowPrices:bool, totalFees:float, targetedRatio:float,
        minTradeDuration:int, smoothingKwargs:"None|EmaSmoothKwargs", 
        rerange3Kwargs:"None|ReRange3Kwargs")->"SeriesGeneratorConfig":
    return SeriesGeneratorConfig(singleSeriesCfg={
    "SENTI": (compute_senti_serie_spike, {
        "arrayHigh":("COLUMN:High" if useHighLowPrices else "COLUMN:Close"), 
        "arrayLow":("COLUMN:Low" if useHighLowPrices else "COLUMN:Close"), 
        "totalFees":totalFees, "targetedRatio":targetedRatio,
        "minTradeDuration":minTradeDuration, "smoothingKwargs":smoothingKwargs,
        "rerange3Kwargs":rerange3Kwargs})
    })

def get_senti_serie_profit_generator(
        useHighLowPrices:bool, totalFees:float, targetedRatio:float,
        minTradeDuration:int, minimumSentiTradeRatio:float, stepHeight:float,
        useToogleCurve:bool, allowStartBefore:bool, stepNeedsStreak:bool,
        smoothingKwargs:"None|EmaSmoothKwargs", rerange3Kwargs:"None|ReRange3Kwargs",
        )->"SeriesGeneratorConfig":
    return SeriesGeneratorConfig(singleSeriesCfg={
    "SENTI": (compute_senti_serie_profit, {
        "arrayHigh":("COLUMN:High" if useHighLowPrices else "COLUMN:Close"), 
        "arrayLow":("COLUMN:Low" if useHighLowPrices else "COLUMN:Close"), 
        "arrayPrice":"COLUMN:Close", "totalFees":totalFees, "stepHeight":stepHeight,
        "minTradeDuration":minTradeDuration, "allowStartBefore":allowStartBefore,
        "stepNeedsStreak":stepNeedsStreak, "useToogleCurve":useToogleCurve,
        "targetedRatio":targetedRatio, "minimumSentiTradeRatio":minimumSentiTradeRatio,
        "smoothingKwargs":smoothingKwargs, "rerange3Kwargs":rerange3Kwargs})
    })


def get_actions_series_profits_generator(
        useHighLowPrices:bool, totalFees:float, targetedRatio:float,
        minTradeDuration:int, neutral_profit_threshold:float,
        transformCfg:"ActionsTransformsCfg|None")->"SeriesGeneratorConfig":
    return SeriesGeneratorConfig(multiSeriesCfg={
    ("ACT_long", "ACT_neutral", "ACT_short"): (
        compute_actions_series_profits, {
        "arrayHigh":("COLUMN:High" if useHighLowPrices else "COLUMN:Close"), 
        "arrayLow":("COLUMN:Low" if useHighLowPrices else "COLUMN:Close"), 
        "arrayPrice":"COLUMN:Close", "totalFees":totalFees, 
        "minTradeDuration": minTradeDuration, "targetedRatio": targetedRatio,
        "neutral_profit_threshold": neutral_profit_threshold, "transformCfg": transformCfg})
    })

def get_actions_series_profits_rng_generator(
        useHighLowPrices:bool, totalFees:float, targetedRatio:float,
        minTradeDuration:int, neutral_profit_threshold:float,
        computeOppositAction:bool, nbIterations: int, meanDuration: float,
        stdDuration: float, minDuration: int, manualScaling: float,
        transformCfg:"ActionsTransformsCfg|None")->"SeriesGeneratorConfig":
    return SeriesGeneratorConfig(multiSeriesCfg={
    ("ACT_long", "ACT_neutral", "ACT_short"): (
        compute_actions_series_profits_fromRandom, {
        "arrayHigh":("COLUMN:High" if useHighLowPrices else "COLUMN:Close"), 
        "arrayLow":("COLUMN:Low" if useHighLowPrices else "COLUMN:Close"), 
        "arrayPrice":"COLUMN:Close", "totalFees":totalFees, 
        "minTradeDuration": minTradeDuration, "targetedRatio": targetedRatio,
        "neutral_profit_threshold": neutral_profit_threshold, "nbIterations": nbIterations,
        "meanDuration": meanDuration, "stdDuration": stdDuration, "minDuration": minDuration, 
        "manualScaling": manualScaling, "transformCfg": transformCfg,
        "computeOppositAction": computeOppositAction})
    })
    
    


def generate_calculationSeries(
        prices_DataFrame:pd.DataFrame, config:"SeriesGeneratorConfig",
        seriesDtype:"type[numpy.floating]")->"dict[_SerieName, _Serie_Float]":
    """it will generate all the calculations of the series specified in `seriesAndParams`
    based on the data in `prices_DataFrame`\n
    `seriesDtype` is the dtype of the outputed series of float\n
    the dict `seriesAndParams` is constructed as :\n
    \t-the KEY is the name of the serie\n
    \t-the VALUE[0] is the callable that will be used\n
    \t-the VALUE[1] is the parameters that will be used\n
    for the params, if one of the params is a string that begin with :\n
    \t-"COLUMN:..." -> the param will be replaced by `prices_DataFrame`[...].to_numpy()\n
    \t-"SERIE:..." -> try to replace the param with the corresponding serie \
    from `seriesAndParams`\n
    ps: the inverted indexes needed by the funcs are automatically done,\
    then automatically reverted to normal\n"""
    def add_serie(serieName:"_SerieName")->None:
        """add the serie to `calculated_series`, \
        using the inputed parameters and solving dependencies of other series\n
        ps: the inverted indexes needed by the funcs are automatically done, \
        then automatically reverted to normal"""
        if serieName in calculated_series: 
            return # skip because already calculated
        
        serieCfg: "_ConfigItemGroup" = config.getCfgItemForSerie(serieName)
        #calcSerieFunc:_Func_serie = series_and_kwargs[serieName][0]
        funcConfig: "_SingleSerieValue | _MultiSerieValue" = serieCfg[1]
        keywords: _Kwargs = funcConfig[1]
        
        # parse all the params to search for COLUMN, or SERIE
        newKeywords:_Kwargs = {}
        for key, key_value in keywords.items():
            if isinstance(key_value, str) and key_value.startswith(("COLUMN:", "SERIE:", "INDEX")):
                if key_value.startswith("COLUMN:"):
                    # => use the data from the dataframe
                    newKeywords[key] = prices_DataFrame[key_value[len("COLUMN:"): ]].\
                        to_numpy(dtype=seriesDtype)[:: -1]
                elif key_value == "INDEX":
                    # => use the data from the dataframe
                    newKeywords[key] = prices_DataFrame.index.to_numpy()[:: -1]
                elif key_value.startswith("SERIE:"): 
                    # => starts with "SERIE:" => use the data from the calculated series
                    serieOtherName = key_value[len("SERIE:"): ]
                    if serieOtherName in config.getAllSeries():
                        add_serie(serieOtherName)
                    else: raise KeyError(f"tryed to add the sub serie:{serieOtherName}, but isn't in `series_and_kwargs`")
                    newKeywords[key] = calculated_series[serieOtherName][:: -1]
                else: raise KeyError(f"invalide key_value: {key_value}")
            
            else: # => basic arg, not a COLUMN / SERIE
                newKeywords[key] = key_value
        
        # calc the new serie
        #match serieCfg:
        #    case (str(), _):
        #        serieCfg
        #    case (tuple(), _):
        #        serieCfg
        #    case _:
        #        serieCfg
        
        if _matchSingleCfgItem(serieCfg):
            calcSerieFunc: "_Func_serie" = serieCfg[1][0]
            try:
                with _profilerSeriesCalculation.mesure(serieName):
                    calculated: "_Serie_Float" = calcSerieFunc(**newKeywords)[:: -1]
            except Exception as err:
                print(f"an error happend durring the calculation of {serieName}, "
                    + f"with func: {calcSerieFunc.__name__}")
                raise err
            # add the new serie and fix the dtype if needed
            calculated_series[serieName] = (
                calculated.astype(seriesDtype) if calculated.dtype != seriesDtype else calculated)
        elif _matchMultiCfgItem(serieCfg):
            seriesNames: "_MultiSeriesKey" = serieCfg[0]
            funcMulti: "_Func_multiSerie" = serieCfg[1][0]
            try:
                with _profilerSeriesCalculation.mesure(serieName) as sp:
                    multiCalculated: "tuple[_Serie_Float, ...]" = funcMulti(**newKeywords)
                assert len(multiCalculated) == len(seriesNames)
            except Exception as err:
                print(f"an error happend durring the calculation of {serieName}, "
                    + f"with func: {funcMulti.__name__}")
                raise err
            for name, serie in zip(seriesNames, multiCalculated):
                # add the new serie and fix the dtype if needed
                calculated_series[name] = (
                    serie.astype(seriesDtype) if serie.dtype != seriesDtype else serie)
                # add the profiling mesure for the other
                if name == serieName: continue
                _profilerSeriesCalculation.addManualMesure(name, mesuredTime=sp)
        else: raise TypeError(f"")

    # assert the config is valid
    config.assertUniquesSeries()

    # compute all the series
    calculated_series:"dict[_SerieName, _Serie_Float]" = dict()
    for seriesKey in config.getAllSeries():
        add_serie(seriesKey)

    # add all the series previously on `prices_DataFrame`
    for columnName in prices_DataFrame.columns.to_list():
        assert columnName in validSerieNames, \
            ValueError(f"invalide column name: {columnName}")
        calculated_series[cast(_SerieName, columnName)] = \
            prices_DataFrame[columnName].to_numpy(dtype=seriesDtype)

    for serieName, calculatedSerie in calculated_series.items():
        if calculatedSerie.dtype != seriesDtype:
            raise TypeError(f"invalide dtype for the calculated serie: {serieName}, "
                            f" got a: {repr(calculatedSerie.dtype)}, but expected: {seriesDtype}")

    return calculated_series

def generate_calculationSeries_dataFrame(
        prices_DataFrame:pd.DataFrame, config:"SeriesGeneratorConfig")->pd.DataFrame:
    """it will generate all the calculations of the series specified in `seriesAndParams`
    based on the data in `prices_DataFrame`\n
    /!\\ a new DataFrame will be generated (the old data and indexes are copied)\n
    the dict `seriesAndParams` is constructed as :\n
    \t-the KEY is the name of the serie\n
    \t-the VALUE[0] is the callable that will be used\n
    \t-the VALUE[1] is the parameters that will be used\n
    for the params, if one of the params is a string that begin with :\n
    \t-"COLUMN:..." -> the param will be replaced by `prices_DataFrame`[...].to_numpy()\n
    \t-"SERIE:..." -> try to replace the param with the corresponding serie \
    from `seriesAndParams`\n
    ps: the inverted indexes needed by the funcs are automatically done, \
    then automatically reverted to normal\n"""
    calculated_series = generate_calculationSeries(
        prices_DataFrame=prices_DataFrame, config=config, seriesDtype=numpy.float64)

    return pd.DataFrame(
            data=calculated_series,
            index=prices_DataFrame.index,
            copy=True)

def printTotalCalculationTime()->None:
    prettyPrint(sorted(_profilerSeriesCalculation.totalTimes().items(), 
                       key=lambda pair: pair[1]), 
                specificFormats={float: prettyTime}, specificCompact={tuple})

########################## correlation

def corrWithGrp(corrMatrix:pd.DataFrame, target:str, columns:"list[str]")->float:
    """return the average correlation betwin the column `target` and the ones in `columns`"""
    if len(columns) == 0: return 0.
    result = 0.
    count = 0
    for col in columns:
        result += corrMatrix[target][col]
        count += 1
    return result / count

def group_by_correlation(dataFrame:pd.DataFrame, threshold_corr_grp:float=0.6,
                         threshold_corr_allInGrp:float=0.6)->"list[list[str]]":
    """try to determine groups betwin the calculated series in `dataFrame`, using correlation\n
    `threshold_corr_grp` represent the correlation minimal of a column to the grp in order to be added\n
    `threshold_corr_allInGrp` represent the correlation minimal of a column with all the other columns
     of the current grp in order to be added\n"""
    columns:"list[str]" = dataFrame.columns.to_list()
    corrMatrix:"pd.DataFrame" = dataFrame.corr()
    result:"list[list[str]]" = [[col] for col in columns] # base liste


    for _ in range(len(columns)):
        for indexPart, group in enumerate(result):
            # add the column that fit the best to the current group
            bestCol:"str|None" = None
            bestCorr:float = -1.
            for col in columns:
                if col not in group:
                    corr = corrWithGrp(corrMatrix, col, group)
                    if (corr > bestCorr) and all([corrMatrix[col][col2] > threshold_corr_allInGrp  for col2 in group]):
                        bestCol = col
                        bestCorr = corr

            if (bestCol != None) and (bestCorr >= threshold_corr_grp):
                result[indexPart].append(bestCol)

    #liste
    return [group_unique.split(', ')      # pop the duplicated grps
            for group_unique in set(
                ", ".join(sorted(group)) for group in result)]


@numba.jit(numba.int64[:](numba.bool_[:]), **fastJitter.getJitKwargs())
def _transform_boolMask_to_intMask(boolMask:_Serie_Boolean)->_Serie_Integer:
    intMaskSize:int = numpy.sum(boolMask)
    intMask:_Serie_Integer = numpy.empty((intMaskSize, ), dtype=numpy.int64)
    indexIntMask:int = 0
    for indexBoolMask in range(boolMask.shape[0]):
        if boolMask[indexBoolMask] == 0:
            intMask[indexIntMask] = indexBoolMask
            indexIntMask += 1
    return intMask

def correlation_perPercentil(
        dataFrame:"pd.DataFrame|dict[str, _Serie_Float]", targetSerie:str, percentils:"list[float]|None"=None,
        corrMethode:str="spearman")->pd.DataFrame:
    """compute the correlation of each series in `dataFrame` compared to the `targetSerie` for each percentil in `percentils`.\n
    `dataFrame` is the df conatining all the series where the corr will be calculated\n
    `targetSerie` is the name the series of reference that will be splited in percentils\n
    `percentils` is the list of percentils that will be used, each percentil is a float in [0.0, 100.0],\
        by default it is [10., 20., 30., ..., 100.0], \
        will be sorted (no side effect), 100.0 will be added if needed, duplicates will not count\n
    `corrMethode` is the name of the calculation methode for the correlation, \
        supported methodes are: ("spearman", "pearson", "kendall")"""

    def calcPercentilsMask(serieValues:_Serie_Float, percentils:"list[float]")->"dict[float, _Serie_Integer]":
        """compute the mask (whether inside or outside of the percentil) for each percentil in `percentils`\n
        `percentils` must be sorted, contain 100. and no duplicates"""
        # compute the values of each percentils percentils
        percentils_bis:"list[float]" = [0.] + percentils
        percentilsValue:"dict[float, float]" = {
            percent:value
            for percent, value in zip(percentils, numpy.percentile(serieValues, percentils_bis))
        }
        percentilsValue[0.] -= 1e-6 # because for the count it wouldn't had been included

        # create the sub series
        percentilsMask:"dict[float, _Serie_Integer]" = {}

        for indexPercentil in range(1, len(percentils_bis)):
            percent:float = percentilsValue[indexPercentil]
            percentPrev:float = percentilsValue[indexPercentil-1]

            # find all values that are in the percentil
            percentilsMask[percent] = _transform_boolMask_to_intMask(
                (serieValues > percentPrev) & (serieValues <= percent)
            )
        return percentilsMask

    # set percentils if unset
    if percentils is None:
        percentils = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100.,]
    # treatment over percentils (refer to the func's doc)
    percentils = sorted(list(set(percentils + [100.0])))
    # assert the percentils are valides
    if (percentils[0] < 0.0) or (percentils[-1] > 100.0):
        raise ValueError(f"the values of percentils are incorrect, please refer to the func's doc: {percentils}")

    # transforme the dataFrame to dict of numpy arrays
    data_series:"dict[str, _Serie_Float]"
    if isinstance(dataFrame, pd.DataFrame):
        data_series = {
            serie: dataFrame[serie].to_numpy()
            for serie in list(map(str, dataFrame.columns))}
    else: data_series = dataFrame

    # check the targeted serie is in the dataFrame and there is at least another serie
    if targetSerie not in data_series.keys():
        raise KeyError("`dataFrame` must contain the `targetSerie`")
    if len(data_series.keys()) < 2:
        raise KeyError("`dataFrame` must contain at least another serie than `targetSerie`")

    # set the selected correlation methode
    corrFunction:"Callable[[_Serie_Float, _Serie_Float], float]"
    if corrMethode == "spearman":
        corrFunction = lambda serie1, serie2: scipy.stats.spearmanr(serie1, serie2).correlation
    elif corrMethode == "pearson":
        corrFunction = lambda serie1, serie2: scipy.stats.pearsonr(serie1, serie2)[0]
    elif corrMethode == "kendall":
        corrFunction = lambda serie1, serie2: scipy.stats.kendalltau(serie1, serie2).correlation
    else: raise ValueError(f"the `corrMethode`: {corrMethode} isn't supported, pleas refer to the func's doc")


    nbPercentils:int = len(percentils)
    percentilsMask:"dict[float, _Serie_Integer]" = \
        calcPercentilsMask(data_series[targetSerie], percentils)

    # compute the correlations of each series with each percentil
    seriesCorrelations:"dict[str, _Serie_Float]" = \
        {serie: numpy.empty((nbPercentils, ))
         for serie in data_series.keys() if serie != targetSerie}

    for percentilIndex in range(nbPercentils):
        percent:float = percentils[percentilIndex]
        maskPercentil:_Serie_Integer = percentilsMask[percent]
        targetSerieValues:_Serie_Float = data_series[targetSerie][maskPercentil]

        for serie in seriesCorrelations.keys():
            seriesCorrelations[serie][percentilIndex] = \
                corrFunction(targetSerieValues, data_series[serie][maskPercentil])

    # create and return the dataFrame containint the results
    return pd.DataFrame(
        data=seriesCorrelations,
        index=percentils,
        copy=True)




