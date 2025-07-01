import attrs
import numpy
import numba

from holo.types_ext import _1dArray_Float, _2dArray_Float
from holo.__typing import Any, Literal, NamedTuple, TypeAlias, Self, get_args
from holo.prettyFormats import PrettyfyClass
from holo.pointers import Pointer

from modules.numbaJit import (
    fastJitter, floatArray, float64Array,
    floatMatrix, float64Matrix, floating, boolean,
)

from save_formats import AsJson_StrategieResults

_Status:TypeAlias = Literal["long", "short"]

## rq: long return formula (without lever):
#  amountAfter = amountBefore * (priceSell / priceBuy) * ((1-makerFees/100)*(1-takerFees/100)) 
# <=> amountAfter = amountBefore * (priceSell / priceBuy) * feesMult
# long return formula (with lever):
#  amountAfter = (amountBefore * lever / priceBuy * (1-makerFees/100)) * (priceSell * (1-takerFees/100)) - (amountBefore * (lever-1))
# <=> amountAfter = amountBefore * (lever * (priceSell / priceBuy) * feesMult) - amountBefore * (lever-1)
# <=> amountAfter = amountBefore * (lever * ((priceSell / priceBuy) * feesMult -1) +1)

@fastJitter(floating(floating, floating, floating, floating))
def computeLongReturn(priceStart:float, priceEnd:float, feesMult:float, lever:float)->float:
    """return the return of the trade as a long (so 0.0 => profits)\n
    (newMoney = (1 + return) * oldMoney)"""
    return lever * ((priceEnd / priceStart) * feesMult -1)

@fastJitter(floating(floating, floating, floating, floating))
def computeShortReturn(priceStart:float, priceEnd:float, feesMult:float, lever:float)->float:
    """return the return of the trade as a short (so 0.0 => profits)\n
    (newMoney = (1 + return) * oldMoney)"""
    return lever * ((priceStart / priceEnd) * feesMult -1)


def _strToListOfFloat(string:str)->"list[float]":
    """return [char1, ..., charSize]"""
    return [float(ord(char)) for char in string]

def _strFromListOfFloat(listOfFloat:"list[float]")->str:
    """return "..." """
    return "".join(chr(int(fChar)) for fChar in listOfFloat)

@fastJitter(floating(floating, floating))
def computeFeesMulti(makerFee:float, takerFee:float)->float:
    """compute the fees multiplicator such as amountAfterFees = feesMult * amountBeforeFees\n
    `makerFee` is the fees when making the trade\n
    `takerFee` is the fees when closing the trade"""
    return (1- makerFee) * (1- takerFee)

def _maxNaN(newValue:float, oldValue:float)->float:
    """return the max(newValue, oldValue) or newValue if oldValue is 'NaN'"""
    if oldValue == float("nan"): return newValue
    return max(newValue, oldValue)

def _minNaN(newValue:float, oldValue:float)->float: 
    """return the min(newValue, oldValue) or newValue if oldValue is 'NaN'"""
    if oldValue == float("nan"): return newValue
    return min(newValue, oldValue)


@attrs.frozen()
class StrategieResults(PrettyfyClass):
    # TODO: when possible add the stats for :
    #   avgLongProfit, avgShortProfit, 
    #   avgLongLoss, avgShortLoss, 
    #   longWinRate, shortWinRate,
    #   nbShorts, nbLongs,
    #   shortsExposureTime, longExposureTime,
    #   avgLongReturn, avgShortReturn,
    #   bestLongReturn, bestShortReturn,
    #   worstLongReturn, worstShortReturn,
    #   maxLongDuration, maxShortDuration,
    #   avgLongDuration, avgShortDuration,
    nbTrades: int
    """the number of trades closed"""
    returnResult: float
    """the resulting percenatge of benefits/loss(negative number) done"""
    durationPeriodes: int
    """number of periodes"""
    exposureTime: float
    """the percenatge of time a position was hold"""
    maxDrawDown: float
    """the worst return acheved (consider unclosed trades)"""
    avgLoss: float
    """the average loss (of lossing closed trades)"""
    maxDrawUp: float
    """the best return acheved (consider unclosed trades)"""
    avgProfit: float
    """the average profit (of profitable closed trades)"""
    winRate: float
    """percentage of the closed trades won"""
    avgTradeReturn: float
    """average return percentage on a single trade closed"""
    bestTradeReturn: float
    """best return percentage on a single trade closed"""
    worstTradeReturn: float
    """worst return percentage on a single trade closed"""
    maxTradeDuration: int
    """duration (nb periodes) of the longest trade closed"""
    avgTradeDuration: float
    """average duration (nb periodes) of the trades closed"""
    longAndHoldReturn: float
    """percentage of return for a long from the start to the end"""
    shortAndHoldReturn: float
    """percentage of return for a short from the start to the end"""
    startegieName: str
    """the name of the stategie used"""

    def toJson(self)->"AsJson_StrategieResults":
        return AsJson_StrategieResults(
            cls=self.__class__.__name__,
            kwargs={field: self.__dict__[field] 
                    for field in get_args(_ResultKargs)})
    @classmethod
    def fromJson(cls, datas:"AsJson_StrategieResults")->"Self":
        assert datas["cls"] == cls.__name__
        results = StrategieResults.__new__(cls)
        StrategieResults.__init__(self=results, **datas["kwargs"])
        return results

_ResultKargs = Literal[
    "nbTrades", "returnResult", "durationPeriodes", "exposureTime",
    "maxDrawDown", "avgLoss", "maxDrawUp", "avgProfit", "winRate",
    "avgTradeReturn", "bestTradeReturn", "worstTradeReturn", "maxTradeDuration", 
    "avgTradeDuration", "longAndHoldReturn", "shortAndHoldReturn", "startegieName"]
# check the keys
__Diffs = set(get_args(_ResultKargs)).symmetric_difference(StrategieResults.__annotations__.keys())
assert len(__Diffs) == 0, KeyError(f"there are keys that are invalides betwin _ResultKargs and StrategieSimpleResults: {__Diffs}")
del __Diffs


@attrs.frozen() # replace with custom one (in order to display the units)
class StrategieSimpleResults(PrettyfyClass):
    nbTrades: int
    """the number of trades closed"""
    returnResult: float
    """the resulting percenatge of benefits/loss(negative number) done"""
    durationPeriodes: int
    """number of periodes"""
    exposureTime: float
    """the percenatge of time a position was hold"""
    winRate: float
    """percentage of the closed trades won"""
    avgTradeReturn: float
    """average return percentage on a single trade closed"""
    avgTradeDuration: float
    """average duration (nb periodes) of the trades closed"""
    longAndHoldReturn: float
    """percentage of return for a long from the start to the end"""
    shortAndHoldReturn: float
    """percentage of return for a short from the start to the end"""
    startegieName: str
    """the name of the stategie used"""

    # fastEvaluate should not produce a "StrategieResults" but a simple `returnResult`
    @classmethod
    def _fromFastEvaluateResults(cls:"type[Self]", array:"_1dArray_Float", startegieName:str)->"StrategieSimpleResults":
        return StrategieSimpleResults(
            nbTrades=int(array[0]), returnResult=float(array[1]),
            durationPeriodes=int(array[2]), exposureTime=float(array[3]),
            winRate=float(array[4]), avgTradeReturn=float(array[5]),
            avgTradeDuration=float(array[6]), longAndHoldReturn=float(array[7]),
            shortAndHoldReturn=float(array[8]), startegieName=startegieName)

_SimpleResultKargs = Literal[
    "nbTrades", "returnResult", "durationPeriodes", "exposureTime", "winRate", "avgTradeReturn",
    "avgTradeDuration", "longAndHoldReturn", "shortAndHoldReturn", "startegieName"]
# check the keys
__Diffs = set(get_args(_SimpleResultKargs)).symmetric_difference(StrategieSimpleResults.__annotations__.keys())
assert len(__Diffs) == 0, KeyError(f"there are keys that are invalides betwin _SimpleResultKargs and StrategieSimpleResults: {__Diffs}")
del __Diffs

@attrs.frozen()
class TradeResult(PrettyfyClass):
    status: "_Status"
    """what kind of trade it was"""
    resultReturn: float
    """the resulting percenatge of benefits/loss(negative number) done"""
    startIndex: int
    """the index in the array the trade started"""
    closeIndex: int
    """the index in the array the trade closed"""

    @property 
    def duration(self)->int:
        return self.closeIndex - self.startIndex
    @property
    def isWinning(self)->int:
        return self.resultReturn > 0.0

class Strategie():
    ... # superclass to regroup the strategies
    
    #def __new__(cls) -> Self:
    #    if cls == Strategie:
    #        raise RuntimeError(f"no object of this class can be created")
    #    return super().__new__(cls)

@attrs.frozen()
class Strategie_basic(Strategie, PrettyfyClass):
    makerFee: float
    """fees when making the transaction (positive value)"""
    takerFee: float
    """fees when closing the transaction (positive value)"""
    longThreshold: float = 0.0
    """when senti[i] <= longThreshold -> go long"""
    shortThreshold: float = 1.0
    """when senti[i] >= shortThreshold -> go short"""
    maxLosseRate: "float|None" = None
    """value in R+ | when loosing more than `maxLosseRate` percentage, close the current trade"""
    maxPofitRate: "float|None" = None
    """value in R+ | when winning more than `maxPofitRate` percentage, close the current trade"""
    closeOnLastPeriode: bool = True
    """True -> close the trade, False -> forget the trade currently opened"""
    lever: float = 1.0
    """how much lever to trade with (ie. mult on volumes traded)"""
    
    @classmethod
    def getDefaultConfig(cls)->"Strategie_basic":
        return Strategie_basic(
            makerFee=0.02/100, takerFee=0.04/100,
            longThreshold=0.1, shortThreshold=0.9,
            maxLosseRate=None, maxPofitRate=None,
            closeOnLastPeriode=True,
        )
    
    def evaluate(self,
            price_closes:"_1dArray_Float", senti_array:"_1dArray_Float",
            tradesHistoryGrabber:"Pointer[list[TradeResult]]|None")->StrategieResults:
        """run the strategie and return the detailled results\n
        `price_closes` array of close price to trade with (T[-n] -> T[0])\n
        `senti_array` array of the senti indicator to use (T[-n] -> T[0])"""
        ... # REMOVED


    def fastEvaluate(self, price_closes:"_1dArray_Float", senti_array:"_1dArray_Float")->"StrategieSimpleResults":
        """run the strategie and return the simple results\n
        `price_closes` array of close price to trade with (T[-n] -> T[0])\n
        `senti_array` array of the senti indicator to use (T[-n] -> T[0])"""
        assert len(price_closes) == len(senti_array)
        simpleResults_array = _internal_Strategie_1__fastEvaluate(
            price_closes=price_closes, senti_array=senti_array, 
            makerFee=self.makerFee, takerFee=self.takerFee,
            longThreshold=self.longThreshold, shortThreshold=self.shortThreshold,
            maxLosseRate=(-1 if self.maxLosseRate is None else self.maxLosseRate),
            maxPofitRate=(-1 if self.maxPofitRate is None else self.maxPofitRate),
            closeOnLastPeriode=self.closeOnLastPeriode, lever=self.lever)
        return StrategieSimpleResults._fromFastEvaluateResults(
            array=simpleResults_array, startegieName=self.__class__.__name__)
        

    def fastMassEvaluate(self, price_closes:"_1dArray_Float", senti_matrix:"_2dArray_Float")->"list[StrategieSimpleResults]":
        """run the strategie and return the simple results\n
        `price_closes` array of close price to trade with (T[-n] -> T[0])\n
        `senti_matrix` all the array of the senti indicator to use (T[-n] -> T[0])"""
        nbSeries:int = len(senti_matrix)
        assert len(price_closes) == len(senti_matrix[0])
        simpleResults_matrix = _internal_Strategie_1__fastMassEvaluate(
            price_closes=price_closes, senti_matrix=senti_matrix, 
            makerFee=self.makerFee, takerFee=self.takerFee,
            longThreshold=self.longThreshold, shortThreshold=self.shortThreshold,
            maxLosseRate=(-1 if self.maxLosseRate is None else self.maxLosseRate),
            maxPofitRate=(-1 if self.maxPofitRate is None else self.maxPofitRate),
            closeOnLastPeriode=self.closeOnLastPeriode, lever=self.lever)
        startegieName:str = self.__class__.__name__
        return [
            StrategieSimpleResults._fromFastEvaluateResults(
                array=simpleResults_matrix[serieIndex], startegieName=startegieName)
            for serieIndex in range(nbSeries)]


@fastJitter(float64Array(floatArray, floatArray, floating, floating, floating,
                         floating, floating, floating, boolean, floating))
def _internal_Strategie_1__fastEvaluate(
        price_closes:"_1dArray_Float", senti_array:"_1dArray_Float", makerFee:float, takerFee:float,
        longThreshold:float, shortThreshold:float, maxLosseRate:float, maxPofitRate:float,
        closeOnLastPeriode:bool, lever:float)->"_1dArray_Float":
    ... # REMOVED


@fastJitter(parallel=True, definition=float64Matrix(floatArray, floatMatrix, floating, floating, floating, floating, floating, floating, boolean, floating))
def _internal_Strategie_1__fastMassEvaluate(
        price_closes:"_1dArray_Float", senti_matrix:"_2dArray_Float", makerFee:float, takerFee:float,
        longThreshold:float, shortThreshold:float, maxLosseRate:float, maxPofitRate:float,
        closeOnLastPeriode:bool, lever:float)->"_2dArray_Float":
    
    nbSeries:int = len(senti_matrix)
    simpleResults_matrix:"_2dArray_Float" = numpy.empty((nbSeries, 9), dtype=numpy.float64)
    
    for serieIndex in numba.prange(nbSeries):
        simpleResults_matrix[serieIndex] = _internal_Strategie_1__fastEvaluate(
            price_closes, senti_matrix[serieIndex], makerFee, takerFee,
            longThreshold, shortThreshold, maxLosseRate, maxPofitRate,
            closeOnLastPeriode, lever)
    
    return simpleResults_matrix
    