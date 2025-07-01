from scraperLib.extract import _TimeStamp, is_TimeStamp

from holo.__typing import NamedTuple, assertIsinstance
from holo import patternValidation


class TimeStampInfos(NamedTuple):
    expectedTime: "float|None"
    expectedMaxPeriodes: "int|None"

class SymbolConfig(NamedTuple):
    platformeName: str
    symbol: str
    timeFramesInfos: "dict[_TimeStamp, TimeStampInfos]"
    realName:"str|None" = None
    """None => use symbol"""
    
    @property
    def fullName(self)->str:
        return f"{self.platformeName}:{self.symbol}"
    
    @property
    def timeframes(self)->"set[_TimeStamp]":
        return set(self.timeFramesInfos.keys())
    
    def filterTimeFrame(self, tfs:"set[_TimeStamp]")->"SymbolConfig":
        return SymbolConfig(
            platformeName=self.platformeName, symbol=self.symbol, realName=self.realName,
            timeFramesInfos={k: v for (k, v) in self.timeFramesInfos.items() if (k in tfs)})
    
    def getKey(self, timeStamp:"_TimeStamp")->str:
        return f"{self.platformeName}_{self.symbol}_{timeStamp}"
    
    def expectedTime(self, tf:"_TimeStamp")->float:
        assert tf in self.timeframes, KeyError(f"the timestamp: {tf} isn't available")
        expectedTime: "float|None" = self.timeFramesInfos[tf].expectedTime
        return (float("+inf") if expectedTime is None else expectedTime)
    
    def expectedMaxPeriodes(self, tf:"_TimeStamp")->int:
        assert tf in self.timeframes, KeyError(f"the timestamp: {tf} isn't available")
        expectedMaxPeriodes: "int|None" = self.timeFramesInfos[tf].expectedMaxPeriodes
        return (int(1e6) if expectedMaxPeriodes is None else expectedMaxPeriodes)
    
    @classmethod
    def createFromText(cls, key:str)->"SymbolConfig":
        """use the following pattern: "<platformeName>_<symbol>_<timeFrame>" """
        ### search the key in the known configs first
        if key in CONFIGURED_KEYS:
            return CONFIGURED_KEYS[key]
        # => the key isn't known
        
        ### create a config manualy
        PATTERN = "<platformeName>_<symbol>_<timeFrame>"
        (matched, datas) = patternValidation(key, PATTERN)
        if matched is False:
            raise ValueError(f"the given string: {repr(key)} doesen't match the pattern: {PATTERN}")
        timeFrame = datas["timeFrame"]
        assert is_TimeStamp(timeFrame)
        
        return SymbolConfig(
            platformeName=assertIsinstance(str, datas["platformeName"]),
            symbol=assertIsinstance(str, datas["symbol"]),
            realName=None, timeFramesInfos={
                timeFrame: TimeStampInfos(expectedTime=None, expectedMaxPeriodes=None)})



CONFIG:"list[SymbolConfig]" = [
    ... # REMOVED
]

CONFIGURED_KEYS: "dict[str, SymbolConfig]" = {
    config.getKey(timeframe): config.filterTimeFrame({timeframe})
    for config in CONFIG 
        for timeframe in config.timeframes}
