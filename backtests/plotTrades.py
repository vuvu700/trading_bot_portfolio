import pandas
from backtesting import Backtest, Strategy

from holo.__typing import TypeAlias, Tuple, TypedDict, NotRequired
from holo.types_ext import _Serie_Float
from holo.pointers import Pointer


_Trade: TypeAlias = Tuple[int, bool]
"""a tuple like: (index of the trade, True -> start a long | False -> start a short)"""
# use a tuple beacause it is MUCH faster than using a class :( 
#   and it will be used in performance critical components

class ExecuteTradesStartKwargs(TypedDict):
    trades: "list[_Trade]"
    tradesSize: "float"
    stopLoss: "float|None"
    takeProfit: "float|None"
    actionsDelay: "int"
    

class ExecuteTradesStrat(Strategy):
    def __init__(self, broker, data, kwargs:"ExecuteTradesStartKwargs")->None:
        super().__init__(broker, data, {})
        self.tradesToDO: "list[_Trade]" = kwargs["trades"]
        self.tradesSize: float = kwargs["tradesSize"]
        if self.tradesSize == 1.0:
            self.tradesSize = self._FULL_EQUITY
        self.stopLoss: "float|None" = kwargs["stopLoss"]
        self.takeProfit: "float|None" = kwargs["takeProfit"]
        self.actionsDelay: int = kwargs["actionsDelay"]
        
    
    def init(self):
        self.currentIndex: int = -1 # => (first next() -> currIndex=0)
        self.nextTradeIndex: int = 0
        """the index of the next trade in self.trades (-1 => finished)"""
        if len(self.tradesToDO) == 0:
            self.nextTradeIndex = -1
        elif self.tradesToDO[0][0] == 0:
            isLongAction: bool = self.tradesToDO[0][1]
            if isLongAction is True:
                self.buy(size=self.tradesSize)
            else: # => (isLongAction is False) => is a short
                self.sell(size=self.tradesSize)
            self.nextTradeIndex += 1

    def next(self):
        ### decide what to do the next step
        self.currentIndex += 1
        if self.nextTradeIndex == -1:
            return # => no more trades to do
        # => has a next trade
        (nextActionIndex, isLongAction) = self.tradesToDO[self.nextTradeIndex]
        nextActionIndex -= 1 # because it needs to decide for the actions for next (not for current)
        nextActionIndex += self.actionsDelay
        if nextActionIndex != self.currentIndex:
            assert nextActionIndex > self.currentIndex
            return # => the action needs to be done later
        # => nextActionIndex == self.currentIndex
        self.position
        self.nextTradeIndex += 1
        if self.nextTradeIndex == len(self.tradesToDO):
            # => last trade reached (close the current)
            self.nextTradeIndex = -1
            self.position.close()
            return 
        # do the action
        price: float = self._broker._adjusted_price(self.tradesSize, price=None)
        stopLossPrice: "float|None" = None
        takeProfitPrice: "float|None" = None
        if isLongAction is True:
            if self.stopLoss: stopLossPrice = price*(1 - self.stopLoss)
            if self.takeProfit: takeProfitPrice = price*(1 + self.takeProfit)
            self.buy(size=self.tradesSize, 
                     sl=stopLossPrice, tp=takeProfitPrice)
        else: # => (isLongAction is False) => is a short
            if self.stopLoss: stopLossPrice = price*(1 + self.stopLoss)
            if self.takeProfit: takeProfitPrice = price*(1 - self.takeProfit)
            self.sell(size=self.tradesSize,
                      sl=stopLossPrice, tp=takeProfitPrice)
    
def runAndPlot(
        datas:"pandas.DataFrame", trades:"list[_Trade]", 
        fileName:str, fees:float, leverage:int=1,
        bidAskSpread:float=0.0, tradesSize:float=1.0, actionsDelay:int=0,
        stopLoss:"float|None"=None, takeProfit:"float|None"=None)->"pandas.Series":
    """simultate the startegie on 1e9 of cash"""
    backtest = Backtest(
        data=datas, strategy=ExecuteTradesStrat, cash=1e9, commission=(fees+bidAskSpread), 
        trade_on_close=True, exclusive_orders=True, hedging=False, margin=(1/leverage))
    results = backtest.run(**ExecuteTradesStartKwargs(
        trades=trades, tradesSize=tradesSize, actionsDelay=actionsDelay,
        takeProfit=takeProfit, stopLoss=stopLoss))
    backtest.plot(results=results, filename=fileName, resample=False)
    return results
    
    
    
    
