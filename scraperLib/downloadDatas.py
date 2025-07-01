import time
from datetime import datetime, timedelta, timezone
from math import ceil, floor
import termcolor, colorama
colorama.init()
from websocket._exceptions import (
    WebSocketAddressException, WebSocketTimeoutException, 
    WebSocketConnectionClosedException,
)
from io import StringIO
from threading import Lock

from .dataScrapingConfig import SymbolConfig
from .extract import (
    _TimeStamp, _StrPath, 
    getAndSaveMessagesPrices, nbSecondsToDate, extractAndSave,
    CONVERT_INTERVAL_TO_SECONDS, AUTOSAVE_MESSAGES_FILES_DIRECTORY,
)

from data.get_data import extractDatasInfos, DataFilesInfos, DEFAULT_FORMAT

from holo.parallel import Manager
from holo.linkedObjects import LinkedList
from holo.__typing import assertIsinstance, TypeAlias, Union
from holo.prettyFormats import prettyTime, print_exception


_ConectionException: TypeAlias = Union[
    WebSocketAddressException, ConnectionResetError, 
    WebSocketTimeoutException, WebSocketConnectionClosedException]


def getNowUTC()->datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def _prettyTimeSimple(nbSecs: int)->str:
    """to pretty formaty time in the case there is not enougth to download"""
    nbDays: int = nbSecs // (24*3600)
    nbSecs %= (24*3600)
    nbHours: int = nbSecs // 3600
    nbSecs %= 3600
    nbMinutes: int = nbSecs // 60
    nbSecs %= 60
    res: "list[str]" = []
    if nbDays > 0: res.append(f"{nbDays} day")
    if (nbHours > 0) and (nbDays < 7):
        res.append(f"{nbHours} h")
    if (nbMinutes > 0) and (nbDays == 0):
        res.append(f"{nbMinutes} m")
    return " ".join(res)

class Downloader(Manager):
    # to determine the number of periodes to download
    RATIO_MIN_PERIODES_TO_DL: float = 0.2
    """ratio of expectedMaxPeriodes before it is needed to download"""
    MAX_PERIODES_TO_DL: int = 50_000
    MARGIN_NB_PERIODES_TO_DL: float = 0.15
    """the margin (0.1 == +10 %) of additional periodes to download"""
    # managing the downloading frequency
    MAX_DL_PER_SEC: float = 1.8
    # when connection is lost
    NB_CONNEXTION_RETRYS: int = 25
    RETRY_DELAY: timedelta = timedelta(seconds=2)
    
    def __init__(self, symbolsToDo:"list[SymbolConfig]",
                 noMinPeriodes:bool, downloadMaxDatas:bool) -> None:
        self.totalWorkNb: int = sum(len(config.timeframes) for config in symbolsToDo)
        super().__init__(nbWorkers=self.totalWorkNb, startPaused=True)
        self.start_datas_infos = extractDatasInfos()
        
        self.lastDL_start: "datetime|None" = None
        self._symbolsWaitingToDL: "LinkedList[str]" = LinkedList()
        """all the keys of the downloads waiting to download (first is the next to download)"""
        self._sync_prints_lock = Lock()
        
        self.failedkeys: "list[str]" = []
        """all the keys of the downloads or transforms that have failed"""
        self.notDownloadedKeys: "list[str]" = []
        """all the keys of works that didn't had enough periodes to be downloaded"""
        self.downloadedKeys: "list[str]" = []
        """all the keys of works that didn't had enough periodes to be downloaded"""
        
        self.noMinPeriodes: bool = noMinPeriodes
        self.doDownloadMaxDatas: bool = downloadMaxDatas
        # if some slow symbols are done firsts there is a time gain at the end
        workToDo: "list[tuple[SymbolConfig, _TimeStamp]]" = [
            (config, tf) for config in symbolsToDo for tf in config.timeframes]
        workToDo = sorted(workToDo, key=lambda work: work[0].expectedTime(work[1]), reverse=True)
        # add the work
        for config, timestamp in workToDo:
            self.addWork(self.downloadAndTransformSymbol, config, timestamp)
        # init the manager and the workers and DONT start working
    
    def start(self)->None:
        self.unPause()
    
    def _waitToDL(self, config:SymbolConfig, timestamp:_TimeStamp)->None:
        # tell it is waiting to download
        thisKey: "str" = config.getKey(timeStamp=timestamp)
        self._symbolsWaitingToDL.append(thisKey) 
        # wait until it is its the first in the queue (=> the next symbol to download is it)
        while self._symbolsWaitingToDL.startValue() != thisKey:
            # => another symbol is waiting first
            time.sleep(0.001)
        # => the first symbol of the list is thiskey
        # wait the time it has to wait
        if self.lastDL_start is not None:
            # => needs to wait
            timeSinceLastDL_start: timedelta = (datetime.now() - self.lastDL_start)
            minimumTime: timedelta = timedelta(seconds=1) / self.MAX_DL_PER_SEC
            if timeSinceLastDL_start < minimumTime:
                time.sleep((minimumTime - timeSinceLastDL_start).total_seconds())
            # else: => time elapsed since last DL is alredy good
        # else: => first download => no need to wait
        
        # remove the symbol from the list to tell to other they can it is not waiting
        self.lastDL_start = datetime.now()
        popedKey = self._symbolsWaitingToDL.pop()
        assert popedKey is thisKey
    
    def _getNbPeriodesToDownload(self, pricesLastTime:"datetime|None", timestamp:_TimeStamp)->int:
        if pricesLastTime is None:
            # => there is no previous datas for this symbol # => download max
            return self.MAX_PERIODES_TO_DL
        # compute how much periodes to download (with a margin-)
        deltaPeriodes: float = \
            (getNowUTC() - pricesLastTime).total_seconds() / CONVERT_INTERVAL_TO_SECONDS[timestamp]
        nbPeriodesToDownload: int = ceil(deltaPeriodes * (1 + self.MARGIN_NB_PERIODES_TO_DL))
        if self.doDownloadMaxDatas is True:
            nbPeriodesToDownload = max(nbPeriodesToDownload, self.MAX_PERIODES_TO_DL)
        return nbPeriodesToDownload
    
    def __downloadMessages_getErrorText(self, err:"_ConectionException", datasKey:str)->str:
        if isinstance(err, (WebSocketAddressException, ConnectionResetError)):
            return (f"No internet found while downloading "
                    + f"key: {repr(datasKey)}, ... retrying{colorama.Fore.RESET}\n")
        elif isinstance(err, (WebSocketTimeoutException, WebSocketConnectionClosedException)):
            return (f"{colorama.Fore.LIGHTYELLOW_EX}timeout reached while downloading key: {repr(datasKey)}, "
                    + f"internet lost ?, ... retrying{colorama.Fore.RESET}\n")
        else: raise TypeError(f"invalide type ({type(err)}) for the error: {err}")
    
    def __sync_print(self, text:str, end:"str|None"=None)->None:
        with self._sync_prints_lock:
            print(text, end=end, flush=True)
    
    def __internal_downloadAndTransformSymbol(self, config:SymbolConfig, timestamp:_TimeStamp)->None:
        # get the true last downloaded time for this symbol and timestamp
        datas_key: str = config.getKey(timeStamp=timestamp)
        datas_infos: "DataFilesInfos|None" = self.start_datas_infos.get(datas_key)
        lastTime: "datetime|None"
        if datas_infos is None:
            lastTime = None # => new symbol
        else: # => not a new symbol (=> must have some datas)
            lastTime = nbSecondsToDate(assertIsinstance(int, datas_infos.getLastEndTime()))
        
        ### decide if it needs to download
        nbPeriodesToDL: int = self._getNbPeriodesToDownload(pricesLastTime=lastTime, timestamp=timestamp)
        expectedMaxPeriodes: int = config.expectedMaxPeriodes(timestamp)
        minPeriodesRequired: int = floor(self.RATIO_MIN_PERIODES_TO_DL * expectedMaxPeriodes)
        if (nbPeriodesToDL < minPeriodesRequired) and (self.noMinPeriodes is False):
            # => no needs to download
            timeToEnougthPeriodes: int = CONVERT_INTERVAL_TO_SECONDS[timestamp] * (minPeriodesRequired - nbPeriodesToDL)
            self.__sync_print(f"{colorama.Fore.LIGHTYELLOW_EX}skiped: {datas_key} "
                  f"asked: {nbPeriodesToDL} / {minPeriodesRequired} periodes "
                  f"({nbPeriodesToDL/minPeriodesRequired:.1%} - "
                  f"{_prettyTimeSimple(timeToEnougthPeriodes)} remaining){colorama.Fore.RESET}")
            self.notDownloadedKeys.append(datas_key)
            return None # don't raise an error
        # => has enough periodes to download
        
        ### download and save the messages
        nbRetrys: int = 0
        """nb of times it failed"""
        startTime: datetime = datetime.now()
        messagesPath: "_StrPath|None" = None
        while nbRetrys <= self.NB_CONNEXTION_RETRYS:
            # loops while it fails (first loop == first try)
            self._waitToDL(config=config, timestamp=timestamp)
            # => allowed to start downloading
            self.__sync_print(f"{datas_key} started downloading {nbPeriodesToDL} periodes")
            startTime = datetime.now()
            try: messagesPath = getAndSaveMessagesPrices(
                    currencie=config.fullName, timeStamp=timestamp, numberOfBarre=nbPeriodesToDL,
                    directory=AUTOSAVE_MESSAGES_FILES_DIRECTORY, verbosity=0) # takes ~2.5s with 5k data
            except (WebSocketAddressException, ConnectionResetError,
                    WebSocketTimeoutException, WebSocketConnectionClosedException) as err:
                self.__sync_print(f"{colorama.Fore.LIGHTYELLOW_EX}"
                      + f"{self.__downloadMessages_getErrorText(err=err, datasKey=datas_key)}"
                      + f"{colorama.Fore.RESET}")
                time.sleep(self.RETRY_DELAY.total_seconds())
                nbRetrys += 1
                continue
            else: break # => succesfully downloaded the message
        if nbRetrys > self.NB_CONNEXTION_RETRYS:
            raise TimeoutError(f"failed to download the messages for the key: {datas_key}, reached max retrys")
        
        assert messagesPath is not None
        ### convert it to prices and save it
        prices, savedFilePath = extractAndSave(
            filePath=messagesPath, dirPath_input="", # dirPath_input is "" because it is alredy in `messagesPath`
            dirPath_target=None, saveFormat=DEFAULT_FORMAT,
            overwrite=True)
        
        self.__sync_print(f"{colorama.Fore.LIGHTBLUE_EX}finished: {datas_key},"
              +f" downloaded {prices.nbPeriodes} / {nbPeriodesToDL} periodes"
              +f" in {prettyTime(datetime.now()-startTime)}{colorama.Fore.RESET}")
        
        self.downloadedKeys.append(datas_key)
        
        ### check that it didn't created a gap
        if lastTime is not None:
            prices_startTime: "datetime" = nbSecondsToDate(prices.getInterval()[0])
            if lastTime < prices_startTime:
                # => there is a gap 
                self.__sync_print(f"{colorama.Fore.RED}couldn't download enough datas, there is a gap for the key: {datas_key} "
                      + f"it is missing: {prettyTime(prices_startTime - lastTime)}{colorama.Fore.RESET}")
    
    
    def downloadAndTransformSymbol(self, config:SymbolConfig, timestamp:_TimeStamp)->None:
        try: self.__internal_downloadAndTransformSymbol(config=config, timestamp=timestamp)
        except Exception as err: # => there was an un handeled exception
            # get the text of the expression
            exception_str = StringIO()
            print_exception(err, file=exception_str)
            # print in a threaded safe way what happened
            datas_key: str = config.getKey(timeStamp=timestamp)
            self.failedkeys.append(datas_key) # mark as failed
            self.__sync_print(f"\n{colorama.Fore.RED}{'-'*50}\n"
                  + f"an exception happened while doing: {repr(datas_key)} ->\n"
                  + exception_str.getvalue() 
                  + f"\n{'-'*50}{colorama.Fore.RESET}")
            
