from os import listdir as os_listDir
from pathlib import Path
from inspect import signature

import pandas as pd
import numpy

from paths_cfg import (
    DIRECTORY_PRICES_CSV_FILES,
    DIRECTORY_PRICES_HDF_FILES,
    DIRECTORY_PRICES_JSON_FILES,
)

from holo import patternValidation, assertIsinstance
from holo.__typing import (
    Callable, Tuple, Dict, List, Iterable,
    Literal, cast, TypeGuard, NamedTuple,
    Iterator, get_args,
)
from holo.prettyFormats import prettyfyNamedTuple

#if __name__ == '__main__':
#   import dtale # very long import (2s)
#   print("dtale imported")

_TimeStamp = Literal["1min", "3min", "5min", "15min", "1hour", 
                     "4hour", "12hour", "1day", "3day", "1week"]

def assertIsTimeStamp(timestamp:str)->"_TimeStamp":
    if timestamp not in get_args(_TimeStamp):
        raise ValueError(f"invalide timestamp: {timestamp}")
    return cast(_TimeStamp, timestamp)

_FileFormat = Literal["csv", "hdf", "json"]

fileFormat_to_Directory:"dict[_FileFormat, Path]" = {
    "csv": DIRECTORY_PRICES_CSV_FILES, 
    "hdf": DIRECTORY_PRICES_HDF_FILES, 
    "json": DIRECTORY_PRICES_JSON_FILES, 
}

HDF_KEY:str = "df" 
"""name that identify in the .hdf5 that need to be consistent"""

DEFAULT_FORMAT:"_FileFormat" = "hdf"

FILENAME_MESSAGE_PATTERN = "msg_prices_<platform>-<currency>_<timestamp>_<dateStart:s>_<houreStart:s>__<dateEnd:s>_<houreEnd:s>.json"


@prettyfyNamedTuple
class Interval(NamedTuple):
    start:int
    end:int
    
    def mergeWith(self, __other:"Interval")->"Interval":
        assert self.intersect(__other), \
            AssertionError(f"they must intersect in order to be merged")
        return Interval(min(self.start, __other.start), max(self.end, __other.end))
    
    def intersect(self, __other:"Interval")->bool:
        if self.fullyAfter(__other): return False
        elif self.fullyBefore(__other): return False
        else: return True
    
    def fullyBefore(self, __other:"Interval")->bool:
        """return whether self is fully before __other (=> no intersection)"""
        return self.end < __other.start
    
    def partialyBefore(self, __other:"Interval")->bool:
        """return whether self is partialy before __other (or starts at the same time)\n
        doesn't imply anything about the intersection"""
        return self.start <= __other.start
    
    def fullyAfter(self, __other:"Interval")->bool:
        """return whether self is fully after __other (=> no intersection)"""
        return __other.end < self.start
    
    def partialyAfter(self, __other:"Interval")->bool:
        """return whether self is partialy after __other (or ends at the same time)\n
        doesn't imply anything about the intersection"""
        return  __other.end <= self.end
    
    def fullyContain(self, __other:"Interval")->bool:
        """return whether __other is fully inside self (=> intersection)"""
        return (self.start <= __other.start) and (__other.end <= self.end)
    
    
    @classmethod
    def expand(cls, allIntervals:"list[Interval]", newInterval:"Interval")->"list[Interval]":
        doIntersect:"list[Interval]" = [
            interval for interval in allIntervals
            if interval.intersect(newInterval)
        ]
        
        if len(doIntersect) == 0: # => don't intersect with any current intervals
            return list(allIntervals) + [newInterval]
        # => have intersected some intervals
        # merge each intervals that intersect with the new one
        mergedIntervals:"list[Interval]" = [
            interval.mergeWith(newInterval) for interval in doIntersect
        ]
        dontIntersect:"list[Interval]" = [
            interval for interval in allIntervals
            if not interval.intersect(newInterval)
        ]
        return mergedIntervals + dontIntersect
    
    @classmethod
    def mergeAll(cls, allIntervals:"Iterable[Interval]")->"list[Interval]":
        mergedIntervals:"list[Interval]" = []
        toMergeintervals:"list[Interval]|None" = None # None <=> first loop
        while (toMergeintervals is None) or (len(mergedIntervals) != len(toMergeintervals)):
            if toMergeintervals is None: toMergeintervals = list(allIntervals)
            else: toMergeintervals = mergedIntervals
            mergedIntervals = []
            for interval in toMergeintervals:
                mergedIntervals = Interval.expand(mergedIntervals, interval)
        return mergedIntervals
    
    @classmethod
    def fromTimesList(cls, allIntervals:"Iterable[tuple[int, int]]")->"list[Interval]":
        return cls.mergeAll(map(lambda times: Interval(*times), allIntervals))

    @classmethod
    def sorted(cls, allIntervals:"Iterable[Interval]")->"list[Interval]":
        """mergeAll and sort the intervals in `allIntervals`"""
        return sorted(cls.mergeAll(allIntervals), key=lambda inter: inter.start)




def get_prices_filename_pattern(fileFormat:_FileFormat)->str:
    return f"Prices__<platform>-<currency>__<timestamp>__<t1:d>__<t2:d>.{fileFormat}"

def extractFileNameInfos(filename:str, pattern:"str|None"=None)->"tuple[bool, dict[str, int|float|str]]":
    if pattern is None:
        pattern = get_prices_filename_pattern(DEFAULT_FORMAT)
    return patternValidation(filename, pattern)

def have_correct_columns(df:pd.DataFrame)->bool:
    return all(df.columns == ['Open', 'High', 'Low', 'Close', 'Volume'])

def check_indexes(index:"pd.Index")->"TypeGuard[pd.Index[pd.Timestamp]]":
    return isinstance(index[0], pd.Timestamp)

def getInterval(df:pd.DataFrame)->Interval:
    indexes = df.index
    assert check_indexes(indexes)
    return Interval(start=indexes[0].value, end=indexes[-1].value)

def getMergedIntervals(liste_dataFrames:"list[pd.DataFrame]")->"list[Interval]":
    """return the merged intervals of the dfs in `liste_dataFrames`\n
    each resulting interval is the interval of each final concatenated df\n
    => no final intervals will intersect each other"""
    return Interval.mergeAll(map(getInterval, liste_dataFrames))

def group_by_finalIntervals(
        liste_dataFrames:"list[pd.DataFrame]",
        *, finalIntervals:"list[Interval]|Literal['auto']")->"list[list[pd.DataFrame]]":
    """return the list of [df that are in the same final interval]\n
    `liste_dataFrames`: the dfs to group\n
    `finalIntervals`: the merged intervals to regroup the dfs on ('auto' -> auto generated)\n
    all intervals of the dfs in `liste_dataFrames` MUST be fully contained in one of `finalIntervals`"""
    if finalIntervals == 'auto': finalIntervals = getMergedIntervals(liste_dataFrames)
    assert len(Interval.mergeAll(finalIntervals)) == len(finalIntervals)
    
    dfs_groups:"list[list[pd.DataFrame]]" = [list() for _ in finalIntervals]
    
    for df in liste_dataFrames:
        # determine in which final interval this df goes
        interval_df:Interval = getInterval(df)
        for index_tructated_dfs, finalInterval in enumerate(finalIntervals):
            if finalInterval.fullyContain(interval_df):
                break # => found it
        else: # => didn't found it
            raise RuntimeError(f"[BUG] couldn't find where to put {interval_df} in {finalIntervals}")
        # add the df to the correct group
        dfs_groups[index_tructated_dfs].append(df)
    
    return dfs_groups

def concatenate_DataFrames(liste_dataFrames:"list[pd.DataFrame]")->"pd.DataFrame":
    """concatenate each dataframes of liste_dataFrames\n
    merging their intervals MUST generate a single Interval"""
    assert len(getMergedIntervals(liste_dataFrames)) == 1
    
    # check all .index of the dfs
    for df in liste_dataFrames:
        assert check_indexes(df.index)
        assert have_correct_columns(df)
    
    if len(liste_dataFrames) == 1: # optim
        return liste_dataFrames[0]
    
    #print("merged interval:", getMergedIntervals(liste_dataFrames))
    #print("all sorted intervals:")
    #for df in sorted(liste_dataFrames, key=lambda df: getInterval(df).start):
    #    print(getInterval(df))
    sorted_liste_dataFrames:"Iterator[pd.DataFrame]" = iter(sorted(
        liste_dataFrames, key=lambda df: getInterval(df).start))
    """the list of dfs, sorted by start time"""
    del liste_dataFrames
    
    # => adding the dfs in this order imply that:
    #   - current_interval.start will never change
    #   - we only append some new dfs (truncated from their start)
    
    truncated_dfs:"list[pd.DataFrame]" = [next(sorted_liste_dataFrames)]
    current_interval:Interval = getInterval(truncated_dfs[0])
    
    # truncate the dfs
    for count, df in enumerate(sorted_liste_dataFrames, start=1):
        interval_df:Interval = getInterval(df)
        
        if current_interval.fullyContain(interval_df):
            continue # => the df is alredy contained in truncated_dfs
        # => is must be added to truncated_dfs => they intersect 
        assert not interval_df.fullyAfter(current_interval), \
            ValueError(f"at df at index {count}, interval_df: {interval_df} can't be fullyAfter current_interval: {current_interval}")
        
        last_truncated_df:pd.DataFrame = truncated_dfs[-1]
        assert getInterval(last_truncated_df).intersect(interval_df)
        
        # they were checked at teh start
        indexes_df = cast("pd.Index[pd.Timestamp]", df.index)
        indexes_last_truncated_df = cast("pd.Index[pd.Timestamp]", last_truncated_df.index)
        
        # determine where to truncate
        index_df_newDatas_start:int = \
            1 + int(assertIsinstance(numpy.int64, indexes_df.searchsorted(
                indexes_last_truncated_df[-1])))
        """the first index of the new datas of df (ie 1+ index of the last data alredy in truncated_dfs)"""
        truncated_dfs.append(df.iloc[index_df_newDatas_start: ])
        current_interval = current_interval.mergeWith(interval_df)
        
    return pd.concat(truncated_dfs)

def assembleDataFrames(liste_dataFrames:"list[pd.DataFrame]")->"list[pd.DataFrame]":
    """assemble all the dataframes it can in `liste_dataFrame` and return the new list of dataFrame"""
    if len(liste_dataFrames) == 0: return [] # nothing to do
    
    regrouped_dfs:"list[list[pd.DataFrame]]" = group_by_finalIntervals(
        liste_dataFrames=liste_dataFrames, finalIntervals='auto')
    del liste_dataFrames # not used later
    
    concatenated_dfs:"list[pd.DataFrame]" = []
    for dfs_to_prepare in regrouped_dfs:
        concatenated_dfs.append(concatenate_DataFrames(dfs_to_prepare))
    return concatenated_dfs
    
def splits_dataFrames(
        dataFrames_list:"list[pd.DataFrame]", splitEvery:"int|None"=None,
        maxNb_dfs_perSplit:"int|None"=None, minimalSize:int=0)->"list[pd.DataFrame]":
    """return a new list of dataframes\n
    `dataFrames_list`: the dataframes to split\n
    `splitEvery`: 
        * None -> don't split
        * int -> when a df is bigger than <splitEvery>,
            it create multiple blocks of size up to <splitEvery>\n
    `maxNb_dfs_perSplit`: max number of splits to do for each original df, 
        do nothing if `splitEvery` is None\n
    `minimalSize`: the minimal size of each final df returned,
        note: it applies for splited dfs and unsplited dfs"""
    if splitEvery is not None:
        # => isinstance(splitEvery, int)
        # => split every dfs in blocks of size up to <splitEvery>
        splitedDataFrames_list:"list[pd.DataFrame]" = []
        for df in dataFrames_list:
            countDfs:int = 0 
            iStart:int = 0
            dataFrame_splited:"pd.DataFrame" = df.iloc[iStart: iStart+splitEvery]
            while (len(dataFrame_splited) > 0) and ((maxNb_dfs_perSplit is None) or (countDfs < maxNb_dfs_perSplit)):
                splitedDataFrames_list.append(dataFrame_splited)
                countDfs += 1
                iStart += splitEvery
                dataFrame_splited = df.iloc[iStart: iStart+splitEvery]
        dataFrames_list = splitedDataFrames_list
    # else: # => don't split the blocks
    
    # only keep the dfs that have the required size (default(0) => keep all)
    return [df for df in dataFrames_list if len(df) >= minimalSize]

def loadAndAssembleDataFrames(
        fileName_liste:"list[str]", fileFormat:"_FileFormat", 
        directory:"Path|None"=None, splitEvery:"int|None"=None,
        nbDfs_perKeys:"int|None"=None, requiredSplitSize:int=0)->"list[pd.DataFrame]":
    """it will load and assemble(see assemeble doc) all the datasets from the files in the liste :\n
    `dirPath` must be the path to the directory where the files are.\n
    `fileName_liste` is the liste names (without extention) for files to assemble (the assembly order is the same as in the liste)."""
    if directory is None: directory = fileFormat_to_Directory[fileFormat]
    dataFrames:"list[pd.DataFrame]"
    if fileFormat == "csv": 
        dataFrames = [loadDataFrame_from_csv(fileName, directory)  for fileName in fileName_liste]
    elif fileFormat == "hdf":
        dataFrames = [loadDataFrame_from_hdf(fileName, directory)  for fileName in fileName_liste]
    else: raise NotImplementedError(f"unsupported fileFormat: {repr(fileFormat)}")
    
    #return assembleDataFrames(
    #    dataFrames, splitEvery=splitEvery, nbDfs_perKeys=nbDfs_perKeys,
    #    requiredSplitSize=requiredSplitSize)
    return splits_dataFrames(
        dataFrames_list=assembleDataFrames(dataFrames),
        splitEvery=splitEvery,
        maxNb_dfs_perSplit=nbDfs_perKeys,
        minimalSize=requiredSplitSize,
    )

def _getFullPath(filePath:"str|Path", dirPath:"Path|None"=None)->Path:
    if dirPath is not None: # => if a dirPath is given merge it with the filePath
        return Path(dirPath).joinpath(filePath)
    else: return Path(filePath)

def loadDataFrame_from_csv(filePath:"str|Path", dirPath:"Path|None"=None)->pd.DataFrame:
    """load a single dataFrame saved in the csv at `dirPath`->`filePath` and return it"""
    return pd.read_csv(_getFullPath(filePath, dirPath), parse_dates=True, index_col=0)

def loadDataFrame_from_hdf(filePath:"str|Path", dirPath:"Path|None"=None)->pd.DataFrame:
    """load a single dataFrame saved in the hdf at `dirPath`->`filePath` and return it"""
    dataFrame = pd.read_hdf(_getFullPath(filePath, dirPath), key=HDF_KEY)
    return assertIsinstance(pd.DataFrame, dataFrame)



def getDataFrames(
        datasInfos:"DataFilesInfos", directory:"Path|None",
        splitEvery:"int|None"=None, nbDfs_perKeys:"int|None"=None,
        requiredSplitSize:int=0)->"list[pd.DataFrame]":
    """get all the .csv with names corresponding to the files generated by `fileNameFunc` and the args in `timesList`
    (`dirPath` is the dir of all the files)\n
    the .csv will be loaded in dataFrames, then assembled with the following parameters:\n
    \t-`parse_dates` (see loadDataFrame_from_csv's doc)\n"""
    return loadAndAssembleDataFrames(
        fileName_liste=datasInfos.generateFileNames(),
        fileFormat=datasInfos.fileFormat, directory=directory,
        splitEvery=splitEvery, nbDfs_perKeys=nbDfs_perKeys,
        requiredSplitSize=requiredSplitSize,
    )

class FileID(NamedTuple):
    startTime: int
    endTime: int

class DataFilesInfos(NamedTuple):
    platform: str
    currency: str
    timestamp: "_TimeStamp"
    fileFormat: "_FileFormat"
    filesIntervals: "list[FileID]"
    
    def getFileName(self, fileID:"FileID")->str:
        return (f"Prices__{self.platform}-{self.currency}__{self.timestamp}"
                + f"__{fileID.startTime}__{fileID.endTime}.{self.fileFormat}")
        
    def generateFileNames(self)->"list[str]":
        return [self.getFileName(fileID) for fileID in self.filesIntervals]

    def getLastEndTime(self)->"int|None":
        """return the latest end of the files intervals (None -> no files)"""
        if len(self.filesIntervals) == 0:
            return None # => there is no datas
        return max(fileID.endTime for fileID in self.filesIntervals)
        

_Type_Datas_infos = Dict[str, DataFilesInfos]

    
    

def extractDatasInfos(fileFormat:"_FileFormat"=DEFAULT_FORMAT, directory:"Path|None"=None)->"_Type_Datas_infos":
    """extract all infos of the prices files that match the patern and return :\n
    \ta dictionary of tuples:\n
    \t\tthe key : $platofrme$_$currency$_$timestamp$,\n
    \t\tthe tuple contain :\n
    \t\t\tthe function that generate back the filename,\r\n
    \t\t\tthe liste(sorted by start time) of pairs of ints (the start and_ the end of the data))\n"""
    FILENAME_PATTERN = get_prices_filename_pattern(fileFormat)
    if directory is None: directory = fileFormat_to_Directory[fileFormat]
    
    liste_parsed_filenames = [(extractFileNameInfos(filename, FILENAME_PATTERN), filename) for filename in os_listDir(directory)]
    
    result:"_Type_Datas_infos" = {}
    for (matched, vars), filename in liste_parsed_filenames:
        if matched is True:
            categorie = f"{vars['platform']}_{vars['currency']}_{vars['timestamp']}"
            platform: str = assertIsinstance(str, vars['platform'])
            currency: str = assertIsinstance(str, vars['currency'])
            timestamp: _TimeStamp = assertIsTimeStamp(assertIsinstance(str, vars['timestamp']))
            t1: int = assertIsinstance(int, vars['t1'])
            t2: int = assertIsinstance(int, vars['t2'])
            if categorie in result: 
                result[categorie].filesIntervals.append(FileID(t1, t2))
            else: # create the categorie
                result[categorie] = DataFilesInfos(
                    platform=platform, currency=currency, timestamp=timestamp, 
                    fileFormat=fileFormat, filesIntervals=[FileID(t1, t2)])
            
        else: raise ValueError(f"the filename: {repr(filename)} didn't matched the pattern: {repr(FILENAME_PATTERN)}")

    # sort the lists of (t1, t2) # should alredy be done because listdir is sorted but to be sure
    for keys in result:
        result[keys].filesIntervals.sort(key=lambda fileID: fileID.startTime)

    return result



def get_all_dataframes_lists(
        filter_in:"None|list[str]"=None, filter_not_in:"None|list[str]"=None,
        splitEvery:"None|int"=None, nbDfs_perKeys:"int|None"=None,
        requiredSplitSize:int=0, fileFormat:"_FileFormat"=DEFAULT_FORMAT)->"dict[str, list[pd.DataFrame]]":
    Files_Prices:"_Type_Datas_infos" = extractDatasInfos(fileFormat)
    all_dataframes:"dict[str, list[pd.DataFrame]]" = {}
    for key in Files_Prices.keys():
        # if no match => skip
        if (filter_in is not None) and (not all(word in key   for word in filter_in)):
            continue
        # if match => skip
        if (filter_not_in is not None) and any(word in key   for word in filter_not_in):
            continue
        all_dataframes[key] = []
        for dataframe in getDataFrames(
                Files_Prices[key], fileFormat_to_Directory[fileFormat],
                splitEvery=splitEvery, nbDfs_perKeys=nbDfs_perKeys,
                requiredSplitSize=requiredSplitSize):
            all_dataframes[key].append(dataframe)
    return all_dataframes

def get_all_dataframes(
        filter_in:"None|list[str]"=None, filter_not_in:"None|list[str]"=None,
        splitEvery:"None|int"=None, nbDfs_perKeys:"int|None"=None,
        requiredSplitSize:int=0, fileFormat:"_FileFormat"=DEFAULT_FORMAT)->"dict[str, pd.DataFrame]":
    # get the dfs lists
    all_dataframes_lists:"dict[str, list[pd.DataFrame]]" = get_all_dataframes_lists(
        filter_in=filter_in, filter_not_in=filter_not_in, splitEvery=splitEvery,
        nbDfs_perKeys=nbDfs_perKeys, requiredSplitSize=requiredSplitSize,
        fileFormat=fileFormat,
    )
    # rearange the data
    all_dataframes:"dict[str, pd.DataFrame]" = {}
    for key, dataFrames in all_dataframes_lists.items():
        for index, dataframe in enumerate(dataFrames):
            all_dataframes[key + f"_{index}"] = dataframe
    return all_dataframes

assert signature(get_all_dataframes_lists).parameters == signature(get_all_dataframes).parameters, \
    ValueError(f"the definition of {get_all_dataframes_lists} and {get_all_dataframes} are not the same")




"""A = loadDataFrame_from_csv(
    Files_Prices["BINANCE_BTCUSDT_1min"][0](
        *Files_Prices["BINANCE_BTCUSDT_1min"][1][0]
    ),    dirPath=DIRECTORY_PRICES_CSV_FILES
)

B = loadDataFrame_from_csv(
    Files_Prices["BINANCE_BTCUSDT_1min"][0](
        *Files_Prices["BINANCE_BTCUSDT_1min"][1][1]
    ),    dirPath=DIRECTORY_PRICES_CSV_FILES
)

C = loadDataFrame_from_csv(
    Files_Prices["BINANCE_BTCUSDT_5min"][0](
        *Files_Prices["BINANCE_BTCUSDT_5min"][1][0]
    ),    dirPath=DIRECTORY_PRICES_CSV_FILES
)"""

"""
listeDataFrames_btc1 = getDataFrames(Files_Prices["BINANCE_BTCUSDT_1min"][1],
              Files_Prices["BINANCE_BTCUSDT_1min"][0],
              DIRECTORY_PRICES_CSV_FILES,
              parse_dates=True,
              checkIntervals=True)
listeDataFrames[0].Open.plot()



all_btc1_dataframes = [loadDataFrame_from_csv(Files_Prices["BINANCE_BTCUSDT_1min"][0](t1,t2),
                                             dirPath=DIRECTORY_PRICES_CSV_FILES, parse_dates=False)
                      for (t1, t2) in Files_Prices["BINANCE_BTCUSDT_1min"][1]]
"""


# patern = "msg_prices_$platforme$-$currency$_$timestamp$_$date1$_$hour1$__$date1$_$hour2$.json"
"""
A = pd.read_csv(DIRECTORY_PRICES_CSV_FILES +  "Prices__BINANCE-BTCUSDT__1min__1637491500__1637791440.csv",
                parse_dates=True, index_col=0)


B = pd.read_csv(DIRECTORY_PRICES_CSV_FILES +  "Prices__BINANCE-BTCUSDT__1min__1655723580__1656023520.csv",
                parse_dates=True, index_col=0)



C = pd.read_csv(DIRECTORY_PRICES_CSV_FILES +  "Prices__BYBIT-BTCUSDT__1min__1655766180__1656066120.csv",
                parse_dates=True, index_col=0)

C2 = pd.read_csv(DIRECTORY_PRICES_CSV_FILES +  "Prices__BINANCE-BTCUSDT__1min__1655723580__1656023520.csv",
                parse_dates=True, index_col=0)

dataset_C = pd.concat([C, C2], axis=1)
dataset_C.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Open2', 'High2', 'Low2',
       'Close2', 'Volume2']
"""

pass
