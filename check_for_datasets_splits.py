import pandas as pd
from pathlib import Path
from time import perf_counter
from typing import Callable, Container
from typing_extensions import Literal
from datetime import datetime
import argparse
import sys
import locale
if sys.platform == "win32":
    locale.setlocale(locale.LC_TIME, "fr")
elif sys.platform == "linux":
    locale.setlocale(locale.LC_TIME, "fr_FR.utf8")
else: raise OSError(f"please implement a local for the platform: {sys.platform}")
import termcolor, colorama
colorama.init()


from data.get_data import (
    extractDatasInfos, _Type_Datas_infos, getDataFrames,
    loadDataFrame_from_csv, DEFAULT_FORMAT, _FileFormat,
    get_all_dataframes_lists
)

from holo import assertIsinstance

SPLITS_COUNT:"dict[str, int]" = {
    ... # REMOVED
}

def print_data_intervals_details(
        key:str, datas:"list[pd.DataFrame]", 
        dateFunc:"Callable[[datetime], str]", maxSizeKey:int=0)->None:
    print(f"{key:<{maxSizeKey}}:")
    for index, dataframe in enumerate(datas):
        startTime:datetime = assertIsinstance(datetime, dataframe.index[0])
        stopTime:datetime = assertIsinstance(datetime, dataframe.index[-1])
        print(f"\t{index:<2d} - ({len(dataframe):>7_d}, {dateFunc(startTime)} -> {dateFunc(stopTime)})")

def getDateFunc(simpleDate:bool)->"Callable[[datetime], str]":
    if simpleDate is True:
        return lambda date: date.strftime("%a %d %b %Y")
    else: return lambda date: str(date)

def show_splits_details(
        all_dataframes_list:"dict[str, list[pd.DataFrame]]",
        simpleDate:bool, showTotalSize:bool,
        symbols_to_show:"Container[str]|None")->None:
    """`all_dataframes_list`: the Dataframes to use\n
    `symbols_to_show`: the symbols to print (None => all)\n
    `simpleDate`: to have a more "human" readable date (only show the day, not the hours)\n
    `showTotalSize`: whether to show the total number of rows / cells (of the ones in `symbols_to_show`)"""
    dateFunc:"Callable[[datetime], str]" = getDateFunc(simpleDate)
    maxSizeKey:int = max(map(len, all_dataframes_list.keys()))
    total_nb_rows:int = 0
    total_nb_cells:int = 0
    
    if symbols_to_show is None: 
        symbols_to_show = all_dataframes_list.keys()

    for key, dfs_list in all_dataframes_list.items():
        if key not in symbols_to_show:
            continue # => don't show it => skip
        
        print_data_intervals_details(key, dfs_list, dateFunc, maxSizeKey)
        # count the number of rows and cells
        for df in dfs_list:
            total_nb_rows += len(df)
            total_nb_cells += len(df) * len(df.columns)

    if showTotalSize is True:
        print(f"total size of dataFrames: {total_nb_rows:_d} (=> total of {total_nb_rows * 5:_d} individual datas)")


def get_expected_nbOfSplits(key)->int:
    expectedNbSplits: "int|None" = SPLITS_COUNT.get(key, None)
    if expectedNbSplits is None:
        expectedNbSplits = 0
        print(f"{colorama.Fore.RED} /!\\ there is no value for the expected number of "
              + f"splits for the key: {repr(key)}, using 0 as default{colorama.Fore.RESET}")
    return expectedNbSplits

def check_split(key:str, Files_Prices:"_Type_Datas_infos")->bool:
    """check if the new number of splits is lower or equal to expected number of splits"""
    newNbOfSplits: int = len(getDataFrames(Files_Prices[key], directory=None, splitEvery=None))
    return newNbOfSplits == get_expected_nbOfSplits(key)

def check_splits(
        all_dataframes_list:"dict[str, list[pd.DataFrame]]",
        ignore_keys:"set[str]|None"=None)->"list[str]":
    """return the list of series with incorrect number of splits (ignore the series of `ignore_keys`)"""
    if ignore_keys is None: ignore_keys = set()
    checks_result:"list[tuple[str, bool]]" = []
    """all the keys where the number of splits is bad"""
    for key, dfs_list in all_dataframes_list.items():
        if key in ignore_keys: continue # => skip it
        # compare the current number of splits to the expected nb of splits
        oldNbOfSplits: int = get_expected_nbOfSplits(key)
        newNbOfSplits: int = len(dfs_list)
        checks_result.append((key, newNbOfSplits == oldNbOfSplits))
    return [key for (key, res) in checks_result if res is False]


def main(showAll:bool, simpleDateFromat:bool, 
         showTotalSize:bool, fileFormat:"_FileFormat")->None:
    
    all_dataframe_lists:"dict[str, list[pd.DataFrame]]" = get_all_dataframes_lists(
        splitEvery=None, nbDfs_perKeys=None, requiredSplitSize=0, fileFormat=fileFormat)
    badSplits:"list[str]" = check_splits(all_dataframe_lists)
    
    keys_to_show:"set[str]"
    if showAll is False: # => show the ones with bad splits count
        keys_to_show = set(badSplits)
    else: # => show all keys
        keys_to_show = set(all_dataframe_lists.keys())
    
    show_splits_details(
        all_dataframes_list=all_dataframe_lists, 
        symbols_to_show=keys_to_show,
        simpleDate=simpleDateFromat, showTotalSize=showTotalSize)
    
    if len(badSplits) == 0:
        termcolor.cprint("all splits are correct !", "blue")
    else: 
        termcolor.cprint("the keys that don't have the expected number of splits:", "red")
        termcolor.cprint(str(badSplits), "red")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("-all", action="store_true", help="show the stats for all keys (default: only show for missing keys)")
    parser.add_argument("-simpleDate", action="store_true", help="print the dates in the splits in a more simple and readable format")
    parser.add_argument("-showTotalSize", action="store_true", help="show total amount of rows and cells in the splits stats shown")
    args = parser.parse_args()
    
    main(showAll=assertIsinstance(bool, args.all),
         simpleDateFromat=assertIsinstance(bool, args.simpleDate),
         showTotalSize=assertIsinstance(bool, args.showTotalSize),
         fileFormat=DEFAULT_FORMAT)
    
    input("finished, press enter to exit ...")
