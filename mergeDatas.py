from pandas import Timestamp
from typing import cast
from typing_extensions import get_args, Literal
from time import perf_counter
from pathlib import Path
import os

from data.get_data import (
    extractDatasInfos, _Type_Datas_infos, getDataFrames,
    HDF_KEY, DEFAULT_FORMAT, _FileFormat, fileFormat_to_Directory, 
    DataFilesInfos, FileID,
)

from holo import prettyTime, print_exception
from holo.parallel import ProcessManager, Task

_Verbose = Literal[0, 1, 2, 3, 4]

def merge_key(categorie:str, fileFormat:"_FileFormat", 
              datasInfos:"DataFilesInfos", directory:"Path", 
              removeOldFiles:bool, verbose:"_Verbose")->None:
    tCategorieStart:float = perf_counter()
    newFilesName: "set[str]" = set()
    if verbose >= 2: print(f">>> starting: {categorie}")
    
    tGetDfsStart: float = perf_counter()
    mergedDataframes = getDataFrames(datasInfos, directory=None, splitEvery=None) 
    # -> the dataframes it gets is the merged results
    if verbose >= 3: print(f"\t({categorie}) dataframes loaded in {prettyTime(perf_counter() - tGetDfsStart)}")
    
    tAllSplitsStart:float = perf_counter()
    for splitNumber, splitedDataframe in enumerate(mergedDataframes, start=1):
        tSplitStart:float = perf_counter()
        if verbose >= 4: print(f"\t({categorie}) doing split n° {splitNumber}")
        indexFirst:"Timestamp" = splitedDataframe.index[0]
        indexLast:"Timestamp" = splitedDataframe.index[-1]
        
        fileName:str = datasInfos.getFileName(
            FileID(startTime=int(indexFirst.timestamp()), 
                    endTime=int(indexLast.timestamp())))
        newFilesName.add(fileName)
        
        if fileFormat == "csv":
            splitedDataframe.to_csv(directory.joinpath(fileName))
        elif fileFormat == "hdf":
            splitedDataframe.to_hdf(directory.joinpath(fileName), key=HDF_KEY)
        elif fileFormat == "json":
            splitedDataframe.to_json(directory.joinpath(fileName))
        else: raise ValueError(f"\t({categorie}) unsupported saveFromat: {repr(fileFormat)}")
        
        if verbose >= 4: print(f"\t({categorie}) split n°{splitNumber} saved in {prettyTime(perf_counter() - tSplitStart)}")
    if verbose >= 3: print(f"\t({categorie}) all splits saved in {prettyTime(perf_counter() - tAllSplitsStart)}")
        
    if removeOldFiles is True:
        tRemoveStart:float = perf_counter()
        for fileID in datasInfos.filesIntervals:
            fileName_oldFile:str = datasInfos.getFileName(fileID)
            if fileName_oldFile in newFilesName:
                continue # => don't supress new files if alredy merged
            os.remove(directory.joinpath(fileName_oldFile))
        if verbose >= 4: print(f"\t({categorie}) removed old files in {prettyTime(perf_counter() - tRemoveStart)}")
    
    if verbose >= 2: print(f" -> ({categorie}) done in {prettyTime(perf_counter() - tCategorieStart)}")



def main(fileFormat:"_FileFormat"=DEFAULT_FORMAT, removeOldFiles:bool=True,
         keysToMerge:"set[str]|None"=None, parallel:bool=False, verbose:"_Verbose"=1)->None:
    tStart:float = perf_counter()
    Files_Prices:"_Type_Datas_infos" = extractDatasInfos(fileFormat=fileFormat)
    if verbose >= 4: print(f"dataframes infos loaded in {prettyTime(perf_counter() - tStart)}")
    
    if keysToMerge is None: 
        # => all keys available will be merged
        keysToMerge = set(Files_Prices.keys())
    
    
    directory: "Path" = fileFormat_to_Directory[fileFormat]
    mergeTasks: "list[Task[None]]" = []
    for categorie, datasInfos in Files_Prices.items():
        if (categorie not in keysToMerge):
            continue # => don't merge this key
        mergeTasks.append(Task(
            merge_key, categorie=categorie, datasInfos=datasInfos, directory=directory,
            fileFormat=fileFormat, removeOldFiles=removeOldFiles, verbose=verbose))
    
    # parallelExec(mergeTasks, nbWorkers=(None if parallel is True else 1))
    if parallel is True:
        manager = ProcessManager(nbWorkers=12, startPaused=False, newContext=False)
        manager.runBatch(mergeTasks)
    else: 
        # execute sequencialy in main thread
        for task in mergeTasks:
            task.func(*task.funcArgs, **task.funcKwargs)
    
    
    
    if verbose >= 1: print(f"finished merging in {prettyTime(perf_counter() - tStart)}")

if __name__ == "__main__":
    try:
        from argparse import ArgumentParser
        
        parser = ArgumentParser(__file__)
        parser.add_argument(
            "--format", action="store", default=DEFAULT_FORMAT, choices=get_args(_FileFormat))
        parser.add_argument(
            "--verbose", action="store", default=1, choices=get_args(_Verbose), type=int)
        parser.add_argument("--parallel", action="store_true", default=True)
        args = parser.parse_args()
        
        main(fileFormat=cast(_FileFormat, args.format), removeOldFiles=True,
             keysToMerge=None, parallel=args.parallel, verbose=args.verbose)
        
    except Exception as err:
        print_exception(err)
    
    finally: input("press enter to exit ...")
    