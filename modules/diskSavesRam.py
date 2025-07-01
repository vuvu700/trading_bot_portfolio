import os
import os.path
import numpy
import pandas
import threading
from random import randint
import time
import weakref
import gc
import warnings

from typing import (
    Generic, TypeVar, Type, Generator,
    Any, Union, List, Mapping, Dict,
    Callable, Iterable,
)
from typing_extensions import Self

NoneType = type(None)

#from paths_cfg import TEMP_DIRECTORY

from holo.files import correctDirPath, mkDirRec, get_unique_name
from holo import print_exception


warnings.warn("this module will be depreciated, do not use in senti3", category=DeprecationWarning)

TEMP_DIRECTORY = correctDirPath("C:/Users/vuvu7/AppData/Local/Temp/")

SESSION_ID:int = randint(int(1e8), int(1e9))

SESSION_UPDATE_AFTER:int = 1*60 # update the session every 1min
SESSION_DELETE_AFTER:int = 30*60 # an unupdated session for 30min is deleted (in case some actions are very long, due to GIL: cant update)
FILENAME_SESSION_INFOS:str = "session.txt"

SAVEMODULE_SUB_DIRECTORY = correctDirPath(TEMP_DIRECTORY + ".SaveModule/")

CREATE_SESSION_DIRNAME = lambda: get_unique_name(
    SAVEMODULE_SUB_DIRECTORY, onlyNumbers=False, nbCharacters=8, guidlike=True,
    randomChoice=True, prefix="session_", suffix="/",
    filter_dirnames=True, filter_filename=False,
)
SAVEMODULE_SESSION_DIRECTORY = correctDirPath(SAVEMODULE_SUB_DIRECTORY + CREATE_SESSION_DIRNAME())

_PRINT_infos:bool = False
_SESSION_verbose:int = 0

#######                                                                               #######
###             WARNING : when in greedy mode inplace modifications                       ###
###       of the exported object dont completely affect the original object               ###
###      all the modifications are applied when the object's __del__ is triger            ###
#######                                                                               #######

#### about typing
_Storable = Union[
    numpy.ndarray,
    pandas.DataFrame,
    Dict[Any, numpy.ndarray],
    List[Any],
]

_DictStorable = Union[
    Dict["_KT_DictM", numpy.ndarray],
    Dict["_KT_DictM", pandas.DataFrame],
    Dict["_KT_DictM", List[Any]],
    Dict["_KT_DictM", "_DictStorable"]
]

_STORABLE_CHECK_ISINSTANCE = (
    numpy.ndarray,
    pandas.DataFrame,
    list,
    NoneType,
)


_VT_ObjM = TypeVar(
    "_VT_ObjM",
    bound="numpy.ndarray | pandas.DataFrame | list | dict[Any, numpy.ndarray]"
)

_KT_DictM = TypeVar("_KT_DictM")
_VT_DictM = TypeVar(
    "_VT_DictM",
    bound="SaveDictModule | _Storable",
)

_T = TypeVar("_T")
_KT = TypeVar("_KT")


_ShapeType = TypeVar("_ShapeType", bound=Any)
_DType_co = TypeVar("_DType_co", covariant=True, bound="numpy.dtype[Any]")
###


# infos about storing dfs at
# https://tech.blueyonder.com/efficient-dataframe-storage-with-apache-parquet/
# https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d

def _check_SaveDir_path():
    ## create the dir to store the files in the TEMP (if needed)
    global SAVEMODULE_SUB_DIRECTORY, SAVEMODULE_SESSION_DIRECTORY
    if not os.path.exists(TEMP_DIRECTORY):
        raise FileNotFoundError(f"the TEMP_DIRECTORY: {TEMP_DIRECTORY} isn't found")
    if not os.path.exists(SAVEMODULE_SESSION_DIRECTORY):
        os.makedirs(SAVEMODULE_SESSION_DIRECTORY)
    else: # => session alredy exist (should not happend)
        SAVEMODULE_SESSION_DIRECTORY = correctDirPath(SAVEMODULE_SUB_DIRECTORY + CREATE_SESSION_DIRNAME())
        _check_SaveDir_path()

# is called to avoid problems
_check_SaveDir_path()

# clean the old cache
def _clean_saveDir()->None:
    for sessionDir in os.listdir(SAVEMODULE_SUB_DIRECTORY):
        sessionDirPath = correctDirPath(SAVEMODULE_SUB_DIRECTORY + sessionDir)
        if sessionDirPath == SAVEMODULE_SESSION_DIRECTORY: continue
        if not os.path.isdir(sessionDirPath): continue # => it is not a session
        # get the sesion file
        try:
            sessionFile = open(sessionDirPath + FILENAME_SESSION_INFOS, mode='r')
            sessionLastUpdate:float = float(sessionFile.read())
            #print(f"session: {sessionDir} dt={time.time() - sessionLastUpdate}")
            if (time.time() - sessionLastUpdate) > SESSION_DELETE_AFTER:
                sessionFile.close()
                os.remove(sessionDirPath + FILENAME_SESSION_INFOS)
                for filename in os.listdir(sessionDirPath):
                    os.remove(sessionDirPath + filename)
                os.rmdir(sessionDirPath)
                #print(f"session: {sessionDir} cleaned")
        except PermissionError as err:
            pass # file is being accessed => session is running
            #print("PermissionError:", err)
        except FileNotFoundError as err:
            # => no session file
            if len(os.listdir(sessionDirPath)) == 0:
                os.rmdir(sessionDirPath)

def disableDebugPrinting()->None:
    global _PRINT_infos
    _PRINT_infos = False
def enableDebugPrinting()->None:
    global _PRINT_infos
    _PRINT_infos = True

##### Session manager

def copyStr(string:str)->str:
    return "".join([char for char in string])

def cleanFile_if_exist(filePath:str)->None:
    if os.path.lexists(filePath) is True:
        os.remove(filePath)


class Session(threading.Thread):
    session:"Session|None" = None

    def __init__(self, verbose:int=0)->None:
        """forced to be daemon=True\\
        verbose: 1-> all, 2-> updates&clean&cleanForgoten, 3->cleanForgoten, 4+ -> nothing"""
        super().__init__(daemon=True)
        self.verbose:int = (verbose if verbose > 0 else int(1e9))
        self.__tracked_objects:"dict[str, weakref.finalize]" = {}
        self.objects_count:int = 0
        self.set_session(self)
        self.start()

    @classmethod
    def set_session(cls, session:"Session")->None:
        if Session.session is None:
            Session.session = session
        else: raise RuntimeError("a session if alredy setted")

    @classmethod
    def get_session(cls)->"Session":
        if Session.session is None:
            raise RuntimeError("no session setted")
        return Session.session

    def track_object(self, file_path:str, object_finalizer:weakref.finalize)->None:
        """copy the tmp_file_path to leave no reference"""
        self.__tracked_objects[copyStr(file_path)] = object_finalizer
        if self.verbose <= 1: print(f"tracking {file_path}")

    def untrack_object(self, file_path:str)->bool:
        """clean the return false if the object was not tracked, only un-track"""
        finalizer:"weakref.finalize|None" = self.__tracked_objects.get(file_path, None)
        if finalizer is not None: # => objetc is tracked, clean it and un-track
            if finalizer.alive is True:
                finalizer()
                if self.verbose <= 2:
                    print(f"{file_path} cleaned")
            else:
                cleanFile_if_exist(file_path)
                self.__tracked_objects.pop(file_path)
                if self.verbose <= 2:
                    print(f"{file_path} untracked")
            return True
        # => not tracked
        return False

    def get_unique_id(self)->int:
        self.objects_count += 1
        return (self.objects_count - 1)

    def update_session(self)->None:
        with open(SAVEMODULE_SESSION_DIRECTORY + FILENAME_SESSION_INFOS, mode="w") as sessionFile:
            sessionFile.write(f"{time.time():.03f}")
            sessionFile.flush()

    def clean_other_sessions(self)->None:
        _clean_saveDir()

    def clean_forgoten_objects(self)->None:
        gc.collect()
        items_to_pop:"list[str]" = []
        # clean the object un-tracked
        for (file_path, object_finalizer) in self.__tracked_objects.items():
            if object_finalizer.alive is False:
                cleanFile_if_exist(file_path)
                items_to_pop.append(file_path)
        # un-track the objects
        for file_path in items_to_pop:
            self.__tracked_objects.pop(file_path)
            if self.verbose <= 3:
                print(f"poped forgoten {file_path}")

    def run(self)->None:
        while True:
            try:
                if self.verbose <= 1: print(f"updating session")
                self.update_session()
                self.clean_other_sessions()
                self.clean_forgoten_objects()
                time.sleep(SESSION_UPDATE_AFTER) # update every

            except Exception as err:
                print("an error happened durring update")
                print_exception(err)


Session(verbose=_SESSION_verbose) # can be garbed with Session.get_session()

##### autoSaves modules

class objectAutoSave():
    def add_container(self, obj:"SaveObjModule")->"Self":
        self._container = obj
        return self
    
    def __del__(self)->None:
        if hasattr(self, "_container"):
            class_mro:"tuple[type, ...]" = self.__class__.__mro__
            super_class:"type|None" = (None if len(class_mro) <= 1 else class_mro[1])
            self._container.setValue(self, newType=super_class, _FromAutoSaveCall=True) # TODO
            print("auto deleted")


class ListAutoSave(list, Generic[_T],  objectAutoSave):
    @classmethod
    def createFromList(cls,  liste:Iterable[_T], container:"SaveObjModule")->"ListAutoSave[_T]":
        return ListAutoSave(liste).add_container(container)

_VT_AutoDict = TypeVar("_VT_AutoDict", bound=numpy.ndarray)
class DictAutoSave(dict, Generic[_KT, _VT_AutoDict], objectAutoSave):
    @classmethod
    def createFromDict(cls, dictionary:"dict[_KT, _VT_AutoDict]", container:"SaveObjModule")->"DictAutoSave[_KT, _VT_AutoDict]":
        return DictAutoSave(dictionary).add_container(container)

class NDarrayAutoSave(numpy.ndarray, Generic[_ShapeType, _DType_co], objectAutoSave):
    @classmethod
    def createFromArray(cls,
            array:"numpy.ndarray[_ShapeType, _DType_co]", container:"SaveObjModule"
            )->"NDarrayAutoSave[_ShapeType, _DType_co]":
        return NDarrayAutoSave(shape=array.shape, dtype=array.dtype, buffer=array).add_container(container)

class DataFrameAutoSave(pandas.DataFrame, objectAutoSave):
    @classmethod
    def createFromDataFrame(cls,
            dataFrame:pandas.DataFrame, container:"SaveObjModule")->"pandas.DataFrame":
        return DataFrameAutoSave(dataFrame, copy=False).add_container(container) # type:ignore "add_container" is not a serie



######


class SaveObjModule(Generic[_VT_ObjM]):
    """a module intended to save large amounts of data \
    on the disk and be able to restaure it in the RAM"""

    #__slot__ = ...

    _Filename_prefix = "__SaveModuleFile_"
    __ConvNpyArrSaveName = "array"


    def __init__(self,
            value:"_VT_ObjM", saveCompressed:bool=False,
            greedySaves:bool=False,
            saveDirectory:"str|None"=None)->None:
        if saveDirectory is None: saveDirectory = SAVEMODULE_SESSION_DIRECTORY
        # keep the value and its type
        self.__value:"_VT_ObjM|None" = value
        self.__valueType:"Type[_VT_ObjM]" = type(value)

        # determine where to save it and how
        self.__filePath:str = self._genFilePath(saveDirectory)
        self.__isSaved:bool = False
        self.__compress:bool = saveCompressed
        self.__greedySaves:bool = greedySaves
        self.__autoSaveClasse_given:bool = False

        if self.__greedySaves is True:
            self.save()

        Session.get_session().track_object(
            self.__filePath, weakref.finalize(self, cleanFile_if_exist, copyStr(self.__filePath))
        )


    @property
    def value(self)->"_VT_ObjM":
        self.load() # no nothing if alredy loaded
        if self.__value is None: raise TypeError(f"unintended type:{type(self.__value)}")
        if self.__greedySaves is False:
            return self.__value
        # => __greedySaves = True
        if self.__autoSaveClasse_given is True:
            raise RuntimeError("while in greedy mode, the value has alredy been given and is still alive")
        # => not an auto save object => create it
        self.__autoSaveClasse_given = True
        if isinstance(self.__value, list):
            return ListAutoSave.createFromList(self.__value, self)
        elif isinstance(self.__value, dict):
            return DictAutoSave.createFromDict(self.__value, self)
        elif isinstance(self.__value, numpy.ndarray):
            return NDarrayAutoSave.createFromArray(self.__value, self)
        elif isinstance(self.__value, pandas.DataFrame):
            return DataFrameAutoSave.createFromDataFrame(self.__value, self)
        # => not a valide type => couldn't give a value
        self.__autoSaveClasse_given = False # because setted to True by default, but exception happend
        raise TypeError(f"unintended type:{type(self.__value)}")

    def setValue(self, newValue:"_VT_ObjM", newType:"Type|bool"=True,
                 forceNoSave:bool=False, _FromAutoSaveCall=False)->None:
        if (_FromAutoSaveCall is False) and (self.__autoSaveClasse_given is True):
            raise RuntimeError("while in greedy mode, you can't set the value while the value has been given and still alive")
        if _FromAutoSaveCall is True:
            self.__autoSaveClasse_given = False
        
        self.__value = newValue
        if newType is True:
            self.__valueType = type(newValue)
        elif newType is not False: self.__valueType = newType
        # else => keep the current type
        self.__isSaved = False
        if (forceNoSave is False) and (self.__greedySaves is True):
            self.save(_FromAutoSaveCall=_FromAutoSaveCall)

    @property
    def typeValue(self)->Type[_Storable]:
        return self.__valueType

    def isSaved(self)->bool:
        return self.__isSaved

    @property
    def isCompressedAtSave(self)->bool:
        return self.__compress

    def setSaveCompressed(self, newValue:bool)->None:
        self.__compress = newValue

    def setGreedySave(self, newValue:bool)->None:
        self.__greedySaves = newValue
        if self.__greedySaves is False:
            self.__autoSaveClasse_given = False


    def _genFilePath(self, saveDirectory:str)->str:
        """determine the filename and assemble with the directory\\
        the generated filename should remain the same during the run"""
        fileName:str = self._Filename_prefix + str(Session.get_session().get_unique_id())
        return correctDirPath(saveDirectory) + fileName

    def unLoad_noSave(self)->None:
        """release the object in memory, without saving it (but will get marked as saved)\\
        (it need to be saved once, in or it will crash when loading)"""
        self.__value = None
        self.__isSaved = True

    def save(self, _force:bool=False, _FromAutoSaveCall:bool=False)->None:
        """save the object (do nothing if alredy saved)\\
        even if the object hasnt been modified it will write and not only free the object"""
        if (_FromAutoSaveCall is True) and (self.__greedySaves is False):
            return # dont save because not in greedy mode anymore
        
        if (self.__isSaved is True) and (_force is False):
            return # alredy saved, nothing to do
        
        if isinstance(self.__value, list):
            # transforme to numpy array and save
            if _PRINT_infos: print("saving list")
            with open(self.__filePath, mode="w+b") as savefile:
                if self.__compress is True:
                    numpy.savez_compressed(
                        savefile,
                        **{self.__ConvNpyArrSaveName:self.__value}
                    )
                else:
                    numpy.savez(
                        savefile,
                        **{self.__ConvNpyArrSaveName:self.__value}
                    )

        elif isinstance(self.__value, dict):
            if _PRINT_infos: print("saving dict of array")
            # the names of the arrays are saves inside
            with open(self.__filePath, mode="w+b") as savefile:
                if self.__compress is True:
                    numpy.savez_compressed(savefile, **self.__value)
                else: numpy.savez(savefile, **self.__value)

        elif isinstance(self.__value, numpy.ndarray):
            if _PRINT_infos: print("saving ndarray")
            with open(self.__filePath, mode="w+b") as savefile:
                if self.__compress is True:
                    numpy.savez_compressed(
                        savefile,
                        **{self.__ConvNpyArrSaveName:self.__value}
                    )
                else: numpy.savez(
                    savefile,
                    **{self.__ConvNpyArrSaveName:self.__value}
                )

        elif isinstance(self.__value, pandas.DataFrame):
            if _PRINT_infos: print("saving DataFrame")
            self.__value.to_parquet(
                self.__filePath,
                engine="fastparquet",
                compression="gzip" if self.__compress else None
            )

        elif self.__value is None:
            if _PRINT_infos: print("saving nothing")
            # nothing to save
            return # so keep __isSaved unchanged

        else: raise TypeError(f"trying to save unsupported type: {type(self.__value)}")

        self.__value = None
        self.__isSaved = True


    def load(self, _force:bool=False)->None:
        """load the object (no nothing if alredy loaded)"""
        if (self.__isSaved is False) and (_force is False):
            return # alredy loaded, nothing to do
            
        with open(self.__filePath, mode="r+b") as savfile:
            if self.__valueType == list:
                if _PRINT_infos: print("loading list")
                loadedValue:"Mapping[str, numpy.ndarray]" = \
                    numpy.load(savfile)
                self.setValue(
                    loadedValue[self.__ConvNpyArrSaveName].tolist(), # type:ignore guranted by the self.__valueType == list
                    forceNoSave=True,
                )

            elif self.__valueType == dict:
                if _PRINT_infos: print("loading dict of array")
                loadedValue:"Mapping[str, numpy.ndarray]" = \
                    numpy.load(savfile)
                self.setValue(
                    dict(loadedValue.items()), # type:ignore guranted by the self.__valueType == dict
                    forceNoSave=True,
                )

            elif self.__valueType == numpy.ndarray:
                if _PRINT_infos: print("loading array")
                loadedValue:"Mapping[str, numpy.ndarray]" = \
                    numpy.load(savfile)
                self.setValue(
                    loadedValue[self.__ConvNpyArrSaveName], # type:ignore guranted by the self.__valueType == numpy.ndarray
                    forceNoSave=True,
                )

            elif self.__valueType == pandas.DataFrame:
                if _PRINT_infos: print("loading DataFrame")
                self.setValue(pandas.read_parquet( # type:ignore guranted by the self.__valueType == pandas.DataFrame
                    savfile,
                    engine="fastparquet",
                ), forceNoSave=True)

            elif self.__valueType == NoneType:
                if _PRINT_infos: print("loading nothing")
                # nothing to load
                return # so keep __isSaved unchanged

            else: raise TypeError(f"trying to load unsupported type: {self.__valueType}")

        # alredy changed by setValue buy more clear
        self.__isSaved = False


    def clean(self, _force:bool=False)->bool:
        """try to clean the file that might have been generated\\
        dont clean anithing if the object isn't loaded, unless forced to clean\\
        return true if it deleted a file, false otherwise"""
        if (self.__isSaved is False) or (_force is True):
            if os.path.exists(self.__filePath):
                if _PRINT_infos: print("file cleaned")
                os.remove(self.__filePath)
                return True
        return False

    def __str__(self)->str:
        if self.__isSaved:
            return f"SaveObjModule({self.__valueType}, SAVED)"
        return f"SaveObjModule({self.__valueType}, LOADED)"

    def __del__(self)->None:
        """called when no ref to it"""
        self.clean(_force=True)
        Session.get_session().untrack_object(self.__filePath)



class SaveDictModule(Mapping, Generic[_KT_DictM, _VT_DictM]):
    def __init__(self,
            dictionary:"_DictStorable",
            greedySaves:bool=False,
            saveCompressed:bool=False,
            saveDirectory:"str|None"=None,
            saveDictOfArrayIndividualy:bool=False)->None:
        if saveDirectory is None: saveDirectory = SAVEMODULE_SESSION_DIRECTORY
        self.__greedySaves:bool = greedySaves
        self.__saveDirectory:str = saveDirectory
        self.__saveCompressed:bool = saveCompressed
        self.__saveDictOfArrayIndividualy:bool = saveDictOfArrayIndividualy

        self.__dictionary:"dict[_KT_DictM, SaveObjModule|SaveDictModule]" = {}
        for (key, value) in dictionary.items():
            self.addItem(key, value)

    def getRawSaveModuleItem(self, key:_KT_DictM)->"SaveDictModule|SaveObjModule":
            # the type error is normal, the conditions of the error cant happend due to the isnatance check
        return self.__dictionary[key]

    def getRawSaveModuleObject(self, key:_KT_DictM)->"SaveObjModule[_VT_DictM]":
            # the type error is normal, the conditions of the error cant happend due to the isnatance check
        saveModule_item:"SaveObjModule|SaveDictModule" = self.__dictionary[key]
        if isinstance(saveModule_item, SaveDictModule):
            raise TypeError("SaveDictModule is not an SaveObjModule")
        else: return saveModule_item

    def setSaveCompressed(self, newValue:bool)->None:
        """also set reccursively"""
        self.__saveCompressed = newValue
        for item in self.__dictionary.values():
            item.setSaveCompressed(newValue)

    def setGreedySave(self, newValue:bool)->None:
        self.__saveCompressed = newValue
        for item in self.__dictionary.values():
            item.setGreedySave(newValue)

    def __getitem__(self, key:_KT_DictM)->_VT_DictM:
        item:"SaveObjModule|SaveDictModule" = self.__dictionary[key]
        if isinstance(item, SaveObjModule):
            return item.value
        else: return item # cant realy ignore typeError

    def __setitem__(self, key:_KT_DictM, newValue:_VT_DictM)->None:
        item:"SaveObjModule|SaveDictModule|None" = self.__dictionary.get(key, None)
        if item is None: # not an existing key
            if isinstance(newValue, SaveDictModule):
                # set the parameters of SaveDictModule accordingly this one and add it
                newValue.__greedySaves = self.__greedySaves
                newValue.__saveDirectory = self.__saveDirectory
                newValue.__saveCompressed = self.__saveCompressed
                self.__dictionary[key] = newValue
                if self.__greedySaves is True:
                    newValue.saveAll()

            else: # => _Storable
                self.addItem(key, newValue)

        # key already exist
        elif isinstance(item, SaveObjModule):
            if isinstance(newValue, _STORABLE_CHECK_ISINSTANCE):
                item.setValue(newValue)
                if self.__greedySaves is True:
                    item.save()

            else: raise TypeError(f"trying to set a new value of a type that isn't supported when setting a SaveObjModule: {type(newValue)}")


        elif isinstance(item, SaveDictModule) and isinstance(newValue, SaveDictModule):
            # set the parameters of SaveDictModule accordingly this one and add it
            item.clear()
            newValue.__greedySaves = self.__greedySaves
            newValue.__saveDirectory = self.__saveDirectory
            newValue.__saveCompressed = self.__saveCompressed
            self.__dictionary[key] = newValue

        else: raise TypeError(f"can't set a {type(newValue)} over a SaveDictModule")


    def addItem(self, newKey:_KT_DictM, value:"_Storable|_DictStorable")->None:
        """try to add an item to the dict, will raise an error if key alredy exist"""
        if newKey in self.__dictionary:
            raise KeyError(f"trying to add a key that alredy exist:{newKey}")

        kwargs:"dict[str, Any]" = {
            "greedySaves":self.__greedySaves,
            "saveDirectory":self.__saveDirectory,
            "saveCompressed":self.__saveCompressed
        }
        if isinstance(value, dict):
            if (self.__saveDictOfArrayIndividualy is True) \
                 or (not all(isinstance(item, numpy.ndarray) for item in value.values())):
                # test if is not: dict[Any, ndarray]
                self.__dictionary[newKey] = SaveDictModule(value, **kwargs)

            else: # => __saveDictOfArrayIndividualy=False and value is dict[Any, ndarray]
                self.__dictionary[newKey] = SaveObjModule(
                    value, # type:ignore the type has been verified to be supported
                    **kwargs
                )

        else:
            self.__dictionary[newKey] = SaveObjModule(
                value, # cant realy ignore typeError
                **kwargs
            )

    def __delitem__(self, key:_KT_DictM)->None:
        self.__dictionary.__delitem__(key)
    delItem = __delitem__


    def loadAll(self, _force:bool=False)->None:
        # inclure une methode de async les operations io ?
        for item in self.__dictionary.values():
            if isinstance(item, SaveObjModule):
                item.load(_force=_force)
            else: item.loadAll(_force=_force)

    def saveAll(self, _force:bool=False)->None:
        for item in self.__dictionary.values():
            if isinstance(item, SaveObjModule):
                item.save(_force=_force)
            else: item.saveAll(_force=_force)

    def save(self, key:_KT_DictM)->None:
        """save a single item"""
        item:"SaveObjModule|SaveDictModule" = self.__dictionary[key]
        if isinstance(item, SaveObjModule):
            item.save()
        else: # => SaveDictModule
            item.saveAll()

    def __iter__(self)->"Generator[_KT_DictM, None, None]":
        yield from self.__dictionary

    def __len__(self)->int:
        return len(self.__dictionary)

    def keys(self)->"list[_KT_DictM]":
        return list(self.__dictionary.keys())

    def items(self)->"Generator[tuple[_KT_DictM, _VT_DictM], None, None]":
        for key in self.__dictionary.keys():
            yield (key, self[key])

    def values(self)->"Generator[_VT_DictM, None, None]":
        for key in self.__dictionary.keys():
            yield self[key]

    def clear(self)->None:
        # clear all the contents and files
        for item in self.__dictionary.values():
            if isinstance(item, SaveDictModule):
                item.clear()
            # else => item:SaveObjModule: no clean, it can still be hold somewere else
        self.__dictionary.clear()

    def unLoad_noSave(self, key:_KT_DictM)->None:
        self.getRawSaveModuleObject(key).unLoad_noSave()

    def to_dict(self)->"dict[_KT_DictM, _VT_DictM]":
        """create a dict for what it contain (not recursive)"""
        resDict:"dict[_KT_DictM, SaveDictModule|_Storable]" = {}
        for (key, item) in self.__dictionary.items():
            if isinstance(item, SaveObjModule):
                resDict[key] = item.value
            else: resDict[key] = item
        return resDict # dont resolve but should be ok

    def __str__(self)->str:
        return \
            "{" + ", ".join(
                f"{key}: {item}"
                for key, item in self.__dictionary.items()
            ) + "}"

    def __del__(self):
        self.clear()
        if _PRINT_infos: print("cleaned dictSaveModule")

