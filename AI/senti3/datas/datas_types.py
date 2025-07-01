"""Datas_dataset -> Datas_series_raw -> Datas_series_regularized -> Datas_array -> Datas_InputOutput"""
# TODO: add a Datas_array.toInputOutput -> Datas_InputOutput
if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

import attrs
import numpy
import pandas as pd
from datetime import datetime, timedelta
import numba
import joblib # pre-import the lib for holo.ramDiskSave, not used here
from math import ceil

from modules.numbaJit import fastJitter, void, floating, JitType, floatArray, int32
from calculationLib import (
    _SeriesGeneratorConfig, generate_calculationSeries, _SerieName)
import AI
from save_formats import (
    AsJson_DatasetInfos, 
    AsJson_Datetime, datetimeFromJson, datetimeToJson, )

from holo import getDuplicated
from holo.types_ext import _Serie_Float, _3dArray_Float
from holo.__typing import (
    NamedTuple, LiteralString, TypeVar,
    Generic, assertIsinstance, DefaultDict,
    Generator, Literal, )
from holo.ramDiskSave import Session, SaveArgs, ObjectSaver



float3D = JitType("float", nbDims=3)

@fastJitter(void(float3D, floatArray, int32), parallel=True)
def _copy_serie_into_Datas(
        datas:_3dArray_Float, serieToCopy:_Serie_Float,
        indexFeature:int)->None:
    """copy the datas for a feature like: 
        nb periodes = 3
        initial datas:
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        final datas:
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
            [2, 3, 4, 5, 6, 7, 8, 9, 10]
    the shape of the datas: (nbFeatures, nbPeriodes, nbSamples)\n
    if the shape of your datas don't match use sawpaxis before :)"""
    nbPeriodes: int = datas.shape[1]
    nbSamples: int = datas.shape[2]
    for indexSample in numba.prange(nbSamples):
        for indexPeriode in range(nbPeriodes):
            datas[indexFeature, indexPeriode, indexSample] = \
                serieToCopy[indexSample + indexPeriode]
    return None

@fastJitter(void(floatArray, floating), parallel=True)
def add_noise_inplace_serie(array:"_Serie_Float", coef:float)->None:
    for i in numba.prange(array.shape[0]):
        noiseValue: float = coef * (numpy.random.random()*2.0 -1.0)
        array[i] = numpy.maximum(numpy.minimum(array[i] + noiseValue, 1.0), 0.0)


def assertSeriesSubSet(datas_series: "set[_SerieName]", 
                       subSet_series:"set[_SerieName]")->None:
    seriesNotInDatas = subSet_series.difference(datas_series)
    if len(seriesNotInDatas) != 0:
        raise KeyError("the following series are in `subSet_series`"
                       +f" but not in `datas_series`: {seriesNotInDatas}")

def assertNoDuplicatedSeries(names:"list[_SerieName]")->None:
    duplicatedSeries: "set[_SerieName]" = getDuplicated(names)
    if len(duplicatedSeries) != 0:
        raise KeyError(f"the following series are duplicated: {duplicatedSeries}")

class Periode(NamedTuple):
    start: datetime
    """the first time in the periode"""
    end: datetime
    """the last time in the periode"""
    
    def __contains__(self, key:"Periode|datetime")->bool:
        if isinstance(key, Periode):
            return (key.start in self) and (key.end in self)
        elif isinstance(key, datetime):
            return (self.start <= key <= self.end)
        else: raise TypeError(f"unsupported key type: {type(key)}")

    def intersect(self, _other:"Periode")->bool:
        return (self.start <= _other.end) and (_other.start <= self.end)
    
class DatasetInfos(NamedTuple):
    key: str
    length: int
    startDate: datetime
    endDate: datetime
    
    def toJson(self)->"AsJson_DatasetInfos":
        return AsJson_DatasetInfos(
            cls=self.__class__.__name__,
            key=self.key, length=self.length,
            startDate=datetimeToJson(self.startDate),
            endDate=datetimeToJson(self.endDate))
    @classmethod
    def fromJson(cls, datas:"AsJson_DatasetInfos")->"DatasetInfos":
        assert datas["cls"] == cls.__name__
        return DatasetInfos.__new__(
            cls=cls, key=datas["key"], length=datas["length"], 
            startDate=datetimeFromJson(datas["startDate"]), 
            endDate=datetimeFromJson(datas["endDate"]))




_T_DfKey = TypeVar("_T_DfKey", LiteralString, str)

class Datas_dataset(Generic[_T_DfKey]):
    """holder for the dataframe with """
    __slots__ = ("key", "dataframe", )
    
    def __init__(self, key:"_T_DfKey", dataframe:"pd.DataFrame") -> None:
        self.key: "_T_DfKey" = key
        self.dataframe: "pd.DataFrame" = dataframe
    
    @property
    def startDate(self)->datetime:
        return assertIsinstance(
            pd.Timestamp, self.dataframe.index[0]).to_pydatetime()

    @property
    def endDate(self)->datetime:
        return assertIsinstance(
            pd.Timestamp, self.dataframe.index[-1]).to_pydatetime()
    
    @property
    def length(self)->int:
        return len(self.dataframe.index)
    
    def toDatas_series_raw(
            self, seriesGeneratorConfig:"_SeriesGeneratorConfig",
            dtype:"type[numpy.floating]")->"Datas_series_raw[_T_DfKey]":
        series: "dict[_SerieName, _Serie_Float]" = \
            generate_calculationSeries(self.dataframe, seriesGeneratorConfig, seriesDtype=dtype)
        return Datas_series_raw(
            fromKey=self.key, series=series, startDate=self.startDate,
            endDate=self.endDate, dtype=dtype, valuesRange=None)

    def getPeriode(self)->"Periode":
        return Periode(self.startDate, self.endDate)

    def getInfos(self)->"DatasetInfos":
        return DatasetInfos(
            key=self.key, length=self.length,
            startDate=self.startDate, endDate=self.endDate)

_T_Datas_series = TypeVar("_T_Datas_series", bound="_Datas_series")


class _Datas_series(Generic[_T_DfKey]):
    """base class of datas holder for the series regularized and not regularized"""
    __slots__ = ("fromKey", "series", "startDate", 
                 "endDate", "dtype", "valuesRange", )
    
    def __init__(
            self, fromKey:"_T_DfKey", series:"dict[_SerieName, _Serie_Float]",
            startDate:datetime, endDate:datetime, dtype:"type[numpy.floating]",
            valuesRange:"AI.ValuesRange|None") -> None:
        self.fromKey:"_T_DfKey" = fromKey
        """the key of the dataset used"""
        self.series:"dict[_SerieName, _Serie_Float]" = series
        self.startDate:datetime = startDate
        """the date of the first value of the series"""
        self.endDate:datetime = endDate
        """the date of the last value of the series"""
        self.dtype: "type[numpy.floating]" = dtype
        """the type of all the series (they will be converted to this type if needed)"""
        self.valuesRange: "AI.ValuesRange|None" = valuesRange
        self.assertLengths()
    
    @property
    def seriesNames(self)->"set[_SerieName]":
        return set(self.series.keys())
    
    def addSerie(self, name:"_SerieName", serie:_Serie_Float)->None:
        if len(self.series) == 0: 
            # => no series for now (no further checks)
            self.series[name] = serie
            return 
        # => it has some series
        if name in self.series: 
            # => adding a serie twice
            raise KeyError(f"trying to add the serie: {repr(name)} but it alredy exist")
        # => adding a new serie
        if len(serie) != self.length:
            raise ValueError(f"invalide length: {len(serie)} != {self.length}")
        # add the serie and fix its dtype if needed
        self.series[name] = (serie.astype(self.dtype) if serie.dtype != self.dtype else serie)
    
    def assertLengths(self)->None:
        """assert they all are the same length"""
        seriesLengths: "dict[int, set[_SerieName]]" = \
            DefaultDict(lambda: set())
        for name, serie in self.series.items():
            seriesLengths[len(serie)].add(name)
        if len(seriesLengths) > 1:
            # => different lengths
            raise ValueError("there are series with diferant lengths "
                             f"(lengths -> series): {seriesLengths}")
    
    def filterSeries(self:"_T_Datas_series", selectedSeries:"set[_SerieName]")->"_T_Datas_series":
        """create a new datas series object with a shallow copy of the `selectedSeries`\n
        self needs to have all the series in `selectedSeries`"""
        # series selection
        assertSeriesSubSet(self.seriesNames, selectedSeries)
        # create the new object
        return type(self)(
            fromKey=self.fromKey,
            series={name: self.series[name]
                    for name in selectedSeries},
            startDate=self.startDate, endDate=self.endDate,
            dtype=self.dtype, valuesRange=self.valuesRange)
    
    def slice(self:"_T_Datas_series", *, startIndex:"int", endIndex:"int|None")->"_T_Datas_series":
        """return a new datas series object with the series cuted like python slice:
         - `startIndex` is the index of the first data to be kept
         - `endIndex` is the index AFTER the last to be kept \
             (it can be negative or None like slicing end index)"""
        # indexes computation
        currentLength: int = self.length
        if endIndex is None: endIndex = currentLength
        elif endIndex < 0: endIndex = currentLength + endIndex
        assert startIndex < endIndex, \
            IndexError(f"in valide start/end indexes: startIndex={startIndex} >= endIndex={endIndex}")
        # compute the new dates
        periodeDuration: timedelta = self.periodeDuration
        newStartDate: datetime = self.startDate + periodeDuration * startIndex
        newEndDate: datetime = self.endDate - periodeDuration * (currentLength - endIndex)
        # create the new structure
        return type(self)(
            fromKey=self.fromKey,
            series={name: serie[startIndex: endIndex]
                    for name, serie in self.series.items()},
            startDate=newStartDate, endDate=newEndDate,
            dtype=self.dtype, valuesRange=self.valuesRange)
    
    @property
    def length(self)->int:
        """get the length of the series (in number of periodes)"""
        return len(next(iter(self.series.values())))
    
    @property
    def periodeDuration(self)->timedelta:
        return (self.endDate - self.startDate) / (self.length -1)
    
    def shallowCopy(self:"_T_Datas_series")->"_T_Datas_series":
        return type(self)(
            fromKey=self.fromKey, series=self.series.copy(),
            startDate=self.startDate, endDate=self.endDate,
            dtype=self.dtype, valuesRange=self.valuesRange)

    def merge(self:"_T_Datas_series", __other:"_T_Datas_series")->None:
        if self.fromKey != __other.fromKey: 
            raise ValueError(f"invalide key: {self.fromKey} != {__other.fromKey}")
        if self.startDate != __other.startDate: 
            raise ValueError(f"invalide startDate: {self.startDate} != {__other.startDate}")
        if self.endDate != __other.endDate: 
            raise ValueError(f"invalide endDate: {self.endDate} != {__other.endDate}")
        seriesInBoth: "set[_SerieName]" = \
            self.seriesNames.intersection(__other.series.keys())
        if len(seriesInBoth) != 0:
            raise KeyError(f"some series are in both datas: {seriesInBoth}")
        # => other can be merged in self
        for serieName, serieArray in __other.series.items():
            self.series[serieName] = serieArray
    
    def getPeriode(self)->"Periode":
        return Periode(self.startDate, self.endDate)
    
    def getInfos(self)->"DatasetInfos":
        return DatasetInfos(
            key=self.fromKey, length=self.length,
            startDate=self.startDate, endDate=self.endDate)
    
    def __str__(self)->str:
        return (f"{self.__class__.__name__}(fromKey={self.fromKey}, "
                f"length={self.length}, startDate={self.startDate}, "
                f"endDate={self.endDate}, periodeDuration={self.periodeDuration}, "
                f"seriesNames:{self.series.keys()})")


class Datas_series_raw(_Datas_series[_T_DfKey]):
    """datas holder for the series that arn't regularized"""
    # don't use custom init (or check compat with subclass)
    valuesRange: None
    
    def regularize(self, regularizeConfig:"AI.RegularizeConfig"
                   )->"Datas_series_regularized[_T_DfKey]":
        regularizedSeries = AI.regularize_datas(
            seriesToRegularize=self.series,
            seriesSelection=None, # => regularize all
            rescale=regularizeConfig.rescale,
            preferReRangePlus=regularizeConfig.preferReRangePlus,
            valuesRange=regularizeConfig.valuesRange)
        return Datas_series_regularized(
            fromKey=self.fromKey, series=regularizedSeries,
            startDate=self.startDate, endDate=self.endDate,
            dtype=self.dtype, valuesRange=regularizeConfig.valuesRange)



class Datas_series_regularized(_Datas_series[_T_DfKey]):
    """datas holder for the series that are regularized"""
    # don't use custom init (or check compat with subclass)
    valuesRange: "AI.ValuesRange"
    
    def toData_array(self, nbPeriodes:int, seriesOrder:"list[_SerieName]", 
                     dtype:"type[numpy.floating]")->"Datas_array[_T_DfKey]":
        """transform the datas to an array form\n
        the resulting datas will have:
         - trueStartDate = self.startDate
         - endDate = self.endDate"""
        assert set(seriesOrder) == self.seriesNames
        nbFeatures: int = len(seriesOrder)
        nbSamples: int = self.length - nbPeriodes + 1
        assert nbSamples > 0, \
            ValueError(f"the series arn't long enought (length={self.length}) "
                       f"to support nbPeriodes={nbPeriodes}")
        datasSwapedAxis: "_3dArray_Float" = \
            numpy.zeros((nbFeatures, nbPeriodes, nbSamples), dtype=dtype)
        newStartDate: datetime = self.startDate + self.periodeDuration * (nbPeriodes -1)
        
        # copy the datas for each features and periodes
        for featureIndex, serieName in enumerate(seriesOrder):
            serieToCopy: "_Serie_Float" = self.series[serieName]
            if serieToCopy.dtype != dtype:
                serieToCopy = serieToCopy.astype(dtype)
            _copy_serie_into_Datas(
                datas=datasSwapedAxis, indexFeature=featureIndex,
                serieToCopy=serieToCopy)
        return Datas_array(
            fromKey=self.fromKey, datas=datasSwapedAxis.swapaxes(0, 2),
            seriesNames=seriesOrder, trueStartDate=self.startDate, startDate=newStartDate,
            endDate=self.endDate, valuesRange=self.valuesRange)

class Datas_array(Generic[_T_DfKey]):
    """a datas holder in an array, ready for the ai, with the series that are regularized"""
    __slots__ = ("fromKey", "datas", "seriesNames", "startDate",
                 "trueStartDate", "endDate", "valuesRange", )
    
    def __init__(self, fromKey:"_T_DfKey", datas:"_3dArray_Float", 
                 seriesNames:"list[_SerieName]", trueStartDate:datetime, startDate:datetime,
                 endDate:datetime, valuesRange: "AI.ValuesRange")->None:
        """datas shape: (nbSamples, nbPeriodes, nbFeatures)"""
        self.fromKey: "_T_DfKey" = fromKey
        self.valuesRange: "AI.ValuesRange" = valuesRange
        self.datas: "_3dArray_Float" = datas
        """the shape is: (nbSamples, nbPeriodes, nbFeatures)"""
        assertNoDuplicatedSeries(seriesNames)
        self.seriesNames: "list[_SerieName]" = seriesNames
        self.trueStartDate: datetime = trueStartDate
        """the date of the first value of the serie of first periode"""
        
        self.startDate: datetime = startDate
        """the date of the first value of the serie of last periode"""
        self.endDate: datetime = endDate
        """the date of the last value of the serie of last periode"""
        assert len(self.datas.shape) == 3
        assert len(seriesNames) == self.nbFeatures
    
    @property
    def shape(self)->"tuple[int, int, int]":
        """(nbSamples, nbPeriodes, nbFeatures)"""
        shape = self.datas.shape
        assert len(shape) == 3
        return shape
    
    @property
    def nbSamples(self)->int:
        return self.datas.shape[0]
    
    @property
    def nbPeriodes(self)->int:
        return self.datas.shape[1]
    
    @property
    def nbFeatures(self)->int:
        return self.datas.shape[2]
    
    @property
    def dtype(self)->"type[numpy.floating]":
        return self.datas.dtype.type
    
    @property
    def periodeDuration(self)->timedelta:
        return (self.endDate - self.startDate) / (self.nbSamples - 1)
    
    def getSeriesIndexes(self, seriesNames:"set[_SerieName]")->"dict[_SerieName, int]":
        return {serieName: index
                for index, serieName in enumerate(self.seriesNames)
                    if serieName in seriesNames}

    def __str__(self)->str:
        return (f"{self.__class__.__name__}(fromKey={self.fromKey}, "
                f"nbSamples={self.nbSamples}, nbPeriodes={self.nbPeriodes}, "
                f"nbFeatures={self.nbFeatures}, trueStartDate={self.trueStartDate}, "
                f"startDate={self.startDate}, endDate={self.endDate}, "
                f"periodeDuration={self.periodeDuration}, seriesNames={self.seriesNames})")

    def getInfos(self)->"_DatasArrayInfos":
        return _DatasArrayInfos(
            fromKey=self.fromKey,
            trueStartDate=self.trueStartDate,
            startDate=self.startDate,
            endDate=self.endDate,
            nbSamples=self.nbSamples,
            nbFeatures=self.nbFeatures,
            nbPeriodes=self.nbPeriodes)
    
    def getPeriode(self)->"Periode":
        return Periode(self.trueStartDate, self.endDate)


class _DatasArrayInfos(NamedTuple):
    """a class that hold the infos of the datas contained in a datas array"""
    fromKey: str
    trueStartDate: datetime
    startDate: datetime
    endDate: datetime
    nbSamples: int
    nbFeatures: int
    nbPeriodes: int
    
    def getPeriode(self)->"Periode":
        return Periode(self.trueStartDate, self.endDate)
    
class _DatasInfos_InputOutput(NamedTuple):
    inputInfos: "_DatasArrayInfos"
    outputInfos: "_DatasArrayInfos"

class Datas_InputOutput(Generic[_T_DfKey]):
    """regroup the input and output datas for the ai\n
    ensure they have the same nbumber of samples"""
    __slots__ = ("inputs_datas", "outputs_datas", )
    
    def __init__(self, inputs_datas:"Datas_array[_T_DfKey]", 
                 outputs_datas:"Datas_array[_T_DfKey]")->None:
        if inputs_datas.nbSamples != outputs_datas.nbSamples:
            raise IndexError("the given datas don't have the same amount of samples"
                             f"inputs_datas has: {inputs_datas.nbSamples} "
                             f"and outputs_datas has: {outputs_datas.nbSamples}")
        self.inputs_datas: "Datas_array[_T_DfKey]" = inputs_datas
        self.outputs_datas: "Datas_array[_T_DfKey]" = outputs_datas
    
    @property
    def nbSamples(self)->int:
        return self.inputs_datas.nbSamples

    def getInfos(self)->"_DatasInfos_InputOutput":
        return _DatasInfos_InputOutput(
            inputInfos=self.inputs_datas.getInfos(),
            outputInfos=self.outputs_datas.getInfos())


class _DatasArray_InputOutput(NamedTuple):
    """regroup the input and output datas for the ai \
        in there aray shape (ready to feed for fitting)"""
    inputs_array: "_3dArray_Float"
    outputs_array: "_3dArray_Float"
    
    @classmethod
    def fromInputOutput(cls, datas:"Datas_InputOutput")->"_DatasArray_InputOutput":
        return _DatasArray_InputOutput(
            inputs_array=datas.inputs_datas.datas, 
            outputs_array=datas.outputs_datas.datas)
    
    @property
    def nbSamples(self)->int:
        return self.inputs_array.shape[0]




class Datas_training_Generator():
    __slots__ = ("__allTrainDatas", "__allValidationDatas", 
                 "__nbEpochesToRepeate", "__batchSize", 
                 "__nbTrainingSteps", "__nbValidationSteps", 
                 "__trainingDatasInfos", "__validationDatasInfos", )
    
    def __init__(self, nbEpoches:int, batchSize:int)->None:
        """create an empty datas generator to handle \
            the training and validation datas of the ai\n
        `nbEpoches` is the number of epoches to do with the given datas \
            (use the same datas at each epoche)\n
        `randomizeDfsOrder` whether to randomize the order of training dfs, \
            but the batches of the df will be in the same order"""
        assert (nbEpoches >= 1), ValueError(f"nbEpoches: {nbEpoches} must be >= 1")
        self.__allTrainDatas: "list[ObjectSaver[_DatasArray_InputOutput]]" = []
        self.__allValidationDatas: "list[ObjectSaver[_DatasArray_InputOutput]]" = []
        self.__nbEpochesToRepeate: int = nbEpoches
        self.__batchSize: int = batchSize
        self.__nbTrainingSteps: int = 0
        self.__nbValidationSteps: int = 0
        self.__trainingDatasInfos: "list[_DatasInfos_InputOutput]" = []
        self.__validationDatasInfos: "list[_DatasInfos_InputOutput]" = []
    
    @property
    def nbTrainingSteps(self)->int:
        return self.__nbTrainingSteps
    
    @property
    def nbValidationSteps(self)->int:
        return self.__nbValidationSteps
    
    @property
    def nbEpoches(self)->int:
        return self.__nbEpochesToRepeate
    
    def __addDatas(
            self, datas:"_DatasArray_InputOutput", infos:"_DatasInfos_InputOutput",
            kind:"Literal['training', 'validation']")->None:
        nbSamples: int = datas.nbSamples
        saveObj = ObjectSaver(datas)
        saveObj.save()
        if kind == "training":
            self.__allTrainDatas.append(saveObj)
            self.__nbTrainingSteps += self.__getNbSteps(nbSamples)
            self.__trainingDatasInfos.append(infos)
        elif kind == "validation":
            self.__allValidationDatas.append(saveObj)
            self.__nbValidationSteps += self.__getNbSteps(nbSamples)
            self.__validationDatasInfos.append(infos)
        else: raise ValueError(f"invalide king: {repr(kind)}")
    
    def addTrainingDatas(self, datas:"Datas_InputOutput")->None:
        self.__addDatas(
            datas=_DatasArray_InputOutput.fromInputOutput(datas),
            infos=datas.getInfos(), kind="training")
    
    def addValidationDatas(self, datas:"Datas_InputOutput")->None:
        self.__addDatas(
            datas=_DatasArray_InputOutput.fromInputOutput(datas),
            infos=datas.getInfos(), kind="validation")
    
    def getTrainingGenerator(self)->"Generator[_DatasArray_InputOutput, None, None]":
        for epoche in range(1, self.__nbEpochesToRepeate+1):
            for datasSaveObject in self.__getShuffuled(self.__allTrainDatas):
                try: yield from self.__batchs_generator(
                    datasSaveObject.value)  # will load the datas
                finally: datasSaveObject.unLoad_noSave()
    
    def getValidationGenerator(self)->"Generator[_DatasArray_InputOutput, None, None]":
        for epoche in range(1, self.__nbEpochesToRepeate+1):
            for datasSaveObject in self.__allValidationDatas:
                try: yield from self.__batchs_generator(datasSaveObject.value)
                finally: datasSaveObject.unLoad_noSave()
    
    def getTrainingInfos(self)->"list[_DatasInfos_InputOutput]":
        return self.__trainingDatasInfos.copy()
    
    def getValidationInfos(self)->"list[_DatasInfos_InputOutput]":
        return self.__validationDatasInfos.copy()
    
    def __getNbSteps(self, nbSamples:int)->int:
        """return the number of steps (ie the number of batches) \
            to do for this datas"""
        return ceil(nbSamples / self.__batchSize)
    
    def __batchs_generator(
            self, datas:"_DatasArray_InputOutput",
            )->"Generator[_DatasArray_InputOutput, None, None]":
        # calculate the number of batches for training data
        nbBatchs: int = self.__getNbSteps(datas.nbSamples)

        # yield each batch of training data
        for batch_index in range(nbBatchs):
            start_idx = batch_index * self.__batchSize
            end_idx = (batch_index + 1) * self.__batchSize
            inputsArray: "_3dArray_Float" = datas.inputs_array[start_idx: end_idx]
            outputsArray: "_3dArray_Float" = datas.outputs_array[start_idx: end_idx]
            yield _DatasArray_InputOutput(inputsArray, outputsArray)
    
    def __getShuffuled(
            self, datasList:"list[ObjectSaver[_DatasArray_InputOutput]]",
            )->"list[ObjectSaver[_DatasArray_InputOutput]]":
        """return a permutaion of size: `nbElements`"""
        permutation: "list[int]" = \
            list(numpy.random.permutation(len(datasList)))
        return [datasList[index] for index in permutation]
        
    def clear(self)->None:
        """remove all the datas it has and clean their files\n
        the equivalent is done when the object is deleted"""
        # clearing the lists will del the objects 
        #   and so clean the generated files
        self.__allTrainDatas.clear()
        self.__allValidationDatas.clear()
        self.__nbTrainingSteps: int = 0
        self.__nbValidationSteps: int = 0
    
    def getSubSample(
            self, keepTrainingProp:float, keepValidationProp:float)->"Datas_training_Generator":
        newGen = Datas_training_Generator(self.nbEpoches, batchSize=self.__batchSize)
        newNbTrainDatas: int = ceil(keepTrainingProp * len(self.__allTrainDatas))
        newNbValDatas: int = ceil(keepValidationProp * len(self.__allValidationDatas))
        # select the index of the datas to keep
        trainDatasIndexes = numpy.random.permutation(len(self.__allTrainDatas))[: newNbTrainDatas]
        valDatasIndexes = numpy.random.permutation(len(self.__allValidationDatas))[: newNbValDatas]
        # add the datas to keep to the new gen
        for trainIndex in trainDatasIndexes:
            newGen.__addDatas(
                datas=self.__allTrainDatas[trainIndex].value,
                infos=self.__trainingDatasInfos[trainIndex], kind="training")
            self.__allTrainDatas[trainIndex].unLoad_noSave()
        for valIndex in valDatasIndexes:
            newGen.__addDatas(
                datas=self.__allValidationDatas[valIndex].value,
                infos=self.__validationDatasInfos[valIndex], kind="validation")
            self.__allValidationDatas[valIndex].unLoad_noSave()
        return newGen
                