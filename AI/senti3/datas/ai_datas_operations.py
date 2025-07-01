if __name__ == "__main__":
    raise ImportError(f"the {__file__} must be imported from main dir")

import numba
import numpy

from .types_config import _SerieName

from modules.numbaJit import (
    fastJitter, JitType, integer, floating,
    floatArray, floatMatrix, void, )
from .datas_types import (
    Datas_series_regularized, Datas_array,
    _T_DfKey, assertSeriesSubSet, 
    add_noise_inplace_serie,
)

from holo.__typing import Literal
from holo.types_ext import _Serie_Float, _2dArray_Float, _Serie_Integer


_RegroupMethode = Literal["avg", "geom", "first"]
"""methodes to regroup a '_2dArray_Float' of a feature to a '_Serie_Float'"""

float3D = JitType("float", nbDims=3)


def add_noise(datas: "Datas_series_regularized[_T_DfKey]", *, inPlace:"bool",
              noiseConfigs:"dict[_SerieName, float|Literal[True]]",
              )->"Datas_series_regularized[_T_DfKey]":
    """applie some noise to the designated series:
     - float -> add a noise scaled with this coefficient
     - True -> replace the serie with a noise serie"""
    assertSeriesSubSet(set(datas.series.keys()), set(noiseConfigs.keys()))
    
    result_datas: "Datas_series_regularized[_T_DfKey]"
    if inPlace is True:
        result_datas = datas
    else: result_datas = datas.shallowCopy()
    
    for featureName, noiseScale in noiseConfigs.items():
        if noiseScale is True: # => fully random
            result_datas.series[featureName] = \
                numpy.random.random(result_datas.series[featureName].shape)
        else: add_noise_inplace_serie(result_datas.series[featureName], noiseScale)
    return result_datas


#TODO look to paralelize the loops

@fastJitter(floatArray(floatMatrix))
def matrixToArray_geom(mat:"_2dArray_Float")->"_Serie_Float":
    """compress the matrix of the feature (of shape: (nbSamples, nbPeriodes)) 
        to an array using geometric coefficients"""
    nbSamples: int = mat.shape[0]
    nbPeriodes: int = mat.shape[1]
    GEOM_COEF: float = 1 - 4 / nbPeriodes
    resultsSize: int = nbSamples + nbPeriodes -1
    result: "_Serie_Float" = numpy.zeros(resultsSize, dtype=mat.dtype)
    weights: "_Serie_Float" = numpy.zeros(resultsSize, dtype=mat.dtype)
    for indexPeriode in range(nbPeriodes):
        coeff: float = GEOM_COEF ** indexPeriode
        
        for indexSample in range(nbSamples):
            result_index:int = indexSample + indexPeriode
            result[result_index] += mat[indexSample, indexPeriode] * coeff
            weights[result_index] += coeff
    return result / weights

@fastJitter(floatArray(floatMatrix))
def matrixToArray_avg(mat:"_2dArray_Float")->"_Serie_Float":
    """compress the matrix of the feature (of shape: (nbSamples, nbPeriodes)) 
        to an array by computing the average of values at the same instant"""
    nbSamples: int = mat.shape[0]
    nbPeriodes: int = mat.shape[1]
    resultsSize: int = nbSamples + nbPeriodes -1
    result: "_Serie_Float" = numpy.zeros(resultsSize, dtype=mat.dtype)
    weights: "_Serie_Integer" = numpy.zeros(resultsSize, dtype=numpy.int32)
    for indexSample in range(nbSamples):
        for indexPeriode in range(nbPeriodes):
            result_index:int = indexSample + indexPeriode
            result[result_index] += mat[indexSample, indexPeriode]
            weights[result_index] += 1
    return result / weights

@fastJitter(floatArray(floatMatrix))
def matrixToArray_first(mat:"_2dArray_Float")->"_Serie_Float":
    """compress the matrix of the feature (of shape: (nbSamples, nbPeriodes)) 
        to an array by taking the first periode available"""
    nbSamples:int; nbPeriodes:int
    nbSamples, nbPeriodes = mat.shape
    resultsSize:int = nbSamples+nbPeriodes-1
    result:"_Serie_Float" = numpy.zeros(resultsSize, dtype=mat.dtype)
    for indexSample in range(nbSamples):
        result[indexSample] = mat[indexSample, 0]
    for indexPeriode in range(1, nbPeriodes):
        result[(nbSamples-1) + indexPeriode] = mat[(nbSamples-1), indexPeriode]
    return result


def compressMatrixToArray(
        datasMatrix:"_2dArray_Float", compressionMethode:"_RegroupMethode")->"_Serie_Float":
    if compressionMethode == "first":
        return matrixToArray_first(datasMatrix)
    elif compressionMethode == "avg":
        return matrixToArray_avg(datasMatrix)
    elif compressionMethode == "geom":
        return matrixToArray_geom(datasMatrix)
    else: raise ValueError(f"compressionMethode: {compressionMethode} isn't supported")


def compressDatasArray(
        datas:"Datas_array[_T_DfKey]", selectedSeries:"set[_SerieName]",
        *, compressionMode:"_RegroupMethode|dict[_SerieName, _RegroupMethode]",
        )->"Datas_series_regularized[_T_DfKey]":
    """compress the `datas` back to regularized series\n
    `selectedSeries` the series to compress (the only series that will be in the resulting datas) \n
    `compressionMode` witch methode to use in order to compress the datas:
         - "None" -> take the first periode value available
         - "avg" -> compute the average of each values of the same instant
         - "geom" -> compute a geometric ponderation of values of the same instant"""
    selectedSeriesIndexes: "dict[_SerieName, int]" = datas.getSeriesIndexes(selectedSeries)
    resultDatas = Datas_series_regularized(
        fromKey=datas.fromKey, series={}, dtype=datas.dtype, startDate=datas.trueStartDate,
        endDate=datas.endDate, valuesRange=datas.valuesRange)
    
    compMode: "_RegroupMethode"
    for featureName, featureIndex in selectedSeriesIndexes.items():
        featureMatrixArray: "_2dArray_Float" = datas.datas[:, :, featureIndex]
        compMode = (compressionMode if isinstance(compressionMode, str)
                    else compressionMode[featureName])
        resultDatas.series[featureName] = compressMatrixToArray(featureMatrixArray, compMode)
    return resultDatas
