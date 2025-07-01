""" 
a file dedicated to the funcs that do moving averages
"""
import numpy
import numba

from modules.numbaJit import fastJitter, integer, floating, floatArray, boolean

from holo.types_ext import _Serie_Float, _Serie_Integer, _Serie_Boolean, _2dArray_Float

if __name__ == "__main__":
    raise RuntimeError("this file must be imported from the root of the project")


from .rolling_windows_funcs import rolling_sum_exact_serie, divide2_numpy_serie
from save_formats import registerFunction

"""
important :
when speaking of index:
- 0 is the most recent data
- len(array)-1 is the oldest data
"""


### moving average (SMA) range : ]-inf, inf[
def sma_numpy(array:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    """if index <= 0 --> invalide value returned"""
    if index <= array.size - nbPeriodes:
        return array[index: index+nbPeriodes].sum() / nbPeriodes

    return array[index: ].sum() / (array.size - index)

@registerFunction
@fastJitter(floatArray(floatArray, integer))
def sma_numpy_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    # compute most of the sma serie (the periodes will have the wrong weights)
    smaSerie = rolling_sum_exact_serie(array, nbPeriodes) / nbPeriodes
    # reajust the coeff for the first periodes
    start = max(0, len(array) - nbPeriodes+1) # avoid starting at negative index
    for index in range(start, len(array)):
        smaSerie[index] *= nbPeriodes / (len(array) - index)

    return smaSerie
###

def sma_inv_numpy(smaValue:float, array:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    if index <= array.size - nbPeriodes:
        return nbPeriodes * smaValue - array[index+1: index+nbPeriodes].sum()

    return (array.size - index) * smaValue - array[index+1:].sum()

def sma_inv_numpy_serie(smaSerie:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    array = numpy.empty_like(smaSerie)

    for index in range(array.size-1, array.size-nbPeriodes-1, -1): #(len-1  ->  len-nbperiode)
        array[index] = (array.size-index)*smaSerie[index] - array[index+1:].sum()

    for index in range(array.size-nbPeriodes-1, -1, -1): #(len-nbperiode-1  ->  0)
        array[index] = nbPeriodes*smaSerie[index] - array[index+1: index+nbPeriodes].sum()

    return array
###


### volume weighted moving average (VWMA) range : ]-inf, inf[
@fastJitter(floating(floatArray, floatArray, integer, integer))
def vwma_numpy(array:_Serie_Float, volumeArray:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    """if index <= 0 --> invalide value returned"""
    weigths_sum:float = numpy.sum(volumeArray[index: index+nbPeriodes])
    if weigths_sum != 0.0:
        return numpy.sum(
                array[index: index+nbPeriodes] * volumeArray[index: index+nbPeriodes]
            ) / weigths_sum
    return array[index]
    # old version
    # return (array[index: index+nbPeriodes] * volumeArray[index: index+nbPeriodes]).sum() / volumeArray[index: index+nbPeriodes].sum()

"""# old version
@numba.jit((numba.float64[:], numba.float64[:], numba.int64), nopython=True, nogil=True, cache=NUMBA_CACHE)
def vwma_numpy_serie(array:_Serie_Float, volumeArray:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    wsma_serie:_Serie_Float = numpy.empty_like(array)
    volumeAndArray_serie:_Serie_Float = array * volumeArray
    for index in range(0, array.size):
        divisor:float = numpy.sum(volumeArray[index: index+nbPeriodes])
        if divisor != 0.:
            wsma_serie[index] = numpy.sum(volumeAndArray_serie[index: index+nbPeriodes]) / divisor
        else:
             wsma_serie[index] = 0.

    return wsma_serie"""

@registerFunction
@fastJitter(floatArray(floatArray, floatArray, integer))
def vwma_numpy_serie(array:_Serie_Float, volumeArray:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    # wsma_serie:_Serie_Float = numpy.empty_like(array)
    # volumeAndArray_serie:_Serie_Float = array * volumeArray
    # weighted_sum_value:float = 0.0
    # divisor:float = 0.0

    # for index in range(len(array)-1, len(array)-nbPeriodes-1, -1): #(last  ->  last-nbPeriodes)
    #     weighted_sum_value += volumeAndArray_serie[index]
    #     divisor += volumeArray[index]
    #     if divisor != 0.:
    #         wsma_serie[index] = weighted_sum_value / divisor
    #     else:
    #          wsma_serie[index] = array[index]

    # for index in range(len(array)-1, -1, -1): #(last-nbPeriodes-1  ->  0)
    #     weighted_sum_value += volumeAndArray_serie[index] - volumeAndArray_serie[index+nbPeriodes+1]
    #     divisor += volumeArray[index] - volumeArray[index+nbPeriodes+1]
    #     if divisor != 0.:
    #         wsma_serie[index] = weighted_sum_value / divisor
    #     else:
    #          wsma_serie[index] = array[index]

    # return wsma_serie

    return divide2_numpy_serie(
        rolling_sum_exact_serie(array * volumeArray, nbPeriodes),
        rolling_sum_exact_serie(volumeArray, nbPeriodes),
        array
    )
###


### smoothed moving average (SMMA) range : ]-inf, inf[
def smma_numpy(array:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    res = array[-1]
    alpha = 1 / nbPeriodes

    for index in range(array.size-2, index-1, -1): #(len-2  ->  index)
        res = res * (1 - alpha) + array[index] * alpha

    return res

@registerFunction
@fastJitter(floatArray(floatArray, integer))
def smma_numpy_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    smmaSerie = numpy.empty_like(array)
    smmaSerie[-1] = array[-1]
    alpha = 1 / nbPeriodes

    for index in range(array.size-2, -1, -1): #(len-2  ->  0)
        smmaSerie[index] = smmaSerie[index + 1] * (1 - alpha) + array[index] * alpha

    return smmaSerie


def smma_inv_numpy(smmaValue:float, array:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    if index < array.size - 1:
        alpha = 1/ nbPeriodes
        return (smmaValue - (1-alpha) * smma_numpy(array=array, index=index+1, nbPeriodes=nbPeriodes)) / alpha

    return smmaValue

def smma_inv_numpy_serie(smmaSerie:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    array = numpy.empty_like(smmaSerie)
    array[-1] = smmaSerie[-1]
    alpha = 1 / nbPeriodes

    for index in range(array.size-2, -1, -1): #(len-2  ->  0)
        array[index] = (smmaSerie[index] - (1-alpha)*smmaSerie[index+1]) / alpha

    return array
###


### exponential moving average (EMA) range : ]-inf, inf[
def ema_numpy(array:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    res = array[-1]
    alpha = 2 / (nbPeriodes + 1)

    for index in range(array.size-2, index-1, -1): #(len-2  ->  index)
        res = res * (1 - alpha) + array[index] * alpha

    return res

@registerFunction
@fastJitter(floatArray(floatArray, integer))
def ema_numpy_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    emaSerie = numpy.empty_like(array)
    emaSerie[-1] = array[-1]
    alpha = 2 / (nbPeriodes + 1)

    for index in range(array.size-2, -1, -1): #(len-2  ->  0)
        emaSerie[index] = emaSerie[index + 1] * (1 - alpha) + array[index] * alpha

    return emaSerie

@registerFunction
@fastJitter(floatArray(floatArray, floating))
def ema_numpy_serie2(array:_Serie_Float, coef:float)->_Serie_Float: #DONE
    emaSerie = numpy.empty_like(array)
    emaSerie[-1] = array[-1]
    for index in range(array.size-2, -1, -1): #(len-2  ->  0)
        emaSerie[index] = emaSerie[index + 1] * (1 - coef) + array[index] * coef
    return emaSerie

@registerFunction
@fastJitter(floatArray(floatArray, integer))
def ema_numpy_serie_normalDir(array:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    """this version consider the first data at index 0"""
    emaSerie = numpy.empty_like(array)
    emaSerie[0] = array[0]
    alpha = 2 / (nbPeriodes + 1)
    for index in range(1, len(array)): #(1 -> size-1)
        emaSerie[index] = emaSerie[index - 1] * (1 - alpha) + array[index] * alpha
    return emaSerie

@registerFunction
@fastJitter(floatArray(floatArray, floating))
def ema_numpy_serie_normalDir2(array:_Serie_Float, coef:float)->_Serie_Float: #DONE
    """this version consider the first data at index 0"""
    emaSerie = numpy.empty_like(array)
    emaSerie[0] = array[0]
    for index in range(1, len(array)): #(1 -> size-1)
        emaSerie[index] = emaSerie[index - 1] * (1 - coef) + array[index] * coef
    return emaSerie

@registerFunction
@fastJitter(floatArray(floatArray, floating, floating))
def ema_numpy_serie_senti2(array:_Serie_Float, coef:float, tolerence:float)->_Serie_Float: #DONE
    """this version consider the first data at index 0, \
        keep the near 1.0 and near 0.0, and clip the values to [0.0, 1.0]"""
    emaSerie = numpy.empty_like(array)
    emaSerie[0] = array[0]
    for index in range(1, len(array)): #(1 -> size-1)
        value: float = array[index]
        if (value > 0.0+tolerence) and (value < 1.0-tolerence):
            emaSerie[index] = emaSerie[index - 1] * (1 - coef) + value * coef
        elif value >= 1.0: emaSerie[index] = 1.0
        elif value <= 0.0: emaSerie[index] = 0.0
        else: emaSerie[index] = value # => within the tolerance => no ema
            
    return emaSerie


def ema_smoothing_biDir(arr:_Serie_Float, coef:float)->_Serie_Float:
    return (ema_numpy_serie2(arr, coef) + ema_numpy_serie_normalDir2(arr, coef)) / 2


def ema_inv_numpy(emaValue:float, array:_Serie_Float, index:int, nbPeriodes:int)->float: #DONE
    if index < array.size - 1:
        alpha = 2/ (nbPeriodes + 1)
        return (emaValue - (1-alpha) * ema_numpy(array=array, index=index+1, nbPeriodes=nbPeriodes)) / alpha

    return emaValue

@registerFunction
def ema_inv_numpy_serie(emaSerie:_Serie_Float, nbPeriodes:int)->_Serie_Float: #DONE
    array = numpy.empty_like(emaSerie)
    array[-1] = emaSerie[-1]
    alpha = 2/ (nbPeriodes + 1)

    for index in range(array.size-2, -1, -1): #(len-2  ->  0)
        array[index] = (emaSerie[index] - (1-alpha)*emaSerie[index+1]) / alpha

    return array
###
