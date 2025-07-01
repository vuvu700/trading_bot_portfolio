""" 
a file dedicated to the funcs that are optimizable with special rolling windows technics
or repetives loopes that maight leed to errors
"""
import numpy
import numba

if __name__ == "__main__":
    raise RuntimeError("this file must be imported from the root of the project")

from modules.numbaJit import fastJitter, JitType, integer, floating, floatArray, boolean, floatMatrix
from save_formats import registerFunction

from holo.types_ext import _Serie_Float, _Serie_Integer, _Serie_Boolean, _2dArray_Float


"""
important :
when speaking of index:
- 0 is the most recent data
- len(array)-1 is the oldest data
"""



@registerFunction
@fastJitter(floatArray(floatArray, floatArray, floating))
def divide_numpy_serie(
        arrayNumerator:_Serie_Float, arrayDivisor:_Serie_Float,
        zeroDivReplacement:"float|numpy.floating")->_Serie_Float:
    Divided_serie:_Serie_Float = numpy.empty_like(arrayNumerator)
    
    for index in range(len(arrayNumerator)-1, -1, -1): # (last -> 0)
        if arrayDivisor[index] != 0.0:
            Divided_serie[index] = arrayNumerator[index] / arrayDivisor[index]
        else:Divided_serie[index] = zeroDivReplacement
    
    return Divided_serie

@registerFunction
@fastJitter(floatArray(floatArray, floatArray, floatArray))
def divide2_numpy_serie(
        arrayNumerator:_Serie_Float, arrayDivisor:_Serie_Float,
        zeroDivReplacementArray:_Serie_Float)->_Serie_Float:
    Divided_serie:_Serie_Float = numpy.empty_like(arrayNumerator)
    
    for index in range(len(arrayNumerator)-1, -1, -1): # (last -> 0)
        if arrayDivisor[index] != 0.0:
            Divided_serie[index] = arrayNumerator[index] / arrayDivisor[index]
        else:Divided_serie[index] = zeroDivReplacementArray[index]
    
    return Divided_serie

@registerFunction
@fastJitter(floatArray(floatArray, floating))
def invert_numpy_serie(
        arrayDivisor:_Serie_Float, zeroDivReplacement:"float|numpy.floating")->_Serie_Float:
    """compute 1 / `arrayDivisor` and use `zeroDivReplacement` as result when there is a 0 as divisor"""
    Divided_serie:_Serie_Float = numpy.empty_like(arrayDivisor)
    
    for index in range(len(arrayDivisor)-1, -1, -1): # (last -> 0)
        if arrayDivisor[index] != 0.0:
            Divided_serie[index] = 1 / arrayDivisor[index]
        else:Divided_serie[index] = zeroDivReplacement
    
    return Divided_serie

@registerFunction
@fastJitter(floatArray(floatArray, floatArray, floating, floatArray))
def divide_numpy_out_serie(
        arrayNumerator:_Serie_Float, arrayDivisor:_Serie_Float,
        zeroDivReplacement:"float|numpy.floating", outArray:_Serie_Float)->_Serie_Float:
    
    for index in range(len(arrayNumerator)-1, -1, -1): # (last -> 0)
        if arrayDivisor[index] != 0.0:
            outArray[index] = arrayNumerator[index] / arrayDivisor[index]
        else:outArray[index] = zeroDivReplacement
        
    return outArray

### Normalize (*_norm) range(keepZero) : [-1, 1], range(not keepZero) : [0, 1]
def _normalize_numpy(array:_Serie_Float, index:int, nbPeriodes:int, doKeepZero:bool)->float: #TOTEST
    if doKeepZero is True:
        div = min(array[index: index+nbPeriodes]) if array[index] >= 0 else max(array[index: index+nbPeriodes])
        if div == 0:
            return 0.
        return  array[index] / div

    else:
        tmp = array[index] - min(array[index: index+nbPeriodes])
        maxTmp = max(tmp)
        if maxTmp == 0:
            return 0.
        return tmp / maxTmp

@registerFunction
def normalize_numpy_serie(array:_Serie_Float, nbPeriodes:int, doKeepZero:bool)->"_Serie_Float": #TOTEST
    lowest_serie = rolling_min_serie(array, nbPeriodes)

    if doKeepZero is True:
        highest_serie = rolling_max_serie(array, nbPeriodes)

        div = lowest_serie * (array < 0) + (array >= 0) * highest_serie
        return divide_numpy_serie(array, div, zeroDivReplacement=0.0)


    else:
        tmp = array - lowest_serie
        highest_tmp_serie = rolling_max_serie(tmp, nbPeriodes)
        return divide_numpy_serie(tmp, highest_tmp_serie, zeroDivReplacement=0.0)
###



### create all the moving windows
@registerFunction
@fastJitter(floatMatrix(floatArray, integer))
def create_windows_numpy_serie(array:_Serie_Float, nbPeriodes:int)->_2dArray_Float:
    """it will create all the FULL windows over the array, \
        !! the shape of the returned object is (len(array)-nbPeriodes, nbPeriodes) !!"""
    windowed_array:_2dArray_Float = numpy.empty(
        (len(array) - nbPeriodes, nbPeriodes), dtype=array.dtype)

    for index in range(nbPeriodes, len(array)):
        windowed_array[index] = array[index - nbPeriodes: index]

    return windowed_array
###



"""# old version
@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True, cache=NUMBA_CACHE)
def rolling_max_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    maximum_serie = numpy.empty_like(array)

    for index in range(0, array.size - nbPeriodes):
        maxi = array[index]
        for shiftIndex in range(1, nbPeriodes):
            if maxi < array[index + shiftIndex]:
                maxi = array[index + shiftIndex]
        maximum_serie[index] = maxi

    deltaShift = 0
    for index in range(array.size - nbPeriodes, array.size):
        maxi = array[index]
        for shiftIndex in range(1, nbPeriodes - deltaShift):
            if maxi < array[index + shiftIndex]:
                maxi = array[index + shiftIndex]
        maximum_serie[index] = maxi
        deltaShift += 1

    return maximum_serie

@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True, cache=NUMBA_CACHE)
def rolling_min_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    minimum_serie = numpy.empty_like(array)

    for index in range(0, array.size - nbPeriodes):
        mini = array[index]
        for shiftIndex in range(1, nbPeriodes):
            if mini > array[index + shiftIndex]:
                mini = array[index + shiftIndex]
        minimum_serie[index] = mini

    deltaShift = 0
    for index in range(array.size - nbPeriodes, array.size):
        mini = array[index]
        for shiftIndex in range(1, nbPeriodes - deltaShift):
            if mini > array[index + shiftIndex]:
                mini = array[index + shiftIndex]
        minimum_serie[index] = mini
        deltaShift += 1

    return minimum_serie
"""

@registerFunction
@fastJitter(floatArray(floatArray, integer))
def rolling_max_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    maximum_serie = numpy.empty_like(array)
    maxi:float = array[-1]
    deltaIndexMaxi:int = 0

    for index in range(len(array)-1, -1, -1): # (size-1 -> 0)
        if array[index] > maxi:
            maxi = array[index]
            deltaIndexMaxi = 0

        elif deltaIndexMaxi >= nbPeriodes:
            deltaIndexMaxi = int(numpy.argmax(array[index: index + nbPeriodes]))
            maxi = array[index + deltaIndexMaxi]

        maximum_serie[index] = maxi
        deltaIndexMaxi += 1
        
    return maximum_serie

@registerFunction
@fastJitter(floatArray(floatArray, integer))
def rolling_min_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    minimum_serie = numpy.empty_like(array)
    mini:float = array[-1]
    deltaIndexMini:int = 0

    for index in range(len(array)-1, -1, -1): # (size-1 -> 0)
        if array[index] < mini:
            mini = array[index]
            deltaIndexMini = 0

        elif deltaIndexMini >= nbPeriodes:
            deltaIndexMini = int(numpy.argmin(array[index: index + nbPeriodes]))
            mini = array[index + deltaIndexMini]
        
        minimum_serie[index] = mini
        deltaIndexMini += 1
        
    return minimum_serie



@registerFunction
@fastJitter(JitType("int", nbDims=1, fixedBitwidth=64)(floatArray, integer))
def rolling_deltaArgMax_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Integer:
    deltaArgMax_serie:_Serie_Integer = numpy.empty(array.shape, dtype=numpy.int64)
    maxi:float = array[-1]
    deltaIndexMaxi:int = 0

    for index in range(len(array)-1, -1, -1): # (size-1 -> 0)
        if array[index] >= maxi:
            maxi = array[index]
            deltaIndexMaxi = 0

        elif deltaIndexMaxi >= nbPeriodes:
            deltaIndexMaxi = int(numpy.argmax(array[index: index + nbPeriodes]))
            maxi = array[index + deltaIndexMaxi]
        
        deltaArgMax_serie[index] = deltaIndexMaxi
        deltaIndexMaxi += 1

    return deltaArgMax_serie


@registerFunction
@fastJitter(JitType("int", nbDims=1, fixedBitwidth=64)(floatArray, integer))
def rolling_deltaArgMin_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Integer:
    deltaArgMin_serie:_Serie_Integer = numpy.empty(array.shape, dtype=numpy.int64)
    mini:float = array[-1]
    deltaIndexMini:int = 0

    for index in range(len(array)-1, -1, -1): # (size-1 -> 0)
        if array[index] <= mini:
            mini = array[index]
            deltaIndexMini = 0

        elif deltaIndexMini >= nbPeriodes:
            deltaIndexMini = int(numpy.argmin(array[index: index + nbPeriodes]))
            mini = array[index + deltaIndexMini]
        
        deltaArgMin_serie[index] = deltaIndexMini
        deltaIndexMini += 1

    return deltaArgMin_serie



@registerFunction
@fastJitter(floatArray(floatArray, integer))
def rolling_sum_aprox_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    "this version compute the rolling sum faster but it has floating precision error, in O(N)"
    rolling_Sum_serie:_Serie_Float = numpy.empty_like(array)
    rolling_Sum_value:float = 0.0
    
    for index in range(len(array)-1, len(array)-nbPeriodes-1, -1): # (last -> last-nbPeriodes)
        rolling_Sum_value += array[index]
        rolling_Sum_serie[index] = rolling_Sum_value
        
    for index in range(len(array)-nbPeriodes-1, -1, -1): # (last-nbPeriodes -> 0)
        rolling_Sum_value += array[index] - array[index+nbPeriodes]
        rolling_Sum_serie[index] = rolling_Sum_value
    
    return rolling_Sum_serie


@registerFunction
@fastJitter(floatArray(floatArray, integer))
def rolling_sum_exact_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    "this version compute the rolling sum with no floating precision error, in O(N * nbPeriodes)"
    rolling_Sum_serie:_Serie_Float = numpy.empty_like(array)
        
    for index in range(len(array)-1, -1, -1): # (last -> last-nbPeriodes)
        rolling_Sum_serie[index] = numpy.sum(array[index: index+nbPeriodes])
    
    return rolling_Sum_serie


@registerFunction
@fastJitter(floatArray(floatArray, integer))
def rolling_rerange_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    """change the range from ]-inf, +inf[ -> [0, 1]"""
    reranged_serie:_Serie_Float = numpy.empty_like(array)

    maxi:float = array[-1]
    mini:float = maxi
    deltaIndexMaxi:int = 0
    deltaIndexMini:int = 0

    for index in range(len(array)-1, -1, -1): # (last -> 0)
        value = array[index]
        # efficient rolling max
        if value > maxi:
            maxi = value
            deltaIndexMaxi = 0
        elif deltaIndexMaxi >= nbPeriodes:
            deltaIndexMaxi = int(numpy.argmax(array[index: index + nbPeriodes]))
            maxi = array[index + deltaIndexMaxi]

        # efficient rolling min
        if value < mini:
            mini = value
            deltaIndexMini = 0
        elif deltaIndexMini >= nbPeriodes:
            deltaIndexMini = int(numpy.argmin(array[index: index + nbPeriodes]))
            mini = array[index + deltaIndexMini]

        # rerange
        deltaMaxiMini = maxi - mini
        if deltaMaxiMini != 0.:
            reranged_serie[index] = (value - mini) / (maxi - mini)
        else:
            reranged_serie[index] = 0.5

        # setup next step
        deltaIndexMaxi += 1
        deltaIndexMini += 1

    return reranged_serie


@registerFunction
@fastJitter(floatArray(floatArray, integer))
def rateOfChange_numpy_serie(array:_Serie_Float, nbPeriodes:int)->_Serie_Float:
    ROC_serie:_Serie_Float = numpy.empty_like(array)
    ROC_serie[-1] = 0.0
    
    array_late_value:float = array[-1]
    if array_late_value != 0.0:
        for index in range(len(array)-2, len(array)-1-nbPeriodes-1, -1): # (last-1 -> last-nbPeriodes)
            ROC_serie[index] = (array[index] - array_late_value) / array_late_value
    else: ROC_serie[-nbPeriodes: -1] = 0.0
    
    for index in range(len(array)-1-nbPeriodes-1, -1, -1): # (last-nbPeriodes-1 -> 0)
        array_late_value = array[index + nbPeriodes]
        if array_late_value != 0.0:
            value = (array[index] - array_late_value) / array_late_value
            if value == numpy.inf:
                value = 0.0
            ROC_serie[index] = value
            
        else: ROC_serie[index] = 0.0
    
    return ROC_serie
