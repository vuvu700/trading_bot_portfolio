import numpy
import numba

from holo.types_ext import _Serie_Float, _Serie_Integer, _Serie_Boolean

from modules.numbaJit import fastJitter, floatArray, floating, boolean, void



########################## rerange
def reRange(values: _Serie_Float)->_Serie_Float:
    """compute the rerange of the `values` serie from ]-inf, +inf[ to [0, 1] by truncating,\
        values outside of [0, 1] will loose their details and become respectively 0 or 1\n
     * `values` is the serie that will be used, it will not be modified\n
    return the serie of reranged values in [0, 1]"""
    return numpy.clip(values, 0.0, 1.0)


def reRange2(values:_Serie_Float, alpha:float, noLow:bool=False)->_Serie_Float:
    """compute the rerange of the `values` serie from ]-inf, +inf[ to ]0, 1[ without lossing infos,\
        it is using an hybide linear and sigmoid function, \
        the function is linear in for values in [`alpha`, 1-`alpha`],\
        and sigmoidal betwin ]-inf, `alpha`[ and ]1-`alpha`, +inf[,\
        the function is C1(R), so smooth output for all values in R\n
     * `values` is the serie that will be used, it will not be modified\n
     * `alpha` must be in [0, 0.5], it is the thresshold betwin liear and sigmoid mode\n
     * `noLow` mean that the function will be linear for the lower interval (hard minimum 0)\n
    return the serie of reranged values in ]0, 1["""
    if (alpha < 0.0) or (alpha > 0.5):
        raise ValueError(f"the `alpha({alpha}) parameter need to be in [0.0, 0.5]")
    return _reRange2_internal(values=values, alpha=alpha, noLow=noLow)

@fastJitter(floatArray(floatArray, floating, boolean))
def _reRange2_internal(values:_Serie_Float, alpha:float, noLow:bool)->_Serie_Float:
    """internal function that do the computation of reRange2"""
    outputSerie:_Serie_Float = numpy.empty_like(values)

    # pre computation
    beta:float = 1 / alpha
    beta1:float = beta - 1
    omega:float = beta * beta / beta1
    shiftG2:float = 1 - (alpha + 2 * numpy.log(beta1) / omega) # is the shift for g2

    # computation of every single values
    for index in range(values.shape[0]):
        val:float = values[index]
        if (noLow is False) and (val < alpha):
            # use g1(x)
            val = 1 / (1 + beta1 * numpy.exp(-omega * (val - alpha)))
        elif (val > (1-alpha)):
            # use g1(x)
            val = 1 / (1 + beta1 * numpy.exp(-omega * (val - shiftG2)))
            # === 1 / (1 + (1 / beta1) * numpy.exp(-omega * (val - (1 - alpha))))
        # else:
        #   linear part => no changes (val = val)

        outputSerie[index] = val

    return outputSerie


def reRange3(values:_Serie_Float, coeff:float, reLimit:bool=False)->_Serie_Float:
    """compute the rerange of the `values` serie from ]-inf, +inf[ (centered in 0.5) \
        to ]0, 1[ without lossing infos,\
        using a sigmoid function\n
     * `values` is the serie that will be used, it will not be modified\n
     * `coeff` is how much the values will be stretched\n
     * `reLimit` is whether the function will correct the fact that reRange3([0, 1], ...) is far from [0, 1]\n
    return the serie of reranged values in ]0, 1["""
    return _reRange3_internal(values=values, coeff=coeff, reLimit=reLimit)

@fastJitter(floatArray(floatArray, floating, boolean))
def _reRange3_internal(values:_Serie_Float, coeff:float, reLimit:bool)->_Serie_Float:
    """internal function that do the computation of reRange3"""
    outputSerie:_Serie_Float = numpy.empty_like(values)

    # computation of every single values
    for index in range(len(values)):
        outputSerie[index] = 1 / (1 + numpy.exp(-coeff * (values[index] - 0.5)))

    if reLimit is True:
        lowLimit = 1 / (1 + numpy.exp(-coeff * (0.0 - 0.5)))
        upperLimit = 1 / (1 + numpy.exp(-coeff * (1.0 - 0.5)))
        for index in range(len(values)):
            outputSerie[index] = (outputSerie[index] - lowLimit) / (upperLimit - lowLimit)

    return outputSerie



@fastJitter(void(floatArray, floating, floating), parallel=True)
def _moveRange(array:"_Serie_Float", mini:float, maxi:float)->None:
    """move, inplace, the range of the values from [0, 1] to [mini, maxi]"""
    rangeWidth: float = (maxi - mini)
    for index in numba.prange(len(array)):
        array[index] = (array[index] * rangeWidth) + mini

