import numpy
import numba
from scipy.stats import pearsonr
from math import e, factorial, atan2 as arcTan2, pi as PI
from typing import Callable
import warnings
from holo.types_ext import _Serie_Float, _Serie_Integer, _Serie_Boolean, _2dArray_Float


if __name__ == "__main__":
    raise RuntimeError("this file must be imported, not launched")


from .moving_average_funcs import (
    sma_numpy, sma_numpy_serie,
    vwma_numpy, vwma_numpy_serie,
    smma_numpy, smma_numpy_serie,
    ema_numpy, ema_numpy_serie,
    ema_numpy_serie2, ema_numpy_serie_senti2,
    ema_numpy_serie_normalDir, ema_numpy_serie_normalDir2,
    ema_smoothing_biDir, 
)
from .rolling_windows_funcs import (
    divide_numpy_serie, divide2_numpy_serie, 
    divide_numpy_out_serie, invert_numpy_serie,
    create_windows_numpy_serie,
    rolling_max_serie, rolling_min_serie,
    rolling_deltaArgMax_serie, rolling_deltaArgMin_serie,
    rolling_sum_aprox_serie, rolling_sum_exact_serie,
    rolling_rerange_serie,
    rateOfChange_numpy_serie,
)
from .time_indicators import (
    ... # REMOVED
)

from modules.numbaJit import fastJitter, JitType, integer, floating, floatArray, boolean
from save_formats import registerFunction

FULL_SERIE_SMOOTHING:"list[Callable[..., _Serie_Float|float]]" = [
    smma_numpy, smma_numpy_serie, ema_numpy, ema_numpy_serie,
]

warnings.simplefilter('ignore', numpy.RankWarning)









#NUMBA_MA_FUNC_SERIE_TYPE_64 = numba.types.FunctionType(numba.float64[:](numba.float64[:], numba.int64))
#NUMBA_MA_FUNC_SERIE_TYPE_32 = numba.types.FunctionType(numba.float32[:](numba.float32[:], numba.int32))



# important :
# when speaking of index:
# - 0 is the most recent data
# - len(array)-1 is the oldest data




... # REMOVED (1900 lines)