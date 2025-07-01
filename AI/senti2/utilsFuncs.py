import numba
import numpy
from math import ceil
from typing import Generator, Any, Union, Dict, Tuple
import time
import warnings

from modules.diskSavesRam import SaveDictModule

from holo.types_ext import _Serie_Float, _1dArray_Float, _2dArray_Float, _3dArray_Float, _Serie_Integer

try: from . import _Model_Senti2 # just for typing
except: pass

###### assistance function optimized for split_datas of AI.senti2
@numba.jit(
    numba.float64[:, :, :](numba.float64[:, :, :], numba.float64[:], numba.int64, numba.int64, numba.int64),
    nopython=True, nogil=True, cache=True)
def _copy_serie_into_X_All(
        X_All:_3dArray_Float, serieToCopy:_Serie_Float,
        index_shift:int, index_input_Serie:int, nbPeriodes:int
        )->_3dArray_Float:
    for index_train in range(len(X_All)):
        start_index_data = index_train + index_shift
        X_All[index_train, index_input_Serie, :] = \
            serieToCopy[start_index_data: start_index_data + nbPeriodes]
    return X_All

@numba.jit(
    numba.float64[:, :](numba.float64[:, :], numba.float64[:], numba.int64, numba.int64),
    nopython=True, nogil=True, cache=True)
def _copy_serie_into_Y_All(
        Y_All:_2dArray_Float, serieToCopy:_Serie_Float,
        index_shift:int, index_output_Serie:int
        )->_2dArray_Float:
    for index_train in range(len(Y_All)):
        index_data = index_train + index_shift
        Y_All[index_train, index_output_Serie] = serieToCopy[index_data]
    return Y_All


@numba.jit(
    [numba.float64[:,:,:](numba.float64[:,:,:], numba.float64[:,:], numba.float64),
     numba.float32[:,:,:](numba.float32[:,:,:], numba.float32[:,:], numba.float32)],
    nopython=True, nogil=True, cache=True)
def _randomRepares_X_train(
        X_train:_3dArray_Float, X_restore:_2dArray_Float,
        repareProportion:float)->_3dArray_Float:
    """it repares APROXIMATIVELY the porpotion asked (for optim purpose)"""
    # repare a certain proportion of the X_train before fitting
    # even if tha datas are shuffuled the datas that will be repared
    # need to be chosen randomly in order to avoid learning bias
    if repareProportion <= 0.0:
        return X_train

    for index in range(len(X_train)):
        if numpy.random.random() < repareProportion:
            X_train[index, :, -1] = X_restore[index]
            #nbRepares += 1

    return X_train

@numba.jit(
    numba.float64[:,:,:](numba.float64[:,:,:], numba.float64),
    nopython=True, nogil=True, cache=True)
def add_noise_3d(array:_3dArray_Float, coef:float):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                array[i, j, k] += coef * (numpy.random.random()*2.0 -1.0)
    return array

@numba.jit(
    numba.float64[:,:](numba.float64[:,:], numba.float64),
    nopython=True, nogil=True, cache=True)
def add_noise_2d(array:_3dArray_Float, coef:float):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] += coef * (numpy.random.random()*2.0 -1.0)
    return array

@numba.jit(
    numba.float64[:,:,:](
        numba.float64[:,:,:], numba.float64[:,:], numba.float64[:], numba.int64,
    ), nopython=True, nogil=True, cache=True)
def _updateX_train(
        X_compute:_3dArray_Float, Y_compute:_2dArray_Float,
        Y_before:_1dArray_Float, nbPeriodes:int)->_3dArray_Float:
    """used in _calc_X_train_by_iteration"""
    for i in range(nbPeriodes):
        X_compute[i: , -(i+1), -1] = Y_compute[i: ].flatten()
        X_compute[i, :nbPeriodes-i, -1] = Y_before[i: ]
    return X_compute


##### general pupose funcs



def getLr(lossVal:float, lossLrTable:"list[tuple[float, float]]")->float:
    """return the learning rate regarding of the current loss, and a table\n
    linearly interpolates the losses inside de table and retun the bounds of the table if outside\n
    `lossLrTable` is the table that associate losses to lr like [(loss, lr), ...]\n"""
    lossLrTable = sorted(lossLrTable, key=lambda key:key[0], reverse=True)
    calcLr = lambda ratio, prevLr, lr: prevLr + ratio*(lr-prevLr)
    calcRatio = lambda lossVal, prevLoss, loss: (lossVal - prevLoss) / (loss - prevLoss)

    # if over max loss of the table => return its lr
    if lossVal >= lossLrTable[0][0]:
        return lossLrTable[0][1]
    # if under min loss of the table => return its lr
    elif lossVal <= lossLrTable[-1][0]:
        return lossLrTable[-1][1]

    # find the correct loss interval for lossVal and linearly interpolate the lr
    for index, (loss, lr) in enumerate(lossLrTable[1: ], start=1):
        (prevLoss, prevLr) = lossLrTable[index-1]
        if lossVal >= loss: # => (prevLoss > lossVal >= loss)
            return calcLr(calcRatio(lossVal, prevLoss, loss), prevLr, lr)
    raise ValueError("should to reach here")


def batchs_generator(
        X_data:"_3dArray_Float", Y_data:"_2dArray_Float",
        batch_size:int, shuffle:bool,
        )->"Generator[tuple[_3dArray_Float, _2dArray_Float], None, None]":

    if shuffle is True:
        # shuffle the training data
        permut_table:"_Serie_Integer" = numpy.random.permutation(len(X_data))
        X_data = X_data[permut_table]
        Y_data = Y_data[permut_table]

    # calculate the number of batches for training data
    num_batches = ceil(len(X_data) / batch_size)

    # yield each batch of training data
    for batch_index in range(num_batches):
        start_idx = batch_index * batch_size
        end_idx = (batch_index + 1) * batch_size
        yield X_data[start_idx:end_idx], Y_data[start_idx:end_idx]

XY_datas_SaveDicts = Union[
    SaveDictModule[str, Dict[str, numpy.ndarray]],
    Tuple[SaveDictModule[str, _3dArray_Float], SaveDictModule[str, _2dArray_Float]],
]
def generator_all_XY_datas(
        all_XY_datas:"XY_datas_SaveDicts",
        batch_size:int, shuffle:bool, nb_epoches:int,
        )->"Generator[tuple[_3dArray_Float, _2dArray_Float], None, None]":
    is_tuple:bool = isinstance(all_XY_datas, tuple)
    keys:"list[str]" = (all_XY_datas[0].keys() if is_tuple else all_XY_datas.keys())
    for _ in range(nb_epoches):
        for df_key in keys:
            # get the datas
            X_datas:"_3dArray_Float" = (all_XY_datas[0][df_key] if is_tuple else all_XY_datas[df_key]["X_datas"])
            Y_datas:"_2dArray_Float" = (all_XY_datas[1][df_key] if is_tuple else all_XY_datas[df_key]["Y_datas"])
            if is_tuple:
                all_XY_datas[0].unLoad_noSave(df_key)
                all_XY_datas[1].unLoad_noSave(df_key)
            else: all_XY_datas.unLoad_noSave(df_key)

            # send them to the training
            yield from batchs_generator(X_datas, Y_datas, batch_size, shuffle)
            del X_datas, Y_datas

