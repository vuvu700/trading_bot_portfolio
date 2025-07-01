if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

import tensorflow as tf
import keras
import keras.layers
import keras.optimizers
import keras.callbacks
import keras.activations
import keras.backend
import keras.utils

import os
from io import StringIO
import numpy
import json
import warnings
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Generator
from matplotlib import (pyplot as plt, figure as plt_figure)
import matplotlib.pyplot as plt


import AI
from paths_cfg import AI_SAVE_DIRECTORY, AI_CHECKPOINTS_DIRECTORY
from calculationLib import ema_numpy_serie_normalDir
from .utilsFuncs import (
    _copy_serie_into_X_All, _copy_serie_into_Y_All, add_noise_3d, add_noise_2d,
)


from holo.files import correctDirPath, mkDirRec
from holo.types_ext import _Serie_Float, _3dArray_Float, _2dArray_Float
from holo.__typing import JsonTypeAlias

AI_SAVE_SENTI2_DIRECTORY = AI_SAVE_DIRECTORY.joinpath("senti2/")
AI_SENTI2_CHECKPOINTS_DIRECTORY = AI_CHECKPOINTS_DIRECTORY.joinpath("senti2/")
mkDirRec(AI_SAVE_SENTI2_DIRECTORY)
mkDirRec(AI_SENTI2_CHECKPOINTS_DIRECTORY)

RELIABILITY_MODE:bool = False
def enable_reliable_mode()->None:
    global RELIABILITY_MODE;  RELIABILITY_MODE = True
    tf.compat.v1.disable_v2_behavior()

def get_RELIABILITY_MODE()->bool:
    return RELIABILITY_MODE
def set_RELIABILITY_MODE(value:bool):
    global RELIABILITY_MODE
    RELIABILITY_MODE = value

PLOT_FIGS:"dict[str, plt_figure.Figure]" = {}

USE_MULTIPROCESSING = False
def get_USE_MULTIPROCESSING()->bool:
    return USE_MULTIPROCESSING
def set_USE_MULTIPROCESSING(value:bool):
    global USE_MULTIPROCESSING
    USE_MULTIPROCESSING = value

#################### base model ####################

class _Base_Model_senti2():
    VERSION = "None"
    nbPeriodes:"int|None" = None
    series_selection_inputs:"list[str]|None" = None
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"
    series_selection_inputs_lagged:"list[str]|None" = None
    "the names of the series (lagged by 1) that will be selected as inputs"

    series_selection_outputs:"list[str]|None" = None
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"
    series_selection_outputs_delayed:"list[str]|None" = None
    "the names of the series (delayed by 1) that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:"int|None" = None
    number_series_outputs:"int|None" = None

    _regularize_kargs:"dict[str, Any]|None" = None

    def _popNoneSeriesSelection(self)->None:
        """transforme all the None -> [ ] in the series selection (to improve compatibility)\n
        /!\\should not be called manualy"""
        if self.series_selection_inputs is None: self.series_selection_inputs = []
        if self.series_selection_inputs_lagged is None: self.series_selection_inputs_lagged = []
        if self.series_selection_outputs is None: self.series_selection_outputs = []
        if self.series_selection_outputs_delayed is None: self.series_selection_outputs_delayed = []


    def __init__(self, _network:"None|keras.Sequential"=None) -> None:
        self.network:"None|keras.Model" = _network

        self.optimizer:"None|keras.optimizers.Optimizer"
        if (_network is not None) and ("optimizer" in dir(_network)):
            self.optimizer = _network.optimizer
        else:self.optimizer = None

        self._trainingHistory:"dict[str, list[float]]" = {}
        self.comments:"list[str]" = []
        self._plotCfg:"None|dict[str, dict[str, Any]]" = None
        self._resumeCfg:"None|list[tuple[str, dict[str, Any]]]" = None
        self.custom_loss_func:"None|str" = None
        self._custom_loss_func_kwarg:"None|dict[str, Any]" = None
        self._backTests_metrics:"None|dict[str, str]" = None

        self._current_checkpoint_nb:"None|int" = None
        self._checkpoints_directory:"None|str|Path" = None
        self._trainCalls_history:"None|list[tuple[str, dict[str, Any], bool]]" = None
        """[(funcname, kwargs(parameters only), finished), ...]"""

    def _base_train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int",
            callbacks:"list[Callable|Any]|None"=None,
            shuffle_train:bool=False)->"keras.callbacks.History":
        if self.network is None:
            raise RuntimeError("no network to train")
        else:
            if not isinstance(self.network, keras.Model):
                raise TypeError(f"network isn't a keras.Model instance, but a:{self.network.__class__} consider adding its support")

            return self.network.fit(
                X_train, Y_train,
                epochs=nbEpoches, batch_size=batchSize,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
                verbose=verbose, # type:ignore  bad config from keras
                shuffle=shuffle_train,
                workers=1, use_multiprocessing=False,
            )


    def _base_train_generator(self,
            train_generator:"Generator[tuple[_3dArray_Float, _2dArray_Float], None, None]",
            test_generator:"Generator[tuple[_3dArray_Float, _2dArray_Float], None, None]",
            steps_per_train_epoch:int, steps_per_test_epoch:int,
            nbEpoches:int, verbose:"str|int")->"keras.callbacks.History":
        if self.network is None:
            raise RuntimeError("no network to train")
        else:
            if not isinstance(self.network, keras.Model):
                raise TypeError(f"network isn't a keras.Model instance, but a:{self.network.__class__} consider adding its support")

            return self.network.fit(
                train_generator, steps_per_epoch=steps_per_train_epoch,
                validation_data=test_generator, validation_steps=steps_per_test_epoch,
                epochs=nbEpoches, workers=1, use_multiprocessing=False,
                verbose=verbose, # type:ignore  bad config from keras
            )

    def _base_evaluate(self,
            X_data:"_3dArray_Float", Y_data:"_2dArray_Float", batchSize:int,
            verbose:int, metricName:str="loss")->"float":
        """return the loss of the metric"""
        if self.network is None:
            raise RuntimeError("no network to train")
        else:
            if not isinstance(self.network, keras.Model):
                raise TypeError(f"network isn't a keras.Model instance, but a:{self.network.__class__} consider adding its support")

            metric_value:"numpy.floating|list" = self.network.evaluate(
                X_data, Y_data, batch_size=batchSize,
                verbose=verbose, # type:ignore bad config from keras
            )
            if isinstance(metric_value, list):
                raise TypeError(f"wrong type for {metricName}: {type(metric_value)}, should be a numpy.float...")
            return float(metric_value)

    @classmethod
    def _get_save_base_directory(cls)->Path:
        return AI_SAVE_SENTI2_DIRECTORY

    @classmethod
    def load(cls, filename:str, directory:"Path|None"=None)->"_Model_Senti2":
        """load the model from `directory`/`filename`.(h5|tf) \
            and its parameters from `directory`/`filename`.json\n
        `directory` can end with '/'\n"""
        if directory is None: directory = AI_SAVE_SENTI2_DIRECTORY
        extention:str = "tf" if RELIABILITY_MODE is False else "h5"
        extention2:str = "h5" if RELIABILITY_MODE is False else "tf"
        path_json = directory.joinpath(f"{filename}.json")
        path_model = directory.joinpath(f"{filename}.{extention}")

        if not os.path.lexists(path_json):
            raise FileNotFoundError(f"the json file at: '{path_json}' doesn't exist")
        if not os.path.lexists(path_model):
            path_model = directory.joinpath(f"{filename}.{extention2}")
            if not os.path.lexists(path_model):
                raise FileNotFoundError(f"the {extention} file (or {extention2} file) at: '{path_model}' doesn't exist")

        with open(path_json, mode="r") as file_json:
            data:"dict[str, Any]" = json.load(file_json)
            # the container of the config to construct the AI

        REQUIRED_ATTRIBS = [
            "VERSION",
            "nbPeriodes", "series_selection_inputs",
            "series_selection_inputs_lagged",
            "series_selection_outputs", "comments",
            "number_series_inputs", "number_series_outputs",
        ]
        missing_attribs = [attrib for attrib in REQUIRED_ATTRIBS if attrib not in data.keys()]
        if len(missing_attribs) != 0:
            raise KeyError(
                "all the required attributs are not present in "
                + f"the json save file at:'{path_json}'\n"
                + f"the the missing attributs are: {missing_attribs}")

        # load the custom func
        custom_objects:"None|dict[str, Callable[..., tf.Tensor]]" = {}
        custom_loss_func:"None|str" = data.get("custom_loss_func", None)
        _custom_loss_func_kwarg:"None|dict[str, Any]" = data.get("_custom_loss_func_kwarg", None)
        if (custom_loss_func is None) or (_custom_loss_func_kwarg is None):
            custom_objects = None
        else:
            custom_func:"Callable[..., tf.Tensor]" = get_custom_loss_funcs(custom_loss_func)(**_custom_loss_func_kwarg)
            custom_objects = {custom_func.__name__: custom_func}

        loaded_network = keras.models.load_model(path_model, compile=True, custom_objects=custom_objects)
        if loaded_network is None:
            raise ValueError(f"the network at {path_model} failed to load, return None")

        # auto find the classe of the model
        loaded_model:"_Model_Senti2" = \
            globals()[f"Model_senti2_{data['VERSION']}"](_network=loaded_network)

        # the args that will be checked (thoes whitch are set by default) to be the same (if not show a warning and set the new value)
        ARGS_TO_CHECK = [
            "VERSION",
            "nbPeriodes", "series_selection_inputs",
            "series_selection_inputs_lagged",
            "series_selection_outputs",
            "number_series_inputs", "number_series_outputs",
        ]
        for args_name, value in data.items():
            # fix dunders attributs
            if args_name.startswith("__"):
                raise KeyError("dunders not supported because fuck, it is too hard to fix")

            if args_name in ARGS_TO_CHECK:
                # check if the values are the same
                if value != loaded_model.__getattribute__(args_name):
                    warnings.warn(
                        f"the attribute:'{args_name}' are differents:\n"
                        + f"{value} != {loaded_model.__getattribute__(args_name)}\n"
                        + f" => the new value will replace the existing value")
                    loaded_model.__setattr__(args_name, value)
                else:
                    continue

            else:
                # every values from the model being loaded that are not meant to be checked are directly setted (for future compat)
                loaded_model.__setattr__(args_name, value)

        return loaded_model


    def save(self, filename:str, directory:Path=AI_SAVE_SENTI2_DIRECTORY)->None:
        """save the model to `directory`/`filename`.tf \
            and its parameters to `directory`/`filename`.json\n
        `directory` can end with '/'\n"""
        extention:str = "tf"
        if RELIABILITY_MODE is True:
            extention = "h5"
        mkDirRec(directory)
        path_json = directory.joinpath(f"{filename}.json")
        path_tf = directory.joinpath(f"{filename}.{extention}")

        if self.network is None:
            raise ValueError("can't save a model with a None network")

        data_json = {
            "VERSION":self.VERSION,
            "nbPeriodes":self.nbPeriodes,
            "series_selection_inputs":self.series_selection_inputs,
            "series_selection_inputs_lagged":self.series_selection_inputs_lagged,
            "series_selection_outputs":self.series_selection_outputs,
            "number_series_inputs":self.number_series_inputs,
            "number_series_outputs":self.number_series_outputs,
            "comments":self.comments,
            "_trainingHistory":self._trainingHistory,
            "_plotCfg":self._plotCfg,
            #"_resumeCfg":self._resumeCfg, #it can contain funcs(they dont serialize)
            "custom_loss_func":self.custom_loss_func,
            "_custom_loss_func_kwarg":self._custom_loss_func_kwarg,
            "_backTests_metrics":self._backTests_metrics,
            "_current_checkpoint_nb":self._current_checkpoint_nb,
            "_checkpoints_directory":
                (self._checkpoints_directory.as_posix() if isinstance(self._checkpoints_directory, Path)
                else self._checkpoints_directory),
            "_trainCalls_history":self._trainCalls_history,
        }

        with open(path_json, mode="w") as file_json:
            json.dump(data_json, file_json, indent="\t", sort_keys=False)

        self.network.save(
            path_tf,
            overwrite=True, include_optimizer=True,
            save_format=extention,
        )

    @classmethod
    def _applie_random_noise_on_datas(cls, X_All:"_3dArray_Float",
            size_train:"int|None", applieGlobalNoise:"None|float", applieSenti2Noise:"None|float")->None:
        if applieGlobalNoise is not None:
            X_All[: size_train] = add_noise_3d(X_All[: size_train], float(applieGlobalNoise))

        if applieSenti2Noise is not None:
            applieSenti2Noise = float(applieSenti2Noise)
            X_All[: size_train, -1, :] = add_noise_2d(X_All[: size_train, -1, :], float(applieSenti2Noise))


    def split_datas(self,
            series_data:"dict[str, _Serie_Float]",
            skipNbFirst:int, skipNbLast:int, testSamplesProportion:float,
            chooseTestSamplesRandomly:bool=False,  randomizeSamples:bool=False,
            applieGlobalNoise:"None|float"=None, applieSenti2Noise:"None|float"=None,
            _samples_to_do:"bool|None"=None,
            )->"tuple[tuple[_3dArray_Float, _2dArray_Float], tuple[_3dArray_Float, _2dArray_Float]]":
        """TO optim with numba return the ((X_train, Y_train), (X_test, Y_test)) of the selected series(ordered in the same order)\n
        `series_data` is the dataset of regularized serises, where the series will be extracted (must conatain the keys)\n
        `skipNbFirst` is the amout of indexs skiped before starting to create datas\n
        `skipNbLast` is the amout of indexs skiped at the end\n
        `chooseTestSamplesRandomly` whether the tests samles will be chosen randomly or taken at the end of train samples,\
        the order isn't kept\n
        `randomizeSamples` whether each samples will be shuffuled (not with each orther)\n
        `applieGlobalNoise` is a coef to the nose to be aplied on X_train (rand value in range [-coef, +coef])\n
        `applieSenti2Noise` is a coef to the nose to be aplied on the last serie of X_train (senti2_c normaly)  (rand value in range [-coef, +coef])\n
        `_samples_to_do` None (default) -> both, True -> inputs, False -> outputs\n"""
        if self == _Base_Model_senti2:
            raise RuntimeError("this methode cant be called from the base class")

        if (self.series_selection_inputs is None) or (self.series_selection_inputs_lagged is None) \
            or (self.series_selection_outputs is None) or (self.series_selection_outputs_delayed is None) \
            or (self.number_series_inputs is None) or (self.number_series_outputs is None) \
            or (self.nbPeriodes is None):
            raise ValueError("bad decalaration of the child instance: some None reamining:"
                                + f"({self.series_selection_inputs}, {self.series_selection_inputs_lagged}, "
                                + f"{self.series_selection_outputs}, {self.series_selection_outputs_delayed}, "
                                +f"{self.number_series_inputs}, {self.number_series_outputs}, {self.nbPeriodes})")

        if (len(self.series_selection_inputs_lagged) > 0) and (skipNbFirst < 1):
            raise ValueError(f"if there are some lagged series, skipNbFirst({skipNbFirst}) must be >= 1")
        else: series_inputs_lagged = self.series_selection_inputs_lagged

        if (len(self.series_selection_outputs_delayed) > 0) and (skipNbLast < 1):
            raise ValueError(f"if there are some delayed series, skipNbLast({skipNbLast}) must be >= 1")
        else: series_outputs_delayed = self.series_selection_outputs_delayed

        if _samples_to_do is not False:
            _missing_series_inputs = set(self.series_selection_inputs
                                         + series_inputs_lagged).difference(set(series_data.keys()))
            if len(_missing_series_inputs) > 0:
                raise KeyError("some inupts series are missing from `series_data`:\n"\
                           + f"\tmissing inputs:{_missing_series_inputs}")

        if _samples_to_do is not True:
            _missing_series_outputs = set(self.series_selection_outputs
                                          + series_outputs_delayed).difference(set(series_data.keys()))
            if len(_missing_series_outputs) > 0:
                raise KeyError("some outputs series are missing from `series_data`:\n"\
                            + f"\tmissing outputs:{_missing_series_outputs}")

        nbInputs = self.number_series_inputs
        nbOutputs = self.number_series_outputs
        nbPeriodes = self.nbPeriodes

        size_series:int
        if _samples_to_do is not False:
            if len(self.series_selection_inputs) > 0:
                size_series = len(series_data[self.series_selection_inputs[0]])
            else: size_series = len(series_data[self.series_selection_inputs_lagged[0]])
        elif _samples_to_do is not True:
            size_series = len(series_data[self.series_selection_outputs[0]])

        size_sum_train_test:int = size_series - skipNbFirst - skipNbLast - nbPeriodes
        size_train:int = int(size_sum_train_test * (1 - testSamplesProportion))
        size_test:int = size_sum_train_test - size_train

        # create the empty arrays
        X_All:_3dArray_Float = numpy.empty((size_sum_train_test, nbInputs , nbPeriodes))
        Y_All:_2dArray_Float = numpy.empty((size_sum_train_test,  nbOutputs))


        # inputs: X_train and X_test
        if _samples_to_do is not False:
            for index_input_Serie in range(nbInputs):
                if index_input_Serie < len(self.series_selection_inputs):
                    serie_name = self.series_selection_inputs[index_input_Serie]
                    lag = 0
                else:
                    serie_name = series_inputs_lagged[index_input_Serie - len(self.series_selection_inputs)]
                    lag = 1

                X_All = _copy_serie_into_X_All(
                    X_All, serieToCopy=series_data[serie_name],
                    index_shift=skipNbFirst - lag,
                    index_input_Serie=index_input_Serie,
                    nbPeriodes=nbPeriodes
                )

                # for index_train in range(size_sum_train_test):
                #     start_index_data = index_train + skipNbFirst - lag
                #     X_All[index_train, index_input_Serie, :] = \
                #         series_data[serie_name][start_index_data: start_index_data + nbPeriodes]

        # outputs: Y_train and Y_test
        if _samples_to_do is not True:
            for index_output_Serie in range(nbOutputs):
                if index_output_Serie < len(self.series_selection_outputs):
                    serie_name = self.series_selection_outputs[index_output_Serie]
                    lag = 0
                else:
                    serie_name = series_outputs_delayed[index_output_Serie - len(self.series_selection_outputs)]
                    lag = -1

                Y_All = _copy_serie_into_Y_All(
                    Y_All, serieToCopy=series_data[serie_name],
                    index_shift=skipNbFirst + nbPeriodes-1  -lag,
                    index_output_Serie=index_output_Serie,
                )

                # for index_train in range(size_sum_train_test):
                #     index_data = index_train + skipNbFirst + nbPeriodes - 1
                #     Y_All[index_train, index_output_Serie] = \
                #         series_data[serie_name][index_data]

        # randomize the samples
        if chooseTestSamplesRandomly is True:
            premutation_all = numpy.random.permutation(size_sum_train_test)
            X_All = X_All[premutation_all]
            Y_All = Y_All[premutation_all]

        self._applie_random_noise_on_datas(X_All, size_train, applieGlobalNoise, applieSenti2Noise)

        if _samples_to_do is False:
            X_train = numpy.empty((0, 0 , 0))
            X_test  = numpy.empty((0, 0 , 0))
        else:
            X_train = X_All[: size_train]
            X_test  = X_All[size_train: ]

        if _samples_to_do is True:
            Y_train = numpy.empty((0, 0))
            Y_test  = numpy.empty((0, 0))
        else:
            Y_train = Y_All[: size_train]
            Y_test  = Y_All[size_train: ]

        if (randomizeSamples is True) and (chooseTestSamplesRandomly is False):
            premutation_train = numpy.random.permutation(size_train)
            premutation_test = numpy.random.permutation(size_test)
            X_train = X_train[premutation_train]
            Y_train = Y_train[premutation_train]
            X_test  = X_test[premutation_test]
            Y_test  = Y_test[premutation_test]

        # need to reshape the X_train and X_test to (size_train, nbPeriodes, nbInputs) (AI ready)
        X_train = X_train.swapaxes(1, 2)
        X_test  = X_test.swapaxes(1, 2)

        return ((X_train, Y_train), (X_test, Y_test))

    def create_InputSample(self,
            series_data:"dict[str, _Serie_Float]",
            skipNbFirst:int, skipNbLast:int,
            )->"_3dArray_Float":
        """return the X_sample of the inputs series\n
        `series_data` is the dataset of regularized serises, where the series will be extracted (must conatain the keys)\n
        `skipNbFirst` is the amout of indexs skiped before starting to create datas\n
        `skipNbLast` is the amout of indexs skiped at the end"""
        ((X_train, _), (_, _)) = self.split_datas(
            series_data=series_data,
            skipNbFirst=skipNbFirst, skipNbLast=skipNbLast,
            testSamplesProportion=0.0,
            chooseTestSamplesRandomly=False,  randomizeSamples=False,
            _samples_to_do=True,
        )
        return X_train

    def synconize_array(self,
            full_array:"_Serie_Float", skipNbFirst:int=0, skipNbLast:int=0,
            lag:int=0, shiftNbPeriodes:bool=False)->"_Serie_Float":
        """return a copy of the array, syncronized"""
        additional_shift:int = 0
        if shiftNbPeriodes is True:
            if self.nbPeriodes is None: raise ValueError("")
            else: additional_shift += self.nbPeriodes

        return full_array[skipNbFirst + additional_shift -lag: -lag -skipNbLast]



    def create_OutputSample(self,
            series_data:"dict[str, _Serie_Float]",
            skipNbFirst:int, skipNbLast:int,
            )->"_2dArray_Float":
        """return the Y_sample of the outputs series\n
        `series_data` is the dataset of regularized serises, where the series will be extracted (must conatain the keys)\n
        `skipNbFirst` is the amout of indexs skiped before starting to create datas\n
        `skipNbLast` is the amout of indexs skiped at the end"""
        ((_, Y_train), (_, _)) = self.split_datas(
            series_data=series_data,
            skipNbFirst=skipNbFirst, skipNbLast=skipNbLast,
            testSamplesProportion=0.0,
            chooseTestSamplesRandomly=False,  randomizeSamples=False,
            _samples_to_do=False,
        )
        return Y_train

    def resize_series_as_OutputSamples(self,
            series_data:"dict[str, numpy.ndarray]|pd.DataFrame",
            skipNbFirst:int, skipNbLast:int, seriesSelection:"list[str]",
            )->"dict[str, numpy.ndarray]":
        """extract and reshape the series to the same shape as an output samples (so they are syncro with the outputs)\n
        `series_data` is the dataset of regularized serises, where the series will be extracted (must conatain the keys)\n
        `skipNbFirst` is the amout of indexs skiped before starting to create datas\n
        `skipNbLast` is the amout of indexs skiped at the end\n
        `seriesSelection` the names of the series to select\n"""
        if self == _Base_Model_senti2:
            raise RuntimeError("this methode cant be called from the base class")

        if (self.nbPeriodes is None):
            raise ValueError("bad decalaration of the child instance: some None reamining")

        _missing_series = set(seriesSelection).difference(set(series_data.keys()))
        if len(_missing_series) > 0:
            raise KeyError("some series are missing from `series_data`:\n"\
                           + f"\tmissing series:{_missing_series}")


        size_series:int = len(series_data[seriesSelection[0]])

        index_start = skipNbFirst + self.nbPeriodes - 1
        index_end = size_series - skipNbLast - 1

        if isinstance(series_data, pd.DataFrame):
            return {
                serie_name:series_data[serie_name].to_numpy()[index_start: index_end]
                for serie_name in seriesSelection
            }
        else:
            return {
                serie_name:series_data[serie_name][index_start: index_end]
                for serie_name in seriesSelection
            }


    def create_singleInput(self,
            series_data:"dict[str, _Serie_Float]",
            series_data_lagged:"None|dict[str, _Serie_Float]"=None
            )->_3dArray_Float:
        """create a unique input array of size (1, nbPeriodes, nbSeries)\n
        `series_data` is the dataset of regularized serises of the size (nbPeriodes, ),\
            where the series will be extracted (must conatain the keys)\n
        `series_data_lagged` same as `series_data` but with lagged series (the lag must alredy be aplied)\n"""
        if self == _Base_Model_senti2:
            raise RuntimeError("this methode cant be called from the base class")

        if (self.series_selection_inputs is None) or (self.number_series_inputs is None) \
            or (self.nbPeriodes is None):
            raise ValueError("bad decalaration of the child instance: some None reamining:"
                                + f"({self.series_selection_inputs}, {self.number_series_inputs},"
                                + f"{self.nbPeriodes})")

        if self.series_selection_inputs_lagged is None:
            series_inputs_lagged = []
        else:
            series_inputs_lagged = self.series_selection_inputs_lagged
            if series_data_lagged is None:
                raise KeyError("some inupts series are missing from `series_data_lagged`:\n"\
                               + f"\tmissing inputs:{series_inputs_lagged}")

            _missing_series_inputs_lagged = \
                set(series_inputs_lagged).difference(set(series_data_lagged.keys()))
            if (len(_missing_series_inputs_lagged) > 0):
                raise KeyError("some inupts series are missing from `series_data_lagged`:\n"\
                               + f"\tmissing inputs:{_missing_series_inputs_lagged}")

        # check if all necessary series are in
        _missing_series_inputs = set(self.series_selection_inputs).difference(set(series_data.keys()))
        if len(_missing_series_inputs) > 0:
            raise KeyError("some inupts series are missing from `series_data`:\n"\
                           + f"\tmissing inputs:{_missing_series_inputs}")

        nbInputs = self.number_series_inputs
        nbPeriodes = self.nbPeriodes

        X_train:_3dArray_Float = numpy.empty((1, nbInputs , nbPeriodes))

        for index_input_Serie in range(nbInputs):
            if index_input_Serie < len(self.series_selection_inputs):
                serie_name = self.series_selection_inputs[index_input_Serie]
                X_train[0, index_input_Serie, :] = series_data[serie_name]
            elif series_data_lagged is not None:
                serie_name = series_inputs_lagged[index_input_Serie - len(self.series_selection_inputs)]
                X_train[0, index_input_Serie, :] = series_data_lagged[serie_name]
            else:
                raise IndexError(f"BAD DEV ERROR: the serie n°{index_input_Serie} should be a lagged serie it fucked up somwhere")



        # need to reshape the X_train to (size_train, nbPeriodes, nbInputs) (LSTM ready)
            X_train = X_train.swapaxes(1, 2)

        return X_train

    def regularize_datas(self, dataFrame:"pd.DataFrame")->"dict[str, _Serie_Float]":

        if (self.series_selection_inputs is None) or (self.series_selection_inputs_lagged is None) \
            or (self.series_selection_outputs is None) or (self._regularize_kargs is None):
            raise ValueError("bad decalaration of the child instance: some None reamining:"
                                + f"({self.series_selection_inputs}, {self.series_selection_inputs_lagged},"
                                + f"{self.series_selection_outputs}, {self._regularize_kargs})")

        all_series_selection:"list[str]" = list(set(
            self.series_selection_inputs + self.series_selection_inputs_lagged + self.series_selection_outputs
        ))
        regularized_series:"dict[str, _Serie_Float]" = \
            AI.regularize_datas(dataFrame,  seriesSelection=all_series_selection,
                                **self._regularize_kargs)


        return regularized_series

    def show_in_out_infos(self)->None:
        print(f"VERSION = {self.VERSION}")
        if self.series_selection_inputs is not None:
            series_inputs_lagged:"list[str]"
            if self.series_selection_inputs_lagged is not None:
                series_inputs_lagged = ["LAG:"+name for name in self.series_selection_inputs_lagged]
            else: series_inputs_lagged = []
            print(f"inputs({self.number_series_inputs}): {self.series_selection_inputs + series_inputs_lagged}")
        else:print("inputs(?): Not Setted")

        if self.series_selection_outputs is not None:
            series_outputs_delayed:"list[str]"
            if self.series_selection_outputs_delayed is not None:
                series_outputs_delayed = ["DELAY:"+name for name in self.series_selection_outputs_delayed]
            else: series_outputs_delayed = []
            print(f"outputs({self.number_series_outputs}): {self.series_selection_outputs + series_outputs_delayed}")
        else:print("outputs(?): Not Setted")

    def show_network_layers_infos(self)->None:
        if isinstance(self.network, keras.Model):
            self.network.summary()
            print("\n")
            print("name: input shape, output shape")
            for layer in self.network.layers:
                print(f"{layer.name}: {layer.input_shape}, {layer.output_shape}")
        else: print(f"network isn't a keras.Model instance, but a:{type(self.network)} consider adding its support")

    def plot_model(self,
            to_file:"str|None"=None, show_shapes:bool=True, show_dtype:bool=False,
            show_layer_names:bool=True, expand_nested:bool=True,
            show_layer_activations:bool=True, dpi:int=96,
            rankdir:str="TB", layer_range:"Any|None"=None):
        DEFAULT_FILE_NAME:str = "myModel.png"
        filname:str = (to_file if to_file is not None else DEFAULT_FILE_NAME)
        image_IPython = keras.utils.plot_model(
            self.network, filname, show_shapes=show_shapes, show_dtype=show_dtype,
            show_layer_names=show_layer_names, expand_nested=expand_nested,
            show_layer_activations=show_layer_activations, dpi=dpi,
            rankdir=rankdir, layer_range=layer_range,
        )
        if to_file is None: # => del the crated image
            os.remove(filname)
        return image_IPython

    def _add_trainigHistory(self, trainingMetrics:"dict[str, list[float]]|keras.callbacks.History")->None:
        # convert keras's hist to dict of metrics
        if isinstance(trainingMetrics, keras.callbacks.History):
            trainingMetrics = trainingMetrics.history

        for metric in trainingMetrics.keys():
            if metric in self._trainingHistory:
                self._trainingHistory[metric] += trainingMetrics[metric]
            else: self._trainingHistory[metric] = trainingMetrics[metric]

    def _add_trainigMetricValue(self, metricName, metricvalue)->None:
        if metricName in self._trainingHistory:
            self._trainingHistory[metricName] += [metricvalue]
        else: self._trainingHistory[metricName] = [metricvalue]


    def get_all_traing_history(self)->"dict[str, list[float]]":
        """WARNING: they migth nor be the same length if baddly added the metrics"""
        return self._trainingHistory

    def plot_all_hist(self,
            metrics:"list[str]|None"=None, emaSmoothing:"None|int"=None,
            emaFilters:"list[str]|None"=None, labelsSufixs:str="")->None:
        """plot all the selected metrics and also plot an ema smoothed curve\n
        `metrics` are the metrics to be plotted, None mean all available\n
        `emaSmoothing` is the nbPeriodes of ema smoothing, None mean no ema curves, \
        when enabled (!= None), it plot new curves, it dont replace\n
        `emaFilters` sequences of str that the metric must contain in order to be ema\n
        """
        if emaFilters is None: emaFilters = []

        metricsSelection = list(self._trainingHistory.keys()) if metrics is None else metrics
        for metric in metricsSelection:
            label = f"{metric} {labelsSufixs}"
            metricArray:"_Serie_Float" = numpy.array(self._trainingHistory[metric])
            plt.plot(metricArray, label=label)
            if (emaSmoothing != None) and all((filt in metric) for filt in emaFilters):
                plt.plot(
                    ema_numpy_serie_normalDir(metricArray, emaSmoothing),
                    label=f"{label} ema {emaSmoothing}", linestyle=":" # : => dotted
                )
        plt.legend()
        plt.show()

    def plot_metrics(self, plotCfg:"None|dict[str, dict[str, Any]]"=None, merge:bool=True)->None:
        """plot the metrics by using a config like:\n
        plotCfg = None (details after) | {
            "globalCfg":{
                "nbSubPlots":int|(default 1) # verticals
                "sharex":bool|(default on)
                "figName":str|(default "metrics history")
                "legend":bool|(default on)
                "tight_layout":bool|(default off)
                "hlines":[(val:float, subPlot:int),   ...]|(default empty)
                "yLims":[(yTop:float, yBot:float, subPlot:int),   ...]|(default empty)
            }
            "metricName":{
                "subPlot":int|(default first)
                "emaSmoothing":int|(default off(ie None))
                "label":str|(default metricName)
                "color":str|(default auto)
            }, ... one for all metrics\\
        }
        when None: try to get the one form the model, if none use the default cfg
        """
        # use the right config
        if (plotCfg is None):
            plotCfg = self._plotCfg

        # global config
        nbSubPlots:int = 1
        sharex:bool = True
        figName:"str" = "metrics history"
        legend:bool = True
        tight_layout:bool=False
        hlines:"list[tuple[float, int]]" = []
        yLims:"list[tuple[float, float, int]]" = []
        if (self._plotCfg is not None) and (merge is True):
            nbSubPlots = self._plotCfg["globalCfg"].get("nbSubPlots", nbSubPlots)
            sharex = self._plotCfg["globalCfg"].get("sharex", sharex)
            figName = self._plotCfg["globalCfg"].get("figName", figName)
            legend = self._plotCfg["globalCfg"].get("legend", legend)
            hlines = self._plotCfg["globalCfg"].get("hlines", hlines)
            tight_layout = self._plotCfg["globalCfg"].get("tight_layout", tight_layout)
            yLims = self._plotCfg["globalCfg"].get("yLims", yLims)
        if (plotCfg is not None) and ("globalCfg" in plotCfg): #try to update them
            nbSubPlots = plotCfg["globalCfg"].get("nbSubPlots", nbSubPlots)
            sharex = plotCfg["globalCfg"].get("sharex", sharex)
            figName = plotCfg["globalCfg"].get("figName", figName)
            legend = plotCfg["globalCfg"].get("legend", legend)
            hlines = plotCfg["globalCfg"].get("hlines", hlines)
            tight_layout = plotCfg["globalCfg"].get("tight_layout", tight_layout)
            yLims = plotCfg["globalCfg"].get("yLims", yLims)

        # metrics config
        metricsConfig:"dict[str, dict[str, Any]]" = {}
        if self._plotCfg is not None:
            metricsConfig = {
                metricName:value for (metricName, value) in self._plotCfg.items()
                if (metricName != "globalCfg") and (metricName in self._trainingHistory)
            }
        if plotCfg is not None:
            for (metricName, value) in plotCfg.items():
                if (metricName != "globalCfg") and (metricName in self._trainingHistory):
                    if (merge is False) or (metricName not in metricsConfig):
                        metricsConfig[metricName] = value
                    else : # -> merge
                        for paramName, paramValue in value.items():
                            metricsConfig[metricName][paramName] = paramValue

        elif (self._plotCfg is None): # didnt used the cfg from the class
            for metricName in self._trainingHistory:
                if metricName not in metricsConfig:
                    metricsConfig[metricName] = {}


        # update the metrics config with default when no parameter
        for (metricName, metricCfg) in metricsConfig.items():
            metricCfg["subPlot"] = metricCfg.get("subPlot", 0)
            metricCfg["emaSmoothing"] = metricCfg.get("emaSmoothing", None)
            metricCfg["label"] = metricCfg.get("label", metricName)
            metricCfg["color"] = metricCfg.get("color", None)

        # create (if nessesary) the figure and save it
        axes:"list[plt.Axes]"
        if (plt.fignum_exists(figName) is True) and (figName in PLOT_FIGS.keys()):
            fig = PLOT_FIGS[figName]
            axes = fig.axes
        else:
            fig, axes = plt.subplots(nbSubPlots, sharex=sharex, num=figName)
            PLOT_FIGS[figName] = fig

        # plot everything
        longestMetric:int = max([len(self._trainingHistory[metricName]) for metricName in metricsConfig.keys()])
        for (value, subplot) in hlines:
            plotAxe:"plt.Axes" = axes[subplot]
            plotAxe.hlines([value], -1.0, longestMetric+1.0, linestyles=':', colors=["black"])

        for (yTop, yBot, subplot) in yLims:
            plotAxe:"plt.Axes" = axes[subplot]
            plotAxe.set_ylim(bottom=yBot, top=yTop)


        for (metricName, metricCfg) in metricsConfig.items():
            metricArray:"_Serie_Float" = numpy.array(self._trainingHistory[metricName])
            plotAxe:"plt.Axes" = axes[metricCfg["subPlot"]]
            plotAxe.plot(metricArray, color=metricCfg["color"], label=metricCfg["label"])
            emaSmoothing:"int|None" = metricCfg["emaSmoothing"]
            if emaSmoothing is not None:
                plotAxe.plot(
                    ema_numpy_serie_normalDir(metricArray, emaSmoothing),
                    color=metricCfg["color"], linestyle=":", # ':' => dotted
                    label=f"{metricCfg['label']} ema{emaSmoothing}"
                )
        if tight_layout is True:
            fig.set_tight_layout(tight_layout)

        if len(metricsConfig) > 0:
            if legend is True:
                for ax in axes:
                    ax.legend(loc='upper left')
            fig.show()
        else: print("no metrics to plot")

    def get_currentResume(self,
            cfg:"None|list[tuple[str, dict[str, Any]]]"=None,
            nbPeriodes:int=8, newLineEvery:"int|None"=None)->str:
        """get a str with the resumé of the metrics over up to the N last periodes :\n
        cfg = {
            "sort":{"list"}
            "metricName":{
                "alias":int|(default first)
                "func":Callable[float->str]|(default v->f"{v :.4f}")
            }, ... # one for all metrics to include\\
            if no metrics given use all availables
        } |None (details ->)
        when None: try to get the one form the model, if none use the default cfg
        """
        if cfg is None:
            cfg = self._resumeCfg
        if cfg is None:
            cfg = [(metricName, {}) for metricName in self._trainingHistory.keys()]

        out = StringIO()
        out.write("(-> ")
        nbMetrics_line:"int|None" = (-1 if newLineEvery is None else 1)
        for index, (metricName, metricCfg) in enumerate(cfg):
            if metricName not in self._trainingHistory:
                continue # improve compatibility
            alias:str = metricCfg.get("alias", metricName)
            func:"Callable[[float], str]" = \
                metricCfg.get("func", lambda v:f"{v :.4f}")
            alias:str = metricCfg.get("alias", metricName)
            metricValue:float = ema_numpy_serie_normalDir(numpy.array(
                self._trainingHistory[metricName][-nbPeriodes: ]
            ), nbPeriodes//2)[-1]
            out.write(f"{func(metricValue)} {alias}, ")
            if nbMetrics_line == newLineEvery:
                if index+1 < len(cfg): # dont newline when it was last metric
                    out.write("\n")
                nbMetrics_line = 1
            elif newLineEvery is not None:
                nbMetrics_line += 1
        out.write(")")
        return out.getvalue()

    def get_learning_rate(self)->float:
        if isinstance(self.network, keras.Model):
            if ("optimizer" in dir(self.network)) and ("learning_rate" in dir(self.network.optimizer)):
                return keras.backend.get_value(self.network.optimizer.learning_rate)
            else: raise AttributeError("the AI's model dont have a .optimizer.learning_rate")
            # return float(self.network.optimizer.learning_rate.numpy())
        raise TypeError(f"the AI's model isn't of type 'keras.Model' but {type(self.network)}")

    def set_learning_rate(self, value:float)->None:
        if isinstance(self.network, keras.Model):
            if ("optimizer" in dir(self.network)) and ("learning_rate" in dir(self.network.optimizer)):
                return keras.backend.set_value(self.network.optimizer.learning_rate, value)
            else: raise AttributeError("the AI's model dont have a .optimizer.learning_rate")
            # return float(self.network.optimizer.learning_rate.numpy())
        raise TypeError(f"the AI's model isn't of type 'keras.Model' but {type(self.network)}")


    def add_comment(self, message:str):
        self.comments.append(message)

    def get_concat_comments(self)->str:
        return "\n".join(self.comments)

    def __call__(self, *args, **kargs):
        """equivalent to self.network(args, kargs)"""
        if self.network is None:
            raise RuntimeError("this methode cant be called from the base class")
        else: return self.network(*args, **kargs)


def get_custom_loss_funcs(funcName:str):
    def create_custom_loss_func_1(A:int, N:int):
        def custom_loss_func_1(y_true, y_pred)->"tf.Tensor":
            """(K+N*(2*y_true-1)**A) * (y_true - y_pred)**2\
            of derivative: (N*(2*y_true-1)**A) * 2(y_true - y_pred)"""
            squared_difference = tf.square(y_true - y_pred)
            K = (A - N + 1) / (A + 1)
            accentuation_coef = K + N * tf.pow((2*y_true - 1), A)
            return tf.reduce_mean(accentuation_coef * squared_difference, axis=-1)  # Note the `axis=-1`
        return custom_loss_func_1
    if funcName == "custom_loss_func_1":
        return create_custom_loss_func_1
    else:
        raise KeyError(f"the funcName:{funcName} isn't associated to any loss functions")

def load_model(filename:str, directory:"Path|None"=None)->"_Model_Senti2":
        """load the model from `directory`/`filename`.h5 \
            and its parameters from `directory`/`filename`.json\n
        `directory` can end with '/', by default to the standard dir\n"""
        return _Base_Model_senti2.load(
            filename=filename,
            directory=directory
        )

try: from . import _Model_Senti2 # just for typing
except: pass
