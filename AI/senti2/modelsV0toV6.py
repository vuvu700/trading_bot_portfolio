if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

from .baseModel import _Base_Model_senti2

import keras
import keras.layers
import keras.optimizers
import keras.callbacks

from typing import Any

from AI.transformersLayer import _Transformers_GPT

from holo.types_ext import _3dArray_Float, _2dArray_Float





class Model_senti2_V0(_Base_Model_senti2):
    VERSION = "V0"
    # set the model fixed parameters
    nbPeriodes:int = 15
    series_selection_inputs:"list[str]" = [
        "ERd", "DVRTX2", "RSI",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "DVWMA_n", "MFI", "ADX_e",
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"

    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Sequential"
        if self.network is None:
            self.network = keras.Sequential([
                keras.layers.LSTM(
                    units=self.number_series_inputs,
                    input_shape=(self.nbPeriodes, self.number_series_inputs),
                    return_sequences=True,
                    name='LSTM'),
                keras.layers.Flatten(),
                keras.layers.Dense(self.number_series_outputs, name='Output'),
            ])
            # or make some paralels LSTM layers: nbInputs * LSTM(1, input_shape=(nbPeriodes, 1), return_sequences=True)
            # then concatenate them to a dense layer ...

        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.RMSprop(
                learning_rate=0.001,
                rho=0.9,
                momentum=0.,
                epsilon=1e-08,
                decay=0.0
            )

        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        # A aprofondire
        """self.checkpointer = keras.callbacks.ModelCheckpoint(
            CHECKPOINTS_FILEPATH
        )"""



    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        # ajouter un systeme pour restart le train a partir des checkpoints
        # callbacks=[self.checkpointer, self.early_stopping]
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                callbacks=[self.early_stopping],
                shuffle_train=shuffle_datas
            )
        )



#######
#######
#######
#######
#######
#######
#######
#######



class Model_senti2_V1(_Base_Model_senti2):
    """same LSTM as V0 but more Dense layers and using Adam optimizer and 3x slower lr"""

    VERSION:str = "V1"
    # set the model fixed parameters
    nbPeriodes:int = 15
    series_selection_inputs:"list[str]" = [
        "ERd", "DVRTX2", "RSI",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "DVWMA_n", "MFI", "ADX_e",
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"

    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Sequential"
        if self.network is None:
            self.network = keras.Sequential([
                keras.layers.LSTM(
                    units=self.number_series_inputs,
                    input_shape=(self.nbPeriodes, self.number_series_inputs),
                    return_sequences=True,
                    name='LSTM'),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_0'),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_1'),
                keras.layers.Dense(
                    self.number_series_outputs,
                    activation="sigmoid",
                    name='Output'),
            ])

        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=0.0003,
            )



        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        # A aprofondire
        """self.checkpointer = keras.callbacks.ModelCheckpoint(
            CHECKPOINTS_FILEPATH
        )"""



    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        # ajouter un systeme pour restart le train a partir des checkpoints
        # callbacks=[self.checkpointer, self.early_stopping]
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                callbacks=[self.early_stopping],
                shuffle_train=shuffle_datas
            )
        )


#######
#######
#######
#######
#######
#######
#######
#######

class Model_senti2_V2(_Base_Model_senti2):
    """same as V1 but more data as input, different nbPeriodes and 1.5x faster lr"""

    VERSION = "V2"
    # set the model fixed parameters
    nbPeriodes:int = 35
    series_selection_inputs:"list[str]" = [
        "ER", "ERd", "TR_n", "DVRTX2", "DX", "RSI", "BBW_e",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "DVWMA_n", "MFI", "ATR_en", "ADX_e",
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"

    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Sequential"
        if self.network is None:
                self.network = keras.Sequential([
                keras.layers.LSTM(
                    units=self.number_series_inputs ** 2,
                    input_shape=(self.nbPeriodes, self.number_series_inputs),
                    return_sequences=True,
                    name='LSTM'),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_0'),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_1'),
                keras.layers.Dense(
                    self.number_series_outputs,
                    activation="sigmoid",
                    name='Output'),
            ])

        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=0.0003,
            )



        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        # A aprofondire
        """self.checkpointer = keras.callbacks.ModelCheckpoint(
            CHECKPOINTS_FILEPATH
        )"""



    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        # ajouter un systeme pour restart le train a partir des checkpoints
        # callbacks=[self.checkpointer, self.early_stopping]
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                callbacks=[self.early_stopping],
                shuffle_train=shuffle_datas
            )
        )



#######
#######
#######
#######
#######
#######
#######
#######


class Model_senti2_V3(_Base_Model_senti2):
    """very similar to V2 but it has a laged serie of senti2_c"""

    VERSION = "V3"
    # set the model fixed parameters
    nbPeriodes:int = 35
    series_selection_inputs:"list[str]" = [
        "ER", "ERd", "TR_n", "DVRTX2", "DX", "RSI", "BBW_e",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "DVWMA_n", "MFI", "ATR_en", "ADX_e",
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"

    series_selection_inputs_lagged:"list[str]" = [
        "senti2_c"
    ]
    "the names of the series (lagged by 1) that will be selected as inputs"


    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs) + len(series_selection_inputs_lagged)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Sequential"
        if self.network is None:
            self.network = keras.Sequential([
                keras.layers.LSTM(
                    units=self.number_series_inputs ** 2,
                    input_shape=(self.nbPeriodes, self.number_series_inputs),
                    return_sequences=True,
                    name='LSTM'),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_0'),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_1'),
                keras.layers.Dense(
                    self.number_series_outputs,
                    activation="hard_sigmoid",
                    name='Output'),
            ])

        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=0.0001,
            )



        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        # A aprofondire
        """self.checkpointer = keras.callbacks.ModelCheckpoint(
            CHECKPOINTS_FILEPATH
        )"""



    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                callbacks=[self.early_stopping],
                shuffle_train=shuffle_datas
            )
        )



#######
#######
#######
#######
#######
#######
#######
#######


class Model_senti2_V4(_Base_Model_senti2):
    """similar to V3, with new series: Close_d, ema_d  and a 2nd LSTM layer"""

    VERSION = "V4"
    # set the model fixed parameters
    nbPeriodes:int = 35
    series_selection_inputs:"list[str]" = [
        "ER", "ERd", "TR_n", "DVRTX2", "DX", "RSI", "BBW_e",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "DVWMA_n", "MFI", "ATR_en", "ADX_e",  "CLOSE_n", "EMA_n",
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"
    series_selection_inputs_lagged:"list[str]" = [
        "senti2_c"
    ]
    "the names of the series (lagged by 1) that will be selected as inputs (THE ORDER IS IMPORTANT !)"


    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs) + len(series_selection_inputs_lagged)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {"rescale":True, "preferReRangePlus":True}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Sequential"
        if self.network is None:
            self.network = keras.Sequential([
                keras.layers.LSTM(
                    units=250,
                    input_shape=(self.nbPeriodes, self.number_series_inputs),
                    return_sequences=True,
                    name='LSTM_in'),
                keras.layers.LSTM(
                    units=self.number_series_inputs,
                    return_sequences=True,
                    name='LSTM_process'),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_0'),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    self.number_series_inputs,
                    activation="softplus",
                    name='process_layer_1'),
                keras.layers.Dense(
                    self.number_series_outputs,
                    activation="hard_sigmoid",
                    name='Output'),
            ])

        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=0.0001,
            )



        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                callbacks=[self.early_stopping],
                shuffle_train=shuffle_datas
            )
        )

#######
#######
#######
#######
#######
#######
#######
#######

class Model_senti2_V5(_Base_Model_senti2):
    """similar to V4, with a lot of new series"""

    VERSION = "V5"
    # set the model fixed parameters
    nbPeriodes:int = 35
    series_selection_inputs:"list[str]" = [
        "CLOSE_n", "EMA_n", "RSI", "DVWMA_n", "MFI", "GROW", "CHOP", "CCI",
        "CMF", "CV", "CMF", "W%R", "CMO", "MI", "MIA", "ERI", "MAO",
        "TR_n", "ATR_en",       "ER", "ERd",      "DVRTX", "DVRTX2",
        "DI+", "DI-", "DX", "ADX_e",      "SD_n", "BB%_e", "BBW_e",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "SCAL", "ANGL2", "ANGL", "ANGL_n", "SHARP", "SHARP_n", "FLAT", "FLAT_n",
        "VIN", "VIP",      "AOOU", "AOOD",      "DCW_n", "DC%",
        "SAFU_n", "SAFL_n", "SAFW_n", "SAFD", "SAF%",
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"
    series_selection_inputs_lagged:"list[str]" = [
        "senti2_c"
    ]
    "the names of the series (lagged by 1) that will be selected as inputs (THE ORDER IS IMPORTANT !)"


    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs) + len(series_selection_inputs_lagged)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {"rescale":True, "preferReRangePlus":True}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Sequential"
        if self.network is None:
            self.network = keras.Sequential([
                keras.layers.LSTM(
                    units = 400,
                    input_shape=(self.nbPeriodes, self.number_series_inputs),
                    return_sequences=True,  name='LSTM_in'),
                keras.layers.Dropout(0.1),
                keras.layers.LSTM(
                    units = 75,
                    return_sequences=True,  name='LSTM_process'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(
                    units = self.number_series_inputs,
                    activation="softplus", name='process_layer_0'),
                keras.layers.Dropout(0.1),
                keras.layers.Flatten(),
                keras.layers.Dense(
                    units = self.number_series_inputs,
                    activation="softplus", name='process_layer_1'),
                keras.layers.Dense(
                    units = self.number_series_outputs,
                    activation="hard_sigmoid", name='Output'),
            ])

        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=0.0001,
            )



        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                callbacks=[self.early_stopping],
                shuffle_train=shuffle_datas
            )
        )

#######
#######
#######
#######
#######
#######
#######
#######






class Model_senti2_V6(_Base_Model_senti2):
    """similar to V5, with some TransformerBlocks (GPT like), \
    the LSTM is now Bidirectional and all the series are from T-1 prdicting T+0"""

    VERSION = "V6"
    # set the model fixed parameters
    nbPeriodes:int = 35
    series_selection_inputs:"list[str]" = [
    ]
    "the names of the series that will be selected as inputs (THE ORDER IS IMPORTANT !)"
    series_selection_inputs_lagged:"list[str]" = [
        "CLOSE_n", "EMA_n", "RSI", "DVWMA_n", "MFI", "GROW", "CHOP", "CCI",
        "CMF", "CV", "CMF", "W%R", "CMO", "MI", "MIA", "ERI", "MAO",
        "TR_n", "ATR_en",       "ER", "ERd",      "DVRTX", "DVRTX2",
        "DI+", "DI-", "DX", "ADX_e",      "SD_n", "BB%_e", "BBW_e",
        "DER1_en", "DER2_en", "MACD_en", "MACDh_en",
        "SCAL", "ANGL2", "ANGL", "ANGL_n", "SHARP", "SHARP_n", "FLAT", "FLAT_n",
        "VIN", "VIP",      "AOOU", "AOOD",      "DCW_n", "DC%",
        "SAFU_n", "SAFL_n", "SAFW_n", "SAFD", "SAF%",

        "senti2_c"
    ]
    "the names of the series (lagged by 1) that will be selected as inputs (THE ORDER IS IMPORTANT !)"


    series_selection_outputs:"list[str]" = [
        "senti2_c",
    ]
    "the names of the series that will be selected as outputs (THE ORDER IS IMPORTANT !)"

    number_series_inputs:int = len(series_selection_inputs) + len(series_selection_inputs_lagged)
    number_series_outputs:int = len(series_selection_outputs)

    _regularize_kargs:"dict[str, Any]" = {"rescale":True, "preferReRangePlus":True}

    def __init__(self, _network:"None|keras.Sequential"=None)->None:
        super().__init__(_network=_network)
        self._popNoneSeriesSelection()

        # create the network, otimizer, callbacks, etc.
        self.network:"keras.Model"
        if self.network is None:
            networkDroprate:float = 0.2

            Inputs = keras.Input(
                shape=(self.nbPeriodes, self.number_series_inputs),
                dtype="float32"
            )

            X_network = _Transformers_GPT(
                nb_trasformers=8, nb_heads=8, nbDims_heads=32,
                feedForward_size=4*self.number_series_inputs,
                embeded_dims=self.number_series_inputs,
                dropoutRate=networkDroprate, name="Transformers_GPT"
            )(Inputs)

            X_network = keras.layers.Bidirectional(keras.layers.LSTM(
                units = 400,
                input_shape=(self.nbPeriodes, self.number_series_inputs),
                return_sequences=True,  name='LSTM_main_layer'
            ))(X_network)
            X_network = keras.layers.Dropout(networkDroprate)(X_network)

            """X_network = keras.layers.LSTM(
                units = 75,
                return_sequences=True,  name='LSTM_process'
            )(X_network)
            X_network = keras.layers.Dropout(networkDroprate)(X_network)"""

            X_network = keras.layers.Dense(
                units = self.number_series_inputs,
                activation="relu", name='process_layer_0'
            )(X_network)
            X_network = keras.layers.Flatten()(X_network)
            X_network = keras.layers.Dropout(networkDroprate)(X_network)

            X_network = keras.layers.Dense(
                units = self.number_series_inputs,
                activation="relu", name='process_layer_1'
            )(X_network)

            X_network = keras.layers.Dense(
                units = self.number_series_outputs,
                activation="hard_sigmoid", name='Output'
            )(X_network)

            self.network = keras.Model(
                inputs=Inputs,
                outputs=X_network
            )


        self.optimizer:"keras.optimizers.Optimizer"
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=0.0001,
            )



    def compile(self)->None:
        self.network.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
        )

    def train(self,
            X_train:_3dArray_Float, Y_train:_2dArray_Float,
            X_test:_3dArray_Float, Y_test:_2dArray_Float,
            nbEpoches:int, batchSize:int, verbose:"str|int"=1,
            shuffle_datas:bool=False)->None:
        self._add_trainigHistory(
            self._base_train(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                nbEpoches=nbEpoches, batchSize=batchSize, verbose=verbose,
                shuffle_train=shuffle_datas
            )
        )

