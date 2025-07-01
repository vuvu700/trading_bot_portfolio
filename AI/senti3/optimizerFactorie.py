
import tensorflow as tf
import keras
import keras.layers
import keras.engine.keras_tensor
import keras.optimizers

from save_formats import AsJson_ModelFactoryConfig, AsJson_OptimizerFactoryConfig

from holo.__typing import (
    Any, JsonTypeAlias, TypedDict, Literal, NotRequired, overload,
    Callable, cast, Self, Union, assertIsinstance,
)

keras_Optimizer = Union[keras.optimizers.Optimizer, keras.optimizers.base_optimizer_v2.OptimizerV2]



### optimizer Factory config

_OptimizerName = Literal["Adam", "SGD"]

class _OptimizerFactoryConfig():
    _OPTIMIZERS: "dict[_OptimizerName, Callable[..., keras_Optimizer]]" = {
        "Adam": tf.keras.optimizers.Adam,
        "SGD": tf.keras.optimizers.SGD}
    SUB_CLASSES: "dict[str, type[_OptimizerFactoryConfig]]" = {}
    
    def __init_subclass__(cls) -> None:
        _OptimizerFactoryConfig.SUB_CLASSES[cls.__name__] = cls
    
    def __init__(self, optimizerName:"_OptimizerName", optimizerKwargs:"dict[str, Any]") -> None:
        self.optimizerName:"_OptimizerName" = optimizerName
        self.optimizerKwargs:"dict[str, Any]" = optimizerKwargs

    def createOptimizer(self)->keras_Optimizer:
        optimizerClass:"Callable[..., keras_Optimizer]" = \
            self._OPTIMIZERS[self.optimizerName]
        return optimizerClass(**self.optimizerKwargs)

    def toJson(self)->"AsJson_OptimizerFactoryConfig":
        return AsJson_OptimizerFactoryConfig(
            cls=self.__class__.__name__,
            optimizerName=self.optimizerName,
            optimizerKwargs=self.optimizerKwargs)
    
    @classmethod
    def fromJson(cls, datas:"AsJson_OptimizerFactoryConfig")->"_OptimizerFactoryConfig":
        cls = _OptimizerFactoryConfig.SUB_CLASSES[datas["cls"]]
        optimizerConfig = _OptimizerFactoryConfig.__new__(cls)
        _OptimizerFactoryConfig.__init__(
            self=optimizerConfig,
            optimizerName=datas["optimizerName"],
            optimizerKwargs=datas["optimizerKwargs"])
        return optimizerConfig



### Adam

class OptimizerFactoryConfig_Adam(_OptimizerFactoryConfig):
    optimizerName : "Literal['Adam']"
    optimizerKwargs: "AdamKwargs"
    def __init__(self, optimizerKwargs:"AdamKwargs") -> None:
        super().__init__(optimizerName="Adam", optimizerKwargs=dict(optimizerKwargs))



class AdamKwargs(TypedDict):
    learning_rate: "NotRequired[float]"
    beta_1: "NotRequired[float]"
    beta_2: "NotRequired[float]"
    epsilon: "NotRequired[float]"
    amsgrad: "NotRequired[bool]"
    name: "NotRequired[str]"
    clipvalue: "NotRequired[float]"
    clipnorm: "NotRequired[float]"
    global_clipnorm: "NotRequired[float]"


#### RMS -> https://keras.io/api/optimizers/rmsprop/

class OptimizerFactoryConfig_SGD(_OptimizerFactoryConfig):
    optimizerName : "Literal['SGD']"
    optimizerKwargs: "SgdKwargs"
    def __init__(self, optimizerKwargs:"SgdKwargs") -> None:
        super().__init__(optimizerName="SGD", optimizerKwargs=dict(optimizerKwargs))

class SgdKwargs(TypedDict):
    learning_rate: "NotRequired[float]" # 0.001
    momentum: "NotRequired[float]" # 0.0
    nesterov: "NotRequired[bool]" # False
    weight_decay: "NotRequired[float|None]" # None
    clipnorm: "NotRequired[float|None]" # None
    clipvalue: "NotRequired[float|None]" # None
    global_clipnorm: "NotRequired[float|None]" # None