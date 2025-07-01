if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

from . import (
    baseModel, modelsV0toV6, modelsV7,
    utilsFuncs, reinforcement_learning_funcs,
)

from .baseModel import (
    _Base_Model_senti2, enable_reliable_mode, load_model,
    get_RELIABILITY_MODE, set_RELIABILITY_MODE,
    get_USE_MULTIPROCESSING, set_USE_MULTIPROCESSING,
    AI_SAVE_SENTI2_DIRECTORY,
)
from .modelsV0toV6 import (
    Model_senti2_V0, Model_senti2_V1, Model_senti2_V2, Model_senti2_V3,
    Model_senti2_V4, Model_senti2_V5, Model_senti2_V6
)
from .modelsV7 import Model_senti2_V7, Model_senti2_V7_GPT, _calc_X_train_by_iteration


import tensorflow as tf
from typing import Union
import warnings
import importlib

from holo import print_exception



_Model_Senti2 = Union[
    Model_senti2_V0, Model_senti2_V1, Model_senti2_V2, Model_senti2_V3,
    Model_senti2_V4, Model_senti2_V5, Model_senti2_V6, Model_senti2_V7,
    Model_senti2_V7_GPT,
]


warnings.filterwarnings("ignore", category=Warning, module="backtesting")
warnings.filterwarnings("ignore", category=Warning, module="backtests")
warnings.filterwarnings("ignore", category=UserWarning, module="keras", lineno=2332)
warnings.filterwarnings("ignore", category=UserWarning, module="keras", lineno=107)
warnings.filterwarnings("ignore", category=UserWarning, module="keras", lineno=2356)


def ENABLE_MEMORY_GROWTH():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as err:
            print_exception(err)


def RELOAD_SUB_MODULES()->None:
    from . import (
        baseModel, modelsV0toV6, modelsV7, 
        utilsFuncs, reinforcement_learning_funcs,
    )
    reinforcement_learning_funcs.RELOAD_SUB_MODULES()
    
    importlib.reload(baseModel)
    importlib.reload(modelsV0toV6)
    importlib.reload(modelsV7)
    importlib.reload(reinforcement_learning_funcs)
    importlib.reload(utilsFuncs)
    from . import (
        baseModel, modelsV0toV6, modelsV7, 
        utilsFuncs, reinforcement_learning_funcs,
    )
