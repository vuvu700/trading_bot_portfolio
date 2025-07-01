import pickle
import tensorflow as tf

from save_formats import AsJson_KerasConfig

from holo.__typing import (
    Self, JsonTypeAlias, assertIsinstance, Union
)

class KerasConfig():
    __slots__ = ("tf_v1_compat", "memoryGrowth",
                 "toRevert", "__revertToConfig", )
    
    __MEMORY_GROWTH:bool = False
    __TF_V1_COMPAT:bool = False
    
    def __init__(self, 
            tf_v1_compat:"bool|None"=None, 
            memoryGrowth:"bool|None"=None,
            toRevert:bool=False)->None:
        self.tf_v1_compat:"bool|None" = tf_v1_compat
        self.memoryGrowth:"bool|None" = memoryGrowth
        self.toRevert:bool = toRevert
        self.__revertToConfig:"KerasConfig|None" = None

    def setConfig(self)->None:
        self.__revertToConfig = KerasConfig.getCurrentConfig()
        
        # Reliable Mode
        if (self.tf_v1_compat == KerasConfig.__TF_V1_COMPAT) \
                or (self.tf_v1_compat is None):
            pass # => alredy done
        elif self.tf_v1_compat is True:
            # => enable it
            tf.compat.v1.disable_v2_behavior()
            KerasConfig.__TF_V1_COMPAT = True
        else: # => self.reliableMode is False => disable it
            tf.compat.v1.enable_v2_behavior()
            KerasConfig.__TF_V1_COMPAT = False
        
        # Memory Gorwth
        if (self.memoryGrowth == KerasConfig.__MEMORY_GROWTH) \
                or (self.memoryGrowth is None):
            pass # => alredy done
        else: # => change it 
            print(f"[DEBUG] -> setting to memory growth to: {self.memoryGrowth}")
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, self.memoryGrowth)
            KerasConfig.__MEMORY_GROWTH = self.memoryGrowth
    
    def revert(self)->None:
        if self.toRevert is False:
            return # => nothing to do
        # revert to the config before self was setted
        if self.__revertToConfig is None:
            return # => no config to revert to
        self.__revertToConfig.setConfig()
        self.__revertToConfig = None # avoid re reverting when calling again
    
    @classmethod
    def getCurrentConfig(cls)->"KerasConfig":
        return KerasConfig(
            tf_v1_compat=cls.__TF_V1_COMPAT,
            memoryGrowth=cls.__MEMORY_GROWTH,
            toRevert=False)
    
    @classmethod
    def getCurrentReliableMode(cls)->bool:
        return KerasConfig.__TF_V1_COMPAT
    
    @classmethod
    def getCurrentMemoryGrowth(cls)->bool:
        return KerasConfig.__MEMORY_GROWTH
        
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(reliableMode={self.tf_v1_compat}, memoryGrowth={self.memoryGrowth}, toRevert={self.toRevert})"
    
    def __call__(self)->None:
        self.setConfig()
        
    def __enter__(self)->Self:
        self.setConfig()
        return self

    def __exit__(self, *_, **__)->None:
        self.revert()
    
    def toJson(self)->"AsJson_KerasConfig":
        return AsJson_KerasConfig(
            cls=self.__class__.__name__,
            reliableMode=self.tf_v1_compat,
            memoryGrowth=self.memoryGrowth,
            toRevert=self.toRevert)

    @classmethod
    def fromJson(cls, datas:"AsJson_KerasConfig")->"KerasConfig":
        assert datas["cls"] == cls.__name__
        kerasConfig = KerasConfig.__new__(cls)
        KerasConfig.__init__(
            self=kerasConfig,
            tf_v1_compat=datas["reliableMode"],
            memoryGrowth=datas["memoryGrowth"],
            toRevert=datas["toRevert"])
        return kerasConfig