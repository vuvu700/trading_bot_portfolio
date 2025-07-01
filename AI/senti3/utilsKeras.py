import keras
import keras.optimizers

from pathlib import Path


from .keras_config import KerasConfig

from holo import assertIsinstance
from holo.__typing import (
    Literal, Any,
)

_Extention = Literal["tf", "h5"]

def get_saveModel_fullPath(filename:str, directory:Path)->"tuple[Path, _Extention]":
    extention: "_Extention"
    if KerasConfig.getCurrentReliableMode() is True:
        extention = "h5"
    else: extention = "tf"
        
    if filename.startswith(extention) is False:
        filename = f"{filename}.{extention}"
    fullPath:Path = directory.joinpath(filename)
    return (fullPath, extention)

def save_keras_model(model:keras.Model, filename:str, directory:Path)->Path:
    (fullPath, extention) = get_saveModel_fullPath(
        filename=filename, directory=directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    model.save(
        filepath=fullPath,
        include_optimizer=True, overwrite=True,
        save_format=extention,
    )
    return fullPath
    
def load_keras_model(saveFile_fullPath:Path, customObjects:"dict[str, Any]")->keras.Model:
    return assertIsinstance(keras.Model,
        keras.models.load_model(
            filepath=saveFile_fullPath, 
            compile=True, custom_objects=customObjects))
    
