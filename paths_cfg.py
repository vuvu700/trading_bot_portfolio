import os, os.path, sys
from pathlib import Path



def joinAndEnsure(src:"str|Path", *other:"str|Path")->Path:
    new = (src if isinstance(src, Path) else Path(src))
    for nextDir in other:
        new = new.joinpath(nextDir)
        if new.exists() is False:
            os.mkdir(new)
    assert new.exists(), f"the path: {new.as_posix()!r} don't exist (couldn't be crated)"
    return new


CURRENT_DIRECTORY = Path(__file__).parent

### logs
LOGS_DIRECTORY = joinAndEnsure(CURRENT_DIRECTORY, "logs")


### to save the datas
SAVE_DATAS_DIRECTORY = Path("X:/TradingBot_datas/") # save datas on another dir
if SAVE_DATAS_DIRECTORY.exists() is False: 
    # if not available, save them here
    SAVE_DATAS_DIRECTORY = joinAndEnsure(CURRENT_DIRECTORY, "tradingBot_saves")
AUTOSAVE_MESSAGES_FILES_DIRECTORY = joinAndEnsure(SAVE_DATAS_DIRECTORY, "messages", "autoSaveData")
AUTOSAVE_PRICES_FILES_DIRECTORY = joinAndEnsure(SAVE_DATAS_DIRECTORY, "prices", "autoSaveData")
DIRECTORY_PRICES_CSV_FILES = joinAndEnsure(AUTOSAVE_PRICES_FILES_DIRECTORY, "csv")
DIRECTORY_PRICES_JSON_FILES = joinAndEnsure(AUTOSAVE_PRICES_FILES_DIRECTORY, "json")
DIRECTORY_PRICES_HDF_FILES = joinAndEnsure(AUTOSAVE_PRICES_FILES_DIRECTORY, "hdf")


### ram to disk
TEMP_DIRECTORY: Path # better be a ssd for AI training
if sys.platform == "win32": # => windows
    TEMP_DIRECTORY = joinAndEnsure(os.path.expandvars("%TEMP%"))
elif sys.platform == "linux": # => windows
    TEMP_DIRECTORY = joinAndEnsure(CURRENT_DIRECTORY, "TEMP")
else: # => not supported
    raise OSError(f"the platform: {sys.platform} isn't supported")

### to save AIs
AI_SAVE_DIRECTORY = Path("X:/TradingBot_AIs/") # save IAs on another dir
if AI_SAVE_DIRECTORY.exists() is False: 
    # if not available, save them here
    AI_SAVE_DIRECTORY = joinAndEnsure(CURRENT_DIRECTORY, "TradingBot_AIs")
AI_CHECKPOINTS_DIRECTORY = AI_SAVE_DIRECTORY.joinpath("checkpoints/")
AI_PLOT_CONFIGS_DIRECTORY = AI_SAVE_DIRECTORY.joinpath("plotConfigs/")



# nbLignes: ... (sans le code commenté), ... (avec le code commenté)
# nb de loc dans le module holo dev pour ce projet: ...
