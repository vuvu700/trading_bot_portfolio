# this file is here to describe the save format of the AI and the differents componants

import json
from pathlib import Path
import tkinter.scrolledtext
import numpy
from functools import partial
from datetime import datetime
from io import StringIO
import threading

import tkinter
import tkinter.filedialog
import tkinter.ttk
import tkinter.messagebox

from paths_cfg import CURRENT_DIRECTORY

from holo.__typing import (
    TypedDict, Callable, Generic, cast, NotRequired, Any, TYPE_CHECKING,
    TypeVarTuple, Unpack, TypeVar, overload, Self, Iterable, Literal,
    Union, LiteralString, assertIsinstance, )
from holo.prettyFormats import (
    PrettyPrint_CompactArgs, DEFAULT_COMPACT_RULES, print_exception, prettyPrintToJSON, )
from holo.types_ext import _Serie_Float
from holo.protocols import _P, _T, Protocol

if TYPE_CHECKING:
    import keras
    import keras.layers
    from calculationLib import _SerieName
    import AI
    from AI.senti3.datas.types_config import (
        _Backtests_fields, _DfKey,
        _MetricName, _MetricNameSingle, _MetricNameDfs, 
        _MetricValue, )
    from AI.senti3.training.lossFunctions import (
        _CustomLossFuncName, _NormalLossFuncName)
    from AI.senti3.optimizerFactorie import _OptimizerName
    from AI.senti3.kerasTypes import (
        KerasTensor, _ModelDtype, _ActivationName)
    from AI.senti3.modelsConfigs.layersFactory import (
        _ReferenceMode, _PoolingMethode, _Conv1DPadding,
        _BidirectionalMergeMode, )
    from AI.senti3.ai_versions.baseAI import _T_AI, _AI_Verbose
    from AI.senti3.datas.convergence import (
        _ConvergeVerbose, _RegroupMethode, _DeltasMethodes)
    from metricsPloter.config import _PointStyle, _LineStyle, _Scales
else: # needed due to generics
    _MetricName = TypeVar("_MetricName")
    _MetricNameSingle = TypeVar("_MetricNameSingle")
    _MetricNameDfs = TypeVar("_MetricNameDfs")
    _DfKey = str
    _T_AI = TypeVar("_T_AI") # can't bound correctely

JSON_SEMI_COMPACT_ARGS = \
    PrettyPrint_CompactArgs(
        compactSmaller=False, compactLarger=False,
        keepReccursiveCompact=False,
        compactRules=DEFAULT_COMPACT_RULES)

class SupportJson(Protocol[_T]):
    def toJson(self)->_T: ...
    @classmethod
    def fromJson(cls, datas:"_T")->"Self": ...

def toJsonMultiple(items:"Iterable[SupportJson[_T]]")->"list[_T]":
    return [elt.toJson() for elt in items]

_T_SupportJson = TypeVar("_T_SupportJson", bound=SupportJson)
_T_slice = TypeVar("_T_slice", bound=slice)

def fromJsonMultiple(multipleDatas:"Iterable[_T]", cls:"type[SupportJson[_T]]",
                     cls_:"type[_T_SupportJson]")->"list[_T_SupportJson]":
    """load from json multiple elements of type cls and cls_ (they must be the same)"""
    assert cls is cls_, TypeError(f"cls and cls_ must be the same")
    return [cls_.fromJson(elt) for elt in multipleDatas]






### manual repares

class ManualReparesToolApp(tkinter.Toplevel, Generic[_T_SupportJson, _T]):
    def __init__(self, cls:"type[_T_SupportJson]", datas:"_T", err:"Exception",
                 func:"Callable[[type[_T_SupportJson], _T], SupportJson[_T]]")->None:
        super().__init__()
        self.geometry("850x500")
        self.__destroyed: bool = False
        self.__func: "Callable[[type[_T_SupportJson], _T], SupportJson[_T]]" = func
        self.__cls: "type[_T_SupportJson]" = cls
        self.__baseDatas: "_T" = datas
        self.__baseErr: "Exception" = err
        self.__currentDatas: "_T" = datas
        self.__currentErr: "Exception|None" = err
        """None => the datas have just been updated but not evaluated"""
        self.__datasReady: bool = False
        self.__datasLock = threading.Lock()
        """to be held when opperating on the datas"""
        self.__initTkinter()
        self.updateInterface()
    
    def __initTkinter(self)->None:
        from metricsPloter.widgets import ScrollableFrame, ButtonsLine
        self.title("manual repares tool")
        self.resizable(True, True)
        # error label
        self.errorLabelFrame = ScrollableFrame(
            self, scrollSides="both", width=0, height=0)
        self.errorLabel = tkinter.Label(
            self.errorLabelFrame.scrollable_frame, justify="left")
        self.errorLabel.pack(fill="x", expand=False)
        self.errorLabelFrame.pack(anchor="nw", fill="x", expand=False)
        # text editor
        self.textEditorFrame = ScrollableFrame(
            self, scrollSides="both", width=0, height=0)
        self.textEditor = tkinter.Text(
            self.textEditorFrame.scrollable_frame, width=250, wrap="word")
        self.textEditor.pack(fill="both", expand=True)
        self.textEditorFrame.pack(
            anchor="ne", side="right", after=self.errorLabelFrame, fill="x", expand=True)
        # buttons
        self.buttons = ButtonsLine(self, [
            ("reset datas", self.reset),
            ("validate changes", self.validate)
        ], placement="pack")
        self.buttons.pack(anchor="sw")
    
    def reset(self)->None:
        with self.__datasLock:
            self.__currentDatas = self.__baseDatas
            self.__currentErr = self.__baseErr
            self.__datasReady = False
            self.updateInterface()
    
    def updateInterface(self)->None:
        # update the error message
        errStrBuilder = StringIO()
        if self.__currentErr is not None:
            print_exception(self.__currentErr, file=errStrBuilder)
        else: # => no error for now
            errStrBuilder.write("validation in process ...")
        self.errorLabel["text"] = errStrBuilder.getvalue()
        # update the text
        datasStrBuilder = StringIO()
        prettyPrintToJSON(
            self.__currentDatas, compact=False, 
            indentSequence=" "*2, stream=datasStrBuilder)
        self.textEditor.delete(1.0, tkinter.END)
        self.textEditor.insert(tkinter.END, datasStrBuilder.getvalue())
    
    def validate(self)->None:
        """parse the datas and """
        newStrDatas: str = self.textEditor.get("1.0", "end-1c")
        # => text from line 1 char 0, to the end (without an aditional new line)
        # update the datas and release the lock if the datas got parsed
        with self.__datasLock:
            if self.__datasReady is True:
                raise RuntimeError(f"calling validate but datas are alredy ready")
            # => (self.__datasReady is False)
            try: self.__currentDatas = json.loads(newStrDatas)
            except json.JSONDecodeError as err: 
                tkinter.messagebox.showerror(
                    title="parse failed", 
                    message="failed to parse the datas:\n" + str(err))
            else: self.__datasReady = True
    
    def __callFunc(self)->"_T_SupportJson":
        assert self.__datasLock.locked(), \
            RuntimeError("to call this function the lock should be acquired")
        result = self.__func(self.__cls, self.__currentDatas)
        assert isinstance(result, self.__cls), \
            TypeError(f"error with the wrapped function: {self.__func}"
                      f"it should have returned an instance of {self.__cls}"
                      f"but it returned an instance of {result.__class__}")
        return result
    
    def destroy(self)->None:
        self.__destroyed = True
        super().destroy()
    
    def getResult(self) -> "_T_SupportJson":
        while self.__destroyed is False:
            try:
                self.update()
            except tkinter.TclError: # => destroyed
                raise self.__baseErr
            with self.__datasLock:
                if self.__datasReady is True:
                    try: result = self.__callFunc()
                    except Exception as err:
                        # => failed to use the updated datas
                        # -> update l'app avec les nouvelles datas et avec la nouvelle erreur
                        self.__currentErr = err
                        self.__datasReady = False
                        self.updateInterface()
                    else: 
                        self.destroy()
                        return result
        # => destroyed
        raise self.__baseErr


def manualReparesWrapperGUI(
        func:"Callable[[type[_T_SupportJson], _T], SupportJson[_T]]",
        )->"Callable[[type[_T_SupportJson], _T], _T_SupportJson]":
    def wrapper(cls:"type[_T_SupportJson]", datas:"_T")->"_T_SupportJson":
        while True:
            try: result = func(cls, datas)
            except Exception as err:
                app = ManualReparesToolApp(cls, datas, err, func)
                return app.getResult() # => succeded after repares
            else: # => succeded first try
                assert isinstance(result, cls), (
                    f"error with the wrapped function: {func}"
                    f"it should have returned an instance of {cls}"
                    f"but it returned an instance of {result.__class__}")
                return result
    return wrapper

def manualReparesWrapperCLI(
        func:"Callable[[type[_T_SupportJson], _T], SupportJson[_T]]",
        )->"Callable[[type[_T_SupportJson], _T], _T_SupportJson]":
    def wrapper(cls:"type[_T_SupportJson]", datas:"_T")->"_T_SupportJson":
        isFirstTry: bool = True
        while True:
            try: result = func(cls, datas)
            except Exception as err:
                print("the following exception happend:")
                print_exception(err)
                print(f"with the datas:")
                prettyPrintToJSON(datas, compact=True)
                newStrDatas = input("- enter the new datas\n- 'exit' to stop\n")
                if newStrDatas in ("exit", "'exit'", '"exit"'):
                    print("invalide datas, exited without solving")
                    raise # => exited => reraise
                datas = cast(_T, assertIsinstance(dict, json.loads(newStrDatas)))
                isFirstTry = False
            else: # just because the return type of `func` can't be properly bounded, check it
                assert isinstance(result, cls), (
                    f"error with the wrapped function: {func}"
                    f"it should have returned an instance of {cls}"
                    f"but it returned an instance of {result.__class__}")
                if isFirstTry is False:
                    print(f"datas fixed !\n")
                return result
        raise Exception(f"[BUG] this code shouldn't be reachable")
    return wrapper








### Callables

REGISTERED_FUCNTIONS: "dict[str, Callable]" = {}

def __getFuncID(func:Callable)->str:
    funcPathRelative = Path(func.__code__.co_filename).relative_to(CURRENT_DIRECTORY)
    return f"{func.__name__}@{funcPathRelative.as_posix()}"

_T_Func = TypeVar("_T_Func", bound=Callable)
def registerFunction(func:"_T_Func")->"_T_Func":
    """wrapper to register a function in order to load it from json"""
    global REGISTERED_FUCNTIONS
    funcID: str= __getFuncID(func)
    if funcID in REGISTERED_FUCNTIONS:
        raise KeyError(f"the function: {funcID} is arlredy registerd")
    # => register the new function
    REGISTERED_FUCNTIONS[funcID] = func
    return func

class AsJson_Function(TypedDict, Generic[_P, _T]):
    funcID: str

def toJsonFunc(func:"Callable[_P, _T]")->"AsJson_Function[_P, _T]":
    funcID: str = __getFuncID(func)
    if funcID not in REGISTERED_FUCNTIONS:
        raise KeyError(f"can't convert to json the function: {repr(funcID)} as it isn't registered")
    return AsJson_Function(funcID=funcID)

def fromJsonFunc(datas:"AsJson_Function[_P, _T]")->"Callable[_P, _T]":
    if datas["funcID"] not in REGISTERED_FUCNTIONS:
        raise KeyError(f"can't load the function: {repr(datas['funcID'])} as it isn't registered")
    return cast("Callable[_P, _T]", REGISTERED_FUCNTIONS[datas['funcID']])

class AsJson_PartialFunction(TypedDict, Generic[_P, _T]):
    function: "AsJson_Function[_P, _T]"
    args: "list[Any]"
    kwargs: "dict[str, Any]"

def toJsonPartialFunc(part:"partial[_T]")->"AsJson_PartialFunction[..., _T]":
    return AsJson_PartialFunction(
        function=toJsonFunc(part.func),
        args=list(part.args), kwargs=part.keywords.copy())

def fromJsonPartialFunc(datas:"AsJson_PartialFunction[..., _T]")->"partial[_T]":
    return partial(fromJsonFunc(datas["function"]), *datas["args"], **datas["kwargs"])


### tuple
_T_tuple = TypeVar("_T_tuple", bound=tuple)
_Tuple = TypeVarTuple("_Tuple")
class AsJson_tuple(TypedDict, Generic[Unpack[_Tuple]]):
    cls: str
    elements: "list[Any]"
    
def toJsonTuple(tpl:"tuple[Unpack[_Tuple]]")->"AsJson_tuple[Unpack[_Tuple]]":
    return AsJson_tuple(cls=tpl.__class__.__name__, elements=list(tpl))

@overload
def fromJsonTuple(
        cls:"None", 
        datas:"AsJson_tuple[Unpack[_Tuple]]")->"tuple[Unpack[_Tuple]]": ...
@overload
def fromJsonTuple(
        cls:"type[_T_tuple]", 
        datas:"AsJson_tuple[Unpack[_Tuple]]")->"_T_tuple": ...
def fromJsonTuple(
        cls:"type[_T_tuple]|None", datas:"AsJson_tuple[Unpack[_Tuple]]",
        )->"_T_tuple|tuple[Unpack[_Tuple]]":
    if cls is None: 
        cls = cast("type[_T_tuple]", tuple)
    assert datas["cls"] == cls.__name__
    tpl = tuple.__new__(cls, datas["elements"])
    return tpl

### slice

class AsJson_Slice(TypedDict):
    cls: str
    start: "int|None"
    stop: "int|None"
    step: "int|None"

def fromJsonSlice(datas:"AsJson_Slice")->slice:
    return slice(datas["start"], datas["stop"], datas["step"])

def toJsonSlice(slc:slice)->"AsJson_Slice":
    return AsJson_Slice(
        cls=slc.__class__.__name__,
        start=slc.start, stop=slc.stop, step=slc.step)

### dtypes

_NumpyTypesName_to_type: "dict[str, type[numpy.generic]]" = \
    {npType.__name__: npType for npType in numpy.nbytes.keys()}

_T_numpyType = TypeVar("_T_numpyType", bound=numpy.generic)

class AsJson_dtype(TypedDict, Generic[_T_numpyType]):
    dtypeName: str
    
def toJson_dtype(dtype:"type[_T_numpyType]")->"AsJson_dtype[_T_numpyType]":
    typeName: str = dtype.__name__
    sameDtype: "type[numpy.generic]|None" = _NumpyTypesName_to_type.get(typeName)
    if sameDtype is None:
        raise TypeError(f"the given dtype: {dtype} is not jsonable")
    assert dtype is sameDtype
    return AsJson_dtype(dtypeName=dtype.__name__)

def fromJson_dtype(datas:"AsJson_dtype[_T_numpyType]")->"type[_T_numpyType]":
    typeName: str = datas["dtypeName"]
    dtype: "type[numpy.generic]|None" = _NumpyTypesName_to_type.get(typeName)
    if dtype is None:
        raise TypeError(f"the dtype: {dtype} from the json datas can't be decoded")
    return cast("type[_T_numpyType]", dtype)


### datetime types

class AsJson_Datetime(TypedDict):
    cls: str
    value: str


DATETIME_FORMAT = "%d/%m/%Y-%Hh%M:%S.%f"
def _datetimeToText(t:datetime)->str:
    return t.strftime(DATETIME_FORMAT)
def _datetimeFromText(text:str)->datetime:
    return datetime.strptime(text, DATETIME_FORMAT)

def datetimeToJson(t: datetime)->"AsJson_Datetime":
    return AsJson_Datetime(cls=datetime.__name__, value=_datetimeToText(t))
def datetimeFromJson(datas:"AsJson_Datetime")->"datetime":
    assert datas["cls"] == datetime.__name__
    return _datetimeFromText(datas["value"])









### types for the configs

class AsJson_Model_IO_config(TypedDict):
    cls: str
    nbPeriodes_inputs: int
    nbPeriodes_outputs: int
    outputPeriodesShift: int
    input_series: "list[_SerieName]"
    output_series: "list[_SerieName]"
    datasDtype: "AsJson_dtype[numpy.float16|numpy.float32|numpy.float64]"
    regularizeSeriesConfig: "AsJson_RegularizeConfig"

class AsJson_LoadModelArgs(TypedDict, Generic[_T_AI]):
    cls: str
    ai_cls: str
    configFilePath: str
    
class AsJson_TrainCallHist(TypedDict):
    cls: str
    trainFunc: "AsJson_Function[..., Any]"
    funcsKwargs: "dict[str, Any]"
    nbEpochesDone: int
    hasFinished: bool
    
class AsJson_SeriesGeneratorConfig(TypedDict):
    cls: str
    singleSeriesGenerator: "list[AsJson_SingleSerieGeneratorConfig]"
    multiSeriesGenerator: "list[AsJson_MultiSerieGeneratorConfig]"

class AsJson_SingleSerieGeneratorConfig(TypedDict):
    serieName: "_SerieName"
    func: "AsJson_Function[..., _Serie_Float]"
    kwargs: "dict[str, Any]"
    
class AsJson_MultiSerieGeneratorConfig(TypedDict):
    seriesNames: "list[_SerieName]"
    func: "AsJson_Function[..., tuple[_Serie_Float, ...]]"
    kwargs: "dict[str, Any]"


class AsJson_ResumeConfig(TypedDict, Generic[_MetricName]):
    cls: str
    metricName: "_MetricName"
    alias: "str|None"
    func: "AsJson_Function[[float], str]|None"

class AsJson_CustomLossFuncConfig(TypedDict):
    cls: str
    functionName: "_CustomLossFuncName"
    kwargs: "dict[str, Any]"


class AsJson_LossToLr(TypedDict):
    cls: str
    loss: float
    lr: float

class AsJson_LossToLrTable(TypedDict):
    cls: str
    table: "list[AsJson_LossToLr]"

class AsJson_KerasConfig(TypedDict):
    cls: str
    reliableMode: "bool|None"
    memoryGrowth: "bool|None"
    toRevert: bool


class AsJson_OptimizerFactoryConfig(TypedDict):
    cls: str
    optimizerName: "_OptimizerName"
    optimizerKwargs: "dict[str, Any]"


class AsJson_FittingHistory(TypedDict): 
    cls: str
    fittCallsParams: "list[dict[str, int|str]]"
    

class AsJson_MetricsHistory(TypedDict, Generic[_MetricNameSingle, _MetricNameDfs]):
    cls: str
    singleMetrics: "AsJson_Metrics[_MetricNameSingle]"
    dfsMetrics: "AsJson_MetricsDfs[_MetricNameDfs]"
    
class AsJson_Metrics(TypedDict, Generic[_MetricName]):
    cls: str
    values: "dict[_MetricName, dict[int, _MetricValue]]"

class AsJson_MetricsDfs(TypedDict, Generic[_MetricName]):
    cls: str
    metrics: "list[_MetricName]"
    checkDfKey:"AsJson_Function[[str], bool]|None"
    values: "dict[_DfKey, AsJson_Metrics[_MetricName]]"


class AsJson_ValuesRange(TypedDict):
    cls: str
    mini: float
    maxi: float

class AsJson_RegularizeConfig(TypedDict):
    cls: str
    rescale: bool
    preferReRangePlus: bool
    valuesRange: "AsJson_ValuesRange"


class AsJson_DatasetInfos(TypedDict):
    cls: str
    key: str
    length: int
    startDate: "AsJson_Datetime"
    endDate: "AsJson_Datetime"


class AsJson_ConvergeConfig(TypedDict):
    cls: str
    seriesToConverge: "list[AsJson_ConvergeParams]"
    verbose: "_ConvergeVerbose"
    aiVerbose: "_AI_Verbose" # unchecked
    maxNbIterations: int
    showAvgDeltas: bool

class AsJson_ConvergeParams(TypedDict):
    cls: str
    serieName: "_SerieName"
    convergeMethode: "_RegroupMethode"
    startNoise: "float|bool"
    targetedDelta: float
    deltaMethode: "_DeltasMethodes"
    emaConvergenceCoeff: "float|Literal[False]"

class AsJson_ConvergenceOutputResume(TypedDict):
    cls: str
    inputedDataframeInfos: "AsJson_DatasetInfos"
    nbIterationsDone: int
    deltas_perStep: "list[dict[_SerieName, float]]"
    hasConverged: bool
    averageTimePerStep: float


class AsJson_StrategieResults(TypedDict):
    cls: str
    kwargs: "dict[_Backtests_fields, Any]"

class AsJson_BacktestResult(TypedDict):
    cls: str
    datasetInfos: "AsJson_DatasetInfos"
    result: "AsJson_StrategieResults"

class AsJson_BacktestsResultsStorage(TypedDict):
    cls: str
    perEpochsResults: "dict[_DfKey, dict[int, AsJson_BacktestResult]]"
    otherResults: "list[AsJson_BacktestResult]"


### layers factory

class AsJson_InputLayer(TypedDict):
    cls: str
    nbFeatures: int
    nbPeriodes: int
    dtype: "_ModelDtype"
    
class AsJson_ReferenceLayer(TypedDict):
    cls: str
    index: int
    mode: "_ReferenceMode"
    

class AsJson_DropoutLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    forget: float

class AsJson_BidirectionalLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    subLayer: "AsJson_LSTM_Layer|AsJson_LSTM_CUDNN_Layer"
    merge_mode: "_BidirectionalMergeMode"

class AsJson_LSTM_Layer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    return_sequences: bool
    return_state: bool
    forward: bool
    inputDropoutRate: float
    recurrentDropoutRate: float
    activation: "_ActivationName"
    recurrent_activation: "_ActivationName"
    unroll: bool

class AsJson_GRU(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    return_sequences: bool
    forward: bool
    recurrentDropoutRate: float
    activation: "_ActivationName"
    recurrent_activation: "_ActivationName"

class AsJson_LSTM_CUDNN_Layer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    return_sequences: bool
    return_state: bool
    forward: bool
    
class AsJson_AddLayer(TypedDict):
    cls: str
    name: "str|None"
    inputsRef: "list[AsJson_ReferenceLayer]"

class AsJson_MultiHeadAttentionLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    nbHeads: int
    headsSize: int
    projectionSize: "int|None"

class AsJson_DenseLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    activation: "_ActivationName"

class AsJson_LambdaLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    func: "AsJson_Function[[KerasTensor], KerasTensor]"

class AsJson_NormalizationLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    axis: int
    
class AsJson_GlobalPooling1DLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    methode: "_PoolingMethode"

class AsJson_ReshapeLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    outputShape: "list[int]"

class AsJson_FlattenLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"

class AsJson_Conv1D_Layer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    kernelSize: int
    dilation_rate: int
    activation: "_ActivationName"
    padding: "_Conv1DPadding"

class AsJson_TrimPeriodesLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    nbPeriodesToTrim: int
    trimFirstPeriodes: bool

class AsJson_RandomExtendLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    noiseCoef: float
    nbNewPeriodes: int 
    addAtTheEnd: bool
    
class AsJson_PreTrainedSingleInputLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    preTrainedModelKey: str
    layerGetter: "AsJson_Function[[keras.Model], keras.layers.Layer]"

class AsJson_TimeDistributedLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    subLayer: "AsJson_LayerConfigs"

class AsJson_LayersGroup(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    subLayers: "list[AsJson_LayerConfigs]"

class AsJson_FixedSingleOutputLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    subLayer: "AsJson_LayerConfigs"

class AsJson_SlicesLayer(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    slices: "list[AsJson_Slice|None|int]"

class AsJson_TimeDense(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    nbTimePeriodes: int
    activation: "_ActivationName|None"

class AsJson_Polynomial(TypedDict):
    cls: str
    name: "str|None"
    inputRef: "AsJson_ReferenceLayer|None"
    units: int
    degrees: "list[float]"
    activation: "_ActivationName|None"

AsJson_LayerConfigs = Union[
    AsJson_DropoutLayer, AsJson_BidirectionalLayer,
    AsJson_LSTM_Layer, AsJson_LSTM_CUDNN_Layer, AsJson_AddLayer,
    AsJson_MultiHeadAttentionLayer, AsJson_DenseLayer, 
    AsJson_LambdaLayer, AsJson_NormalizationLayer, 
    AsJson_GlobalPooling1DLayer, AsJson_ReshapeLayer,
    AsJson_FlattenLayer, AsJson_Conv1D_Layer, AsJson_GRU,
    AsJson_TrimPeriodesLayer, AsJson_RandomExtendLayer, 
    AsJson_PreTrainedSingleInputLayer, AsJson_TimeDistributedLayer, 
    AsJson_LayersGroup, AsJson_FixedSingleOutputLayer,
    AsJson_SlicesLayer, AsJson_TimeDense, AsJson_Polynomial]

class AsJson_ModelFactoryConfig(TypedDict):
    cls: str
    inputs: "AsJson_InputLayer"
    layers: "list[AsJson_LayerConfigs]"

class AsJson_AutoEncoderFactoryConfig(TypedDict):
    cls: str
    encoder: "AsJson_ModelFactoryConfig"
    decoder: "AsJson_ModelFactoryConfig"

Union_AsJson_ModelsFactoryConfig = Union[
    AsJson_ModelFactoryConfig, AsJson_AutoEncoderFactoryConfig]

### BaseAI

class AsJson_BaseAI(TypedDict):
    cls: str
    seriesGeneratorConfig: "dict[_SerieName, AsJson_SeriesGeneratorConfig]"
    ioConfig: "AsJson_Model_IO_config"
    preTrainedModels: "dict[str, AsJson_LoadModelArgs]"
    resumeCfg: "list[AsJson_ResumeConfig]"
    plotCfg: "AsJson_PlotConfig"
    loss_func_config: "AsJson_CustomLossFuncConfig|_NormalLossFuncName"
    backtestsResults: "AsJson_BacktestsResultsStorage"
    lossToLearningRate_table: "AsJson_LossToLrTable"
    modelFactoryConfig: "Union_AsJson_ModelsFactoryConfig"
    optimizerFactoryConfig: "AsJson_OptimizerFactoryConfig"
    kerasConfig: "AsJson_KerasConfig"
    batch_size: int
    metricsHistory: "AsJson_MetricsHistory"
    fittingHistory: "AsJson_FittingHistory"
    trainingCallsHistory: "list[AsJson_TrainCallHist]"
    nbEpochesFinished: int 
    modelFullPath: "str|None"
    comments: "list[str]"


### plot config 

class AsJson_Limits(TypedDict):
    cls: str
    mini: float
    maxi: float


class AsJson_DfsMetricConfig(TypedDict):
    cls: str
    allKeysConfig: "list[AsJson_LineConfig]"
    perKeyConfig: "dict[_DfKey, list[AsJson_LineConfig]]"

class AsJson_SingleMetricConfig(TypedDict):
    cls: str
    configs: "list[AsJson_LineConfig]"


class AsJson_AxisMetricsConfigs(TypedDict, Generic[_MetricNameSingle, _MetricNameDfs]):
    cls: str
    disabledSingleMetrics: "list[_MetricNameSingle]"
    disabledDfsMetrics: "list[_MetricNameDfs]"
    singleLinesConfigs: "dict[_MetricNameSingle, AsJson_SingleMetricConfig]"
    dfsLinesConfigs: "dict[_MetricNameDfs, AsJson_DfsMetricConfig]"
    

class AsJson_AxisConfig(TypedDict, Generic[_MetricNameSingle, _MetricNameDfs]):
    cls: str
    name: str
    yLabel: str
    scale: "_Scales"
    hlines: "list[float]"
    yLimits: "AsJson_Limits|None"
    indexs: "int|AsJson_tuple[int, int]"
    """define the size and the position of the axis"""
    metricsConfigs: "AsJson_AxisMetricsConfigs[_MetricNameSingle, _MetricNameDfs]"

class AsJson_LineConfig(TypedDict):
    cls: str
    enabled: bool
    color: "str|None"
    lineStyle: "_LineStyle"
    pointStyle: "_PointStyle"
    width: float
    emaCoeff: "float|None"

class AsJson_FigureConfig(TypedDict):
    cls: str
    figureID: str
    nbRows: int
    nbCols: int
    plotSize: "AsJson_tuple[float, float]"


class AsJson_PlotConfig(TypedDict, Generic[_MetricNameSingle, _MetricNameDfs]):
    cls: str
    axisConfigs: "dict[str, AsJson_AxisConfig[_MetricNameSingle, _MetricNameDfs]]"
    figureConfig: "AsJson_FigureConfig"
    