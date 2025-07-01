import tkinter
import tkinter.filedialog
import tkinter.ttk
import tkinter.messagebox

import copy
import numpy
from pathlib import Path
from io import StringIO, BufferedReader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as plt_Figure


from paths_cfg import AI_PLOT_CONFIGS_DIRECTORY
from .config import (
    PlotConfig, AxisConfig, extractTextList, extractIndexs,
    FigureConfig, LineConfig, Limits, AxisMetricsConfigs, 
    _FigureConfigField, _LineStyle, _PointStyle, _DfKey, _Scales, )
from .widgets import (
    _Field, CustomTopLevel, WidgetsLine, ButtonsLine,
    TextEntryLine, MultipleTextEntryLine, 
    FontsManger, ComboboxLine, OptionMenuLine,
    CheckableManyFrame, CheckableFrameTextEditor, 
    ScrollableFrame, CheckableEntryLine, )

from AI.senti3.datas.types_config import (
    AllMetrics, _MetricNameSingle, _MetricNameDfs, )

from holo import MapDictValues2, Pointer
from holo.__typing import ( 
    Generic, Literal, TracebackType, Self, Callable, TypeAlias,
    override, overload, get_args_LiteralString, Any, assertIsinstance, )
from holo.protocols import SupportsContext
from holo.prettyFormats import print_exception
from holo.types_ext import _Serie_Float
from holo.linkedObjects import History, NoHistoryError
from holo.ramDiskSave import Session, SaveArgs, ObjectSaver


"""
import numpy
import random
import metricsPloter
from AI.senti3.types_config import AllMetrics

config = metricsPloter.PlotConfig(
    axisConfigs={},
    figureConfig=metricsPloter.FigureConfig(
        figureID="fig", nbCols=2, nbRows=2, plotSize=(10, 10)))
config.addAxisConfig(metricsPloter.AxisConfig(
    name="abc", yLabel="loss", singleMetrics={}, dfsMetrics={},
    hlines={0.1, 0.5, 0.9}, indexs=(1, 2),
    yLimits=metricsPloter.Limits(mini=0, maxi=1)))

datas = AllMetrics({"loss", "loss_val", "nbIters", "time"},
                   {"loss", "loss_val", "nbIters"}, None)
datas.singleMetrics.addMultiple({("loss", e): random.random() for e in range(15)})
datas.singleMetrics.addMultiple({("loss_val", e): random.random() for e in range(15)})
datas.singleMetrics.addMultiple(
    {("nbIters", e): random.randint(9, 15) for e in range(15)})
datas.singleMetrics.addMultiple(
    {("time", e): random.uniform(50, 75) for e in range(15)})

for dfKey in ("df1", "df2", "df3"):
    a, b = ((0, 15) if dfKey in ("df1", "df2") else (5, 10))
    datas.dfsMetrics.addMultiple(
        {(dfKey, "loss", e): random.random() for e in range(a, b)})
    datas.dfsMetrics.addMultiple(
        {(dfKey, "loss_val", e): random.random() for e in range(a, b)})
    datas.dfsMetrics.addMultiple(
        {(dfKey, "nbIters", e): random.randint(9, 15) for e in range(a, b)})


app = metricsPloter.App(baseConfig=config, datas=datas)
app.mainloop()
"""


_SaveResponse = Literal['done', 'canceled']
_ALL_DFS = Literal["ALL_DFS"]

_CFG_NAME: TypeAlias = "_Field | tuple[_Field, _DfKey|_ALL_DFS]"
_Configs: TypeAlias = "list[tuple[_CFG_NAME[_Field], LineConfig]]"

DATAS_FILE_TYPES: "list[tuple[str, str]]" = [("CONFIG", ".pltCfg"), ]
"""the file types that are accepted for the datas: [(name, .extention), ...]"""

########## utils

def showError(master:"tkinter.Misc", title:str, message:str)->None:
    tkinter.messagebox.showerror(title=title, message=message)
    master.focus()

def showInfos(master:"tkinter.Misc", title:str, message:str)->None:
    tkinter.messagebox.showinfo(title=title, message=message)
    master.focus()

def showWarrning(master:"tkinter.Misc", title:str, message:str)->None:
    tkinter.messagebox.showwarning(title=title, message=message)
    master.focus()


def ensure_matplotlib_IPY_kernel()->None:
    from IPython import get_ipython # type: ignore
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic("matplotlib qt")

########## application components


class App(tkinter.Tk, Generic[_MetricNameSingle, _MetricNameDfs]):

    def __init__(self, baseConfig:"PlotConfig[_MetricNameSingle, _MetricNameDfs]",
                 datas:"AllMetrics[_MetricNameSingle, _MetricNameDfs]", verbose:bool) -> None:
        ensure_matplotlib_IPY_kernel()
        super().__init__("metrics ploter")
        self.protocol("WM_DELETE_WINDOW", self.exit)
        self.verbose: bool = verbose
        # set the config like that to don't create an history point
        self.__plotConfig: "PlotConfig[_MetricNameSingle, _MetricNameDfs]" = baseConfig
        self.fonts = FontsManger(master=self, normal=None)
        self.datas: "AllMetrics[_MetricNameSingle, _MetricNameDfs]" = datas
        self.plotter: "Plotter[_MetricNameSingle, _MetricNameDfs]" = Plotter(app=self)
        self.history = History(ObjectSaver[PlotConfig[_MetricNameSingle, _MetricNameDfs]])
        self.session = Session(savingArgs=SaveArgs(
            compression=("lz4", 16), methode="allwaysPickle"))
        self.__lastSavedID: int = self.history.getCurrentNodeID()
        """consider it is saved if it didn't chnaged since originaly given"""
        self.__saveFilePath: "Path|None" = None
        """when it is None, it mean that it don't have an assigned save file yet"""
        
        self.menus: "Menus" = Menus(self)
        self.mainFrame: "MainFrame[_MetricNameSingle, _MetricNameDfs]" = MainFrame(self)
        
        self.mainFrame.pack()
        self["menu"] = self.menus
    
    @property
    def plotConfig(self)->"PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        return self.__plotConfig
    
    def updatedConfig(self)->None:
        """tell the app that the config has been edited (-> update the widgets)"""
        self.mainFrame.updatedConfig()
        self.menus.updatedConfig()
    
    def _generateCheckpointDatas(self)->"ObjectSaver[PlotConfig[_MetricNameSingle, _MetricNameDfs]]":
        return ObjectSaver(value=copy.deepcopy(self.plotConfig), session=self.session)
    
    def createCheckPoint(self)->None:
        """save the current config to the history"""
        self.history.addCheckpoint(self._generateCheckpointDatas())
    
    def undo(self)->None:
        """go back to the previous checkpoint (revertable with redo)
        `fail`: True => undo without creating a redo (to redo a failed operation)"""
        try: self._loadPlotConfig(self.history.undoOne().value)
        except NoHistoryError: showError(self, "failed undo", "no history to undo")

    def redo(self)->None:
        """reload the state before the previous undo (revertable with undo)"""
        try: self._loadPlotConfig(self.history.redoOne().value)
        except NoHistoryError: showError(self, "failed redo", "no history to redo")
    
    def plotConfigIsSaved(self)->bool:
        return (self.__lastSavedID == self.history.getCurrentNodeID())
    
    def exit(self)->None:
        """start the process of classing the app (might require user inputs)"""
        saveResponse = self.askToSave()
        if saveResponse == "canceled": 
            pass # => don't exit
        elif saveResponse == "done": 
            # => can exit
            self.destroy()
        else: raise ValueError(f"invalide save response: {saveResponse}")
        return None
    
    def askToSave(self)->"_SaveResponse":
        """if needed, ask the user to save the config\n
        returns "done" if the user saved or not, 
        "canceled" if the user canceled the save operation"""
        if self.plotConfigIsSaved() is True:
            return "done"
        # ask to save it
        response = tkinter.messagebox.askquestion(
            title="unsaved config",
            message="the config has been modified since last save\n" 
                        + "do you whant to save the modifications before closing the app ?",
            type=tkinter.messagebox.YESNOCANCEL)
        if response == tkinter.messagebox.YES:
            # save the datas
            hasSaved = self.saveToFile()
            if hasSaved is False:
                return "canceled" # canceled saving => don't close
            return "done"
        elif response == tkinter.messagebox.NO:
            return "done" # do nothing and quit
        elif response == tkinter.messagebox.CANCEL:
            return "canceled" # abort closing the app
        else: raise ValueError(f"invalide response: {response}")

    def __askFilenameToSaveDatas(self, master:"tkinter.Misc")->"Path|None":
        fileName = tkinter.filedialog.asksaveasfilename(
            initialdir=AI_PLOT_CONFIGS_DIRECTORY, parent=master, 
            defaultextension=DATAS_FILE_TYPES[0][1], filetypes=DATAS_FILE_TYPES, 
            title="select file to save the plot config")
        if fileName == "":
            return None # => canceled
        return Path(fileName)

    def __openFileAndLoadConfig(self)->"PlotConfig[_MetricNameSingle, _MetricNameDfs]|None":
        """ask the file to open and load the datas from it"""
        # ask a file to open
        file = tkinter.filedialog.askopenfile(
            mode="rb", initialdir=AI_PLOT_CONFIGS_DIRECTORY, 
            parent=self, filetypes=DATAS_FILE_TYPES,
            title="select the config file to open")
        if file is None: # => no file selected, don't open anything
            return None
        assert isinstance(file, BufferedReader)
        # read the datas in it and update the datas to the app
        return self.plotConfig.fromFile(allMetrics=self.datas, file=file)

    def __saveToFile(self, path:Path)->None:
        self.plotConfig.saveToFile(path)
        self.__lastSavedID = self.history.getCurrentNodeID()

    def _loadPlotConfig(self, config:"PlotConfig[_MetricNameSingle, _MetricNameDfs]")->None:
        """to load a new config (do not use)"""
        self.__plotConfig = config
        self.updatedConfig()
        
    def openFromFile(self)->None:
        saveResponse = self.askToSave()
        if saveResponse == "canceled": 
            return None
        newConfig = self.__openFileAndLoadConfig()
        if newConfig is None:
            return None # canceled
        self._loadPlotConfig(newConfig)

    def saveToFile(self)->"_SaveResponse":
        """save the datas (ask a file if it don't have one) 
        and return whether the datas was saved"""
        if self.__saveFilePath is None: 
            # => don't have a save path => ask for it
            return self.saveAsToFile() # finished
        # => have a file to save to
        if self.plotConfigIsSaved() is True:
            # => alredy saved
            tkinter.messagebox.showinfo(title="save info", message="alredy saved '~'")
        else: # => not saved
            self.__saveToFile(self.__saveFilePath)
        return "done"
        
    def saveAsToFile(self)->"_SaveResponse":
        """ask a file to save the datas and return whether the datas was saved"""
        # ask a file for saving
        filePath: "Path|None" = self.__askFilenameToSaveDatas(master=self)
        if filePath is None: # => no file selected, don't save anything
            return "canceled"
        self.__saveToFile(filePath)
        return "done"
        
    def newConfig(self)->None:
        self._loadPlotConfig(self.plotConfig.emptyCopy())




class Plotter(Generic[_MetricNameSingle, _MetricNameDfs]):
    """this class handle the plot and how the config and datas are used to be plotted"""
    
    def __init__(self, app:"App[_MetricNameSingle, _MetricNameDfs]")->None:
        self.application: "App[_MetricNameSingle, _MetricNameDfs]" = app
        self.figure: "None|plt_Figure" = None
    
    @property
    def config(self)->"PlotConfig[_MetricNameSingle, _MetricNameDfs]":
        return self.application.plotConfig
    
    def setupPlot(self)->"list[tuple[AxisConfig[_MetricNameSingle, _MetricNameDfs], plt.Axes]]":
        if self.figure is not None:
            try: self.figure.clear()
            except: pass # will be replaced later
        self.figure = self.config.figureConfig.getFigure()
        return self.config.setupFigure(self.figure)
    
    def plotMetrics(self)->None:
        datas = self.application.datas
        axis = self.setupPlot()
        assert self.figure is not None
        
        # plot everything
        for (axConfig, ax) in axis:
            axConfig.plot(ax=ax, datas=datas)
        # show
        self.figure.set_tight_layout(True)
        self.figure.show()
        
        if self.application.verbose is True:
            print("[plotted the datas with the config]")
            from holo.prettyFormats import prettyPrint, PrettyPrint_CompactArgs 
            prettyPrint(self.config, specificCompact={LineConfig, set, Limits, tuple},
                        compact=PrettyPrint_CompactArgs(keepReccursiveCompact=False))






class MainFrame(tkinter.Frame, Generic[_MetricNameSingle, _MetricNameDfs]):
    
    def __init__(self, application:"App[_MetricNameSingle, _MetricNameDfs]") -> None:
        super().__init__(application)
        self.application: "App[_MetricNameSingle, _MetricNameDfs]" = application
        self.addAxisConfigDialog: "AddAxisConfigDialog|None" = None
        self.editAxisConfigDialog: "EditAxisConfigDialog|None" = None
        self.removeAxisConfigDialog: "RemoveAxisConfigDialog|None" = None
        self.axisConfigEditors: "dict[str, AxisConfigEditor[_MetricNameSingle, _MetricNameDfs]]" = {}
        
        self.editPlot = tkinter.Button(
            self, text="edit figure config", command=self.application.menus.startFigureConfigEditor)
        self.axisButons = ButtonsLine(
            master=self, buttonsInfos=[
                ("add axis", self.openAddAxisConfigDialog),
                ("edit axis", self.openEditAxisConfigDialog),
                ("remove axis", self.openRemoveAxisConfigDialog),],
            placement="pack", placementKwargs=dict(pady=5, padx=5))
        self.plot_button = tkinter.Button(
            self, text="Plot", command=self.application.plotter.plotMetrics)
        
        self.editPlot.pack()
        self.axisButons.pack()
        self.plot_button.pack(pady=20)

    def openAddAxisConfigDialog(self)->None:
        if self.addAxisConfigDialog is not None:
            showInfos(master=self.addAxisConfigDialog,
                      title="add axis config dialog", 
                      message="the add axis config dialog is alredy opened")
        else: # => not opened => open a new one
            self.addAxisConfigDialog = AddAxisConfigDialog(self)
        
    def openEditAxisConfigDialog(self)->None:
        if self.editAxisConfigDialog is not None:
            showInfos(master=self.editAxisConfigDialog,
                      title="edit axis config dialog", 
                      message="the edit axis config dialog is alredy opened")
        # => not opened => open a new one
        elif len(self.application.plotConfig.axisConfigs) == 0:
            showError(master=self, 
                      title="no axis available", 
                      message="there are no axis to edit")
        else: # there are axis to edit
            self.editAxisConfigDialog = EditAxisConfigDialog(self)
        
    def openRemoveAxisConfigDialog(self)->None:
        if self.removeAxisConfigDialog is not None:
            showInfos(master=self.removeAxisConfigDialog,
                      title="edit axis config dialog", 
                      message="the edit axis config dialog is alredy opened")
        # => not opened => open a new one
        elif len(self.application.plotConfig.axisConfigs) == 0:
            showError(master=self, 
                      title="no axis available", 
                      message="there are no axis to edit")
        else: # there are axis to remove
            self.removeAxisConfigDialog = RemoveAxisConfigDialog(self)

    def isAxisCfgEditorOpened(self, axisName:str)->"bool":
        # (None => not opened, other => opened)
        return (axisName in self.axisConfigEditors.keys())

    def openConfigEditor(self, axisName:str)->None:
        if self.isAxisCfgEditorOpened(axisName) is True:
            showInfos(master=self.axisConfigEditors[axisName], 
                      title="axis config editor", 
                      message="this axis config editor is alredy opened")
        else: # => not opened => open a new one
            self.axisConfigEditors[axisName] = \
                AxisConfigEditor(self, axisName=axisName)

    def updatedConfig(self)->None:
        if self.addAxisConfigDialog is not None:
            # don't have an update methode (assert it)
            assert not hasattr(self.addAxisConfigDialog, "updatedConfig")
        if self.editAxisConfigDialog is not None:
            self.editAxisConfigDialog.updatedConfig()
        if self.removeAxisConfigDialog is not None:
            self.removeAxisConfigDialog.updatedConfig()
        # (if eraised => their editor should have been closed)
        editorsToRemove: "set[str]" = \
            set(self.axisConfigEditors.keys()).difference(
                set(self.application.plotConfig.axisConfigs.keys()))
        for editorKey in editorsToRemove:
            self.axisConfigEditors[editorKey].destroy()
        
            


class AddAxisConfigDialog(CustomTopLevel[_MetricNameSingle, _MetricNameDfs]):
    def __init__(self, master:"MainFrame[_MetricNameSingle, _MetricNameDfs]")->None:
        self.master: "MainFrame[_MetricNameSingle, _MetricNameDfs]" = master
        super().__init__(
            master, master.application, "add axis config", resizeable=False)
        self.newCfgLabel = tkinter.Label(self, text="base infos on the new axis")
        self.entrys = MultipleTextEntryLine(
            master=self, fixedTextToDefaultText={
                "name": ("axis name:", ""), "label": ("y label:", ""), 
                "indexs": ("axis's plot indexs\n(int|tuple[int, int]):", "")})
        self.createCfgButton = tkinter.Button(
            self, text="creat the axis config", command=self.createNewAxisConfig)
        self.newCfgLabel.grid(row=0)
        self.entrys.grid(row=1)
        self.createCfgButton.grid(row=2)

    def createNewAxisConfig(self)->None:
        axisName: str = self.entrys["name"]
        if axisName == "":
            return showError(self, "invalid entry", "the new axis name can't be empty")
        with ErrorHandlerWithMessage("invalid entry", self):
            indexs = extractIndexs(self.entrys["indexs"])
        # create the new cfg
        newAxisConfig = AxisConfig.empty(
            name=axisName, indexs=indexs,
            yLabel=self.entrys["label"])
        # applie the new cfg to the config
        self.application.createCheckPoint()
        with ErrorHandlerWithMessage("add axis config", self):
            self.application.plotConfig.addAxisConfig(newAxisConfig)
        self.destroy()
    
    @override
    def destroy(self)->None:
        self.master.addAxisConfigDialog = None
        return super().destroy()

class EditAxisConfigDialog(CustomTopLevel[_MetricNameSingle, _MetricNameDfs]):
    def __init__(self, master:"MainFrame[_MetricNameSingle, _MetricNameDfs]") -> None:
        self.master: "MainFrame[_MetricNameSingle, _MetricNameDfs]" = master
        super().__init__(master, master.application, "edit axis config", resizeable=False)
        self.selectCfgLabel = tkinter.Label(self, text="select the axis to edit")
        self.cfgSelector = ComboboxLine(self, "axis choice: ", self.__getAxisChoices())
        self.openEditCfgButton = tkinter.Button(
            self, text="edit this config", command=self.openConfigEditor)
        self.selectCfgLabel.grid(row=0)
        self.cfgSelector.grid(row=1)
        self.openEditCfgButton.grid(row=2)
    
    def __getAxisChoices(self)->"list[str]":
        return list(self.application.plotConfig.axisConfigs.keys())
    
    def updatedConfig(self)->None:
        self.cfgSelector.setValues(self.__getAxisChoices())
    
    def openConfigEditor(self)->None:
        # determmine the axis to edit
        axisName: str = self.cfgSelector.getEntryText()
        self.application.mainFrame.openConfigEditor(axisName=axisName)
        self.destroy()
    
    @override
    def destroy(self) -> None:
        self.master.editAxisConfigDialog = None
        return super().destroy()

class RemoveAxisConfigDialog(CustomTopLevel[_MetricNameSingle, _MetricNameDfs]):
    def __init__(self, master:"MainFrame[_MetricNameSingle, _MetricNameDfs]") -> None:
        self.master: "MainFrame[_MetricNameSingle, _MetricNameDfs]" = master
        super().__init__(master, master.application, "remove axis config", resizeable=False)
        self.selectCfgLabel = tkinter.Label(self, text="select the axis to remove")
        self.cfgSelector = ComboboxLine(self, "axis choice: ", self.__getAxisChoices())
        self.openEditCfgButton = tkinter.Button(
            self, text="remove this config", command=self.removeAxis)
        self.selectCfgLabel.grid(row=0)
        self.cfgSelector.grid(row=1)
        self.openEditCfgButton.grid(row=2)
        
    def __getAxisChoices(self)->"list[str]":
        return list(self.application.plotConfig.axisConfigs.keys())
    
    def updatedConfig(self)->None:
        self.cfgSelector.setValues(self.__getAxisChoices())
    
    def removeAxis(self)->None:
        # determmine the axis to edit
        axisName: str = self.cfgSelector.getEntryText()
        self.application.createCheckPoint()
        with ErrorHandlerWithMessage("add axis config", self):
            self.application.plotConfig.removeAxisConfig(axisName)
        self.destroy()
    
    @override
    def destroy(self) -> None:
        self.master.removeAxisConfigDialog = None
        return super().destroy()



class AxisConfigEditor(CustomTopLevel[_MetricNameSingle, _MetricNameDfs]):
    def __init__(self, master:"MainFrame[_MetricNameSingle, _MetricNameDfs]", axisName:str) -> None:
        self.master: "MainFrame[_MetricNameSingle, _MetricNameDfs]" = master
        super().__init__(master, master.application, "axis config editor", resizeable=False)
        self.axisName: str = axisName
        """the name of the axis config that is targeted"""
        
        ## set the current config values
        currConfig = self.application.plotConfig.axisConfigs[self.axisName]
        self.newMetricsConfig: "AxisMetricsConfigs[_MetricNameSingle, _MetricNameDfs]" = \
            currConfig.metricsConfigs.copy()
        
        self.singleMetricsSelector: "SingleMetricsSelector[_MetricNameSingle]|None" = None
        self.dfsMetricsSelector: "DfsMetricsSelector[_MetricNameDfs]|None" = None
        self.singleMetricsConfigurator: "SingleMetricsConfigurator[_MetricNameSingle]|None" = None
        self.dfsMetricsConfigurator: "DfsMetricsConfigurator[_MetricNameDfs]|None" = None
        
        
        self.entrys = MultipleTextEntryLine(self, {
            "name": ("axis name: ", currConfig.name),
            "label": ("y label: ", currConfig.yLabel),
            "scale": ("y label: ", currConfig.scale),
            "hlines": ("horizontal lines (<empty> / y1, y2, ...): ", 
                       self.__hLinesToText(currConfig.hlines)),
            "limits": ("y limits ('auto' / yFrom -> yTo): ", 
                       self.__YLimitsToText(currConfig.yLimits)),
            "indexs": ("indexes (intFrom[, intTo]): ", 
                       self.__indexesToText(currConfig.indexs))})
        self.metricsSelectorButtons = ButtonsLine(master=self, placement="grid", buttonsInfos=[
            ("select single metrics", self.openSingleMetricsSelector),
            ("select dfs metrics", self.openDfsMetricsSelector)])
        self.metricsConfiguratorButtons = ButtonsLine(master=self, placement="grid", buttonsInfos=[
            ("configure single metrics lines", self.openSingleMetricsConfigurator),
            ("configure dfs metrics lines", self.openDfsMetricsConfigurator)])
        self.controlButtons = ButtonsLine(master=self, placement="grid", buttonsInfos=[
            ("applie changes", self.applieChanges), ("cancel", self.destroy)])
        
        self.entrys.grid(row=0)
        self.metricsSelectorButtons.grid(row=1)
        self.metricsConfiguratorButtons.grid(row=2)
        self.controlButtons.grid(row=3)
        

    def applieChanges(self)->None:
        self.application.createCheckPoint()
        with ErrorHandlerWithMessage("invalide entry", self):
            indexs: "int|tuple[int, ...]" = extractIndexs(self.entrys["indexs"])
            yLimits_text: str = self.entrys["limits"]
            yLimits: "Limits|None" = Limits.fromText(yLimits_text)
            scale: _Scales = AxisConfig.assertScale(self.entrys["scale"])
        axisMetricsConfig = AxisMetricsConfigs(
            disabledSingleMetrics=copy.deepcopy(self.newMetricsConfig.disabledSingleMetrics),
            disabledDfsMetrics=copy.deepcopy(self.newMetricsConfig.disabledDfsMetrics),
            singleLinesConfigs=copy.deepcopy(self.newMetricsConfig.singleLinesConfigs),
            dfsLinesConfigs=copy.deepcopy(self.newMetricsConfig.dfsLinesConfigs))
        newConfig = AxisConfig(
            name=self.entrys["name"], yLabel=self.entrys["label"], scale=scale,
            hlines=set(map(float, extractTextList(self.entrys["hlines"]))),
            metricsConfigs=axisMetricsConfig, indexs=indexs, yLimits=yLimits)
        
        self.application.plotConfig.replaceAxisConfig(
            oldAxisConfigName=self.axisName, newAxisConfig=newConfig)
        if newConfig.name != self.axisName:
            # => rename (close and reopen)
            self.destroy() # close 
            self.master.openConfigEditor(newConfig.name) # reopen new config

    def updatedConfig(self)->None:
        # => config has changed and this editor was oppened
        self.destroy() # close
        if self.axisName in self.application.plotConfig.axisConfigs.keys():
            # => still exists => reopen
            self.master.openConfigEditor(self.axisName)
        
    def openSingleMetricsSelector(self)->None:
        if self.singleMetricsSelector is not None:
            tkinter.messagebox.showinfo(
                title="single metrics selector dialog", 
                message="the single metrics selector is alredy opened")
            self.singleMetricsSelector.focus()
        else: # => not opened => open a new one
            self.singleMetricsSelector = SingleMetricsSelector(self)
        
    def openDfsMetricsSelector(self)->None:
        if self.dfsMetricsSelector is not None:
            tkinter.messagebox.showinfo(
                title="dfs metrics selector dialog", 
                message="the dfs metrics selector is alredy opened")
            self.dfsMetricsSelector.focus()
        else: # => not opened => open a new one
            self.dfsMetricsSelector = DfsMetricsSelector(self)
        
    def openSingleMetricsConfigurator(self)->None:
        if self.singleMetricsConfigurator is not None:
            tkinter.messagebox.showinfo(
                title="single metrics configurator dialog", 
                message="the single metrics configurator is alredy opened")
            self.singleMetricsConfigurator.focus()
        else: # => not opened => open a new one
            self.singleMetricsConfigurator = SingleMetricsConfigurator(self)
        
    def openDfsMetricsConfigurator(self)->None:
        if self.dfsMetricsConfigurator is not None:
            tkinter.messagebox.showinfo(
                title="dfs metrics configurator dialog", 
                message="the dfs metrics configurator is alredy opened")
            self.dfsMetricsConfigurator.focus()
        else: # => not opened => open a new one
            self.dfsMetricsConfigurator = DfsMetricsConfigurator(self)
    
    def destroy(self)->None:
        self.master.axisConfigEditors.pop(self.axisName)
        return super().destroy()

    def __indexesToText(self, indexs:"int|tuple[int, int]")->str:
        return (str(indexs) if isinstance(indexs, int) 
                else ", ".join(map(str, indexs)))
    def __YLimitsToText(self, yLimits:"Limits|None")->str:
        return ("auto" if yLimits is None else str(yLimits))
    def __hLinesToText(self, hlines:"set[float]")->str:
        return ", ".join(map(str, hlines))




class Menus(tkinter.Menu, Generic[_MetricNameSingle, _MetricNameDfs]):
    def __init__(self, app:"App[_MetricNameSingle, _MetricNameDfs]") -> None:
        super().__init__(app)
        self.application: "App[_MetricNameSingle, _MetricNameDfs]" = app
        self.figureConfigEditor: "FigureConfigEditor[_MetricNameSingle, _MetricNameDfs]|None" = None
        
        # fileSubMenu
        self.fileSubMenu = tkinter.Menu(self)
        self.fileSubMenu.add_command(
            label="New", command=self.application.newConfig, accelerator="Ctrl+N")
        self.fileSubMenu.add_command(
            label="Open", command=self.application.openFromFile, accelerator="Ctrl+O")
        self.fileSubMenu.add_command(
            label="Save", command=self.application.saveToFile, accelerator="Ctrl+S")
        self.fileSubMenu.add_command(
            label="Save as", command=self.application.saveAsToFile, accelerator="Ctrl+Shift+S")
        self.fileSubMenu.add_command(label="Exit", command=self.application.exit)
        self.add_cascade(menu=self.fileSubMenu, label="File")
        # editSubMenu
        self.editSubMenu = tkinter.Menu(self)
        self.editSubMenu.add_command(
            label="Edit figure config", command=self.startFigureConfigEditor)
        self.editSubMenu.add_command(
            label="Revert last change", command=self.application.undo, accelerator="Ctrl+Z")
        self.editSubMenu.add_command(
            label="Eedo changes", command=self.application.redo, accelerator="Ctrl+Y")
        self.add_cascade(menu=self.editSubMenu, label="Edit")
        
        # fileSubMenu
        self.application.bind("<Control-n>", func=lambda e: self.application.newConfig())
        self.application.bind("<Control-o>", func=lambda e: self.application.openFromFile())
        self.application.bind("<Control-s>", func=lambda e: self.application.saveToFile())
        self.application.bind("<Control-S>", func=lambda e: self.application.saveAsToFile())
        # editSubMenu
        self.application.bind("<Control-z>", func=lambda e: self.application.undo())
        self.application.bind("<Control-y>", func=lambda e: self.application.redo())

    def startFigureConfigEditor(self)->None:
        if self.figureConfigEditor is not None:
            tkinter.messagebox.showinfo(
                title="edit axis config dialog", 
                message="the edit axis config dialog is alredy opened")
            self.figureConfigEditor.focus()
        else: # => not opened => open a new one
            self.figureConfigEditor = FigureConfigEditor(self)

    def updatedConfig(self)->None:
        if self.figureConfigEditor is not None:
            self.figureConfigEditor.updatedConfig()



class FigureConfigEditor(CustomTopLevel[_MetricNameSingle, _MetricNameDfs]):
    def __init__(self, menusWidget:"Menus[_MetricNameSingle, _MetricNameDfs]") -> None:
        super().__init__(menusWidget, menusWidget.application, title="figure config editor")
        self.menusWidget: "Menus[_MetricNameSingle, _MetricNameDfs]" = menusWidget
        
        figCfg = self.application.plotConfig.figureConfig
        self.entrys: "MultipleTextEntryLine[_FigureConfigField]" = \
            MultipleTextEntryLine(master=self, fixedTextToDefaultText={
                "figureID": ("figure ID: ", figCfg.figureID),
                "nbRows": ("number of rows: ", str(figCfg.nbRows)),
                "nbCols": ("number of cols: ", str(figCfg.nbCols)),
                "plotSize": ("size of plot: ", str(figCfg.plotSize)),
            }); del figCfg
        self.validateButton = tkinter.Button(self, text="validate", bg="maroon1", command=self.saveConfig)
        
        # place each entry line
        self.entrys.grid(row=0, sticky="we")
        self.validateButton.grid(row=1, sticky="we")
        
    def __getFigureConfig(self)->"FigureConfig":
        return FigureConfig.fromText(datas=self.entrys.getEntryTexts())
    
    @override
    def destroy(self)->None:
        """unbind it and destroy the window"""
        self.menusWidget.figureConfigEditor = None
        super().destroy()
    
    def saveConfig(self)->None:
        # get the new Configuration
        # swap the config in the datas with the new one and update the app
        self.application.createCheckPoint()
        self.application._loadPlotConfig(
            self.application.plotConfig.replaceFigCfg(self.__getFigureConfig()))
        self.destroy()

    def updatedConfig(self)->None:
        figCfg = self.application.plotConfig.figureConfig
        self.entrys.entryLines["figureID"].var.set(figCfg.figureID)
        self.entrys.entryLines["nbRows"].var.set(str(figCfg.nbRows))
        self.entrys.entryLines["nbCols"].var.set(str(figCfg.nbCols))
        self.entrys.entryLines["plotSize"].var.set(str(figCfg.plotSize))





class SingleMetricsSelector(CustomTopLevel[_MetricNameSingle, Any]):
    
    def __init__(self, axisCfgEditor:"AxisConfigEditor[_MetricNameSingle, Any]")->None:
        self.axisCfgEditor: "AxisConfigEditor[_MetricNameSingle, Any]" = axisCfgEditor
        super().__init__(axisCfgEditor, axisCfgEditor.application, 
                         f"{axisCfgEditor.axisName}: single metrics selector")
        self.checkboxes = CheckableManyFrame(self, items=self._getMetrics(), nbRows=4)
        for singleMetric in self.newConfig.getSelectedSingleMetrics():
            self.checkboxes.setState(singleMetric, newState=True)
        self.editor = CheckableFrameTextEditor(self, self.checkboxes, "single metrics selected:")
        self.controlButtons = ButtonsLine(master=self, placement="grid", buttonsInfos=[
            ("applie the metrics", self.applieSelection), ("cancel", self.destroy)])
        
        self.checkboxes.grid(column=0)
        self.editor.grid(column=1)
        self.controlButtons.grid(column=2)
    
    @property
    def newConfig(self)->"AxisMetricsConfigs[_MetricNameSingle, Any]":
        return self.axisCfgEditor.newMetricsConfig
    
    def applieSelection(self)->None:
        self.newConfig.updateSelectedSingleMetrics(
            self.checkboxes.getSelected())
        self.destroy()
        
    def destroy(self) -> None:
        self.axisCfgEditor.singleMetricsSelector = None
        return super().destroy()
    
    def _getMetrics(self)->"set[_MetricNameSingle]":
        return set(self.application.datas.singleMetrics.metrics)



class DfsMetricsSelector(CustomTopLevel[Any, _MetricNameDfs]):
    
    def __init__(self, axisCfgEditor:"AxisConfigEditor[Any, _MetricNameDfs]")->None:
        self.axisCfgEditor: "AxisConfigEditor[Any, _MetricNameDfs]" = axisCfgEditor
        super().__init__(axisCfgEditor, axisCfgEditor.application, 
                         f"{axisCfgEditor.axisName}: dfs metrics selector")
        self.checkboxes = CheckableManyFrame(self, items=self._getMetrics(), nbRows=4)
        for dfsMetric in self.newConfig.getSelectedDfsMetrics():
            self.checkboxes.setState(dfsMetric, newState=True)
        self.editor = CheckableFrameTextEditor(self, self.checkboxes, "dfs metrics selected:")
        self.controlButtons = ButtonsLine(master=self, placement="grid", buttonsInfos=[
            ("applie the metrics", self.applieSelection), ("cancel", self.destroy)])
        
        self.checkboxes.grid(column=0)
        self.editor.grid(column=1)
        self.controlButtons.grid(column=2)
    
    @property
    def newConfig(self)->"AxisMetricsConfigs[Any, _MetricNameDfs]":
        return self.axisCfgEditor.newMetricsConfig
    
    def applieSelection(self)->None:
        self.newConfig.updateSelectedDfsMetrics(self.checkboxes.getSelected())
        self.destroy()
        
    def destroy(self)->None:
        self.axisCfgEditor.dfsMetricsSelector = None
        return super().destroy()
    
    def _getMetrics(self)->"set[_MetricNameDfs]":
        return set(self.application.datas.dfsMetrics.metrics)





class SingleMetricsConfigurator(CustomTopLevel[_MetricNameSingle, Any]):
    def __init__(self, master:"AxisConfigEditor[_MetricNameSingle, Any]")->None:
        self.master: "AxisConfigEditor[_MetricNameSingle, Any]"
        super().__init__(master, master.application, title="single metrics lines configurator")
        with ErrorHandlerWithMessage("failed to open config editor", self.master):
            self.editor = LineConfigsEditor(
                self, self.master.newMetricsConfig.getSelectedSingleMetrics(), 
                None, self.applieConfigs)
        self.editor.pack()
        for metric, metricLineCfgs in self.master.newMetricsConfig.singleLinesConfigs.items():
            # don't use getConfigs => it gives the default when empty
            for cfg in metricLineCfgs.configs: 
                self.editor.addFrame(metric, cfg)
    
    def applieConfigs(self, configs:"_Configs[_MetricNameSingle]")->None:
        # assert the config format
        configsChecked: "list[tuple[_MetricNameSingle, LineConfig]]" = []
        for cfgName, cfg in configs:
            assert isinstance(cfgName, str)
            configsChecked.append((cfgName, cfg))
        # clear the existing configs
        self.master.newMetricsConfig.clearSingleMetricsConfigs()
        # add all the active configs
        self.master.newMetricsConfig.addSingleMetricsConfigs(configsChecked)
    
    def destroy(self)->None:
        self.master.singleMetricsConfigurator = None
        return super().destroy()
        
class DfsMetricsConfigurator(CustomTopLevel[Any, _MetricNameDfs]):
    ALL_DFS: _ALL_DFS = "ALL_DFS"
    
    def __init__(self, master:"AxisConfigEditor[Any, _MetricNameDfs]")->None:
        self.master: "AxisConfigEditor[Any, _MetricNameDfs]"
        super().__init__(master, master.application, title="dfs metrics lines configurator")
        with ErrorHandlerWithMessage("failed to open config editor", self.master):
            self.editor = LineConfigsEditor(
                self, self.master.newMetricsConfig.getSelectedDfsMetrics(), 
                set(self.application.datas.dfsMetrics.values.keys()), self.applieConfigs)
        self.editor.pack()
        for metric, metricLineCfgs in self.master.newMetricsConfig.dfsLinesConfigs.items():
            # cfgs specific to a dfKey
            for dfKey, dfConfigs in metricLineCfgs.perKeyConfig.items():
                for cfg in dfConfigs:
                    self.editor.addFrame((metric, dfKey), cfg)
            # cfgs generic to all dfKeys
            for cfg in metricLineCfgs.allKeysConfig:
                self.editor.addFrame((metric, self.ALL_DFS), cfg)
                
    
    def applieConfigs(self, configs:"_Configs[_MetricNameDfs]")->None:
        # assert the config format
        configsChecked: "list[tuple[_MetricNameDfs, _DfKey|None, LineConfig]]" = []
        for cfgName, cfg in configs:
            assert isinstance(cfgName, tuple)
            metric, dfkey = cfgName
            if dfkey == self.ALL_DFS:
                dfkey = None
            configsChecked.append((metric, dfkey, cfg))
        # clear the existing configs
        self.master.newMetricsConfig.clearDfsMetricsConfigs()
        # add all the active configs
        self.master.newMetricsConfig.addDfsMetricsConfigs(configsChecked)
    
    def destroy(self)->None:
        self.master.dfsMetricsConfigurator = None
        return super().destroy()


class LineConfigFrame(tkinter.Frame):
    def __init__(self, lineCfgEditor:"LineConfigsEditor", 
                 frameID:int, cfgName:"_CFG_NAME", currentCfg:"LineConfig")->None:
        super().__init__(master=lineCfgEditor.linesFramesHolder.scrollable_frame)
        self.lineCfgEditor: "LineConfigsEditor" = lineCfgEditor
        self.frameID: int = frameID
        self.cfgName: "_CFG_NAME" = cfgName
        
        # configure the different entrys and buttons
        self.metricLabel = tkinter.Label(self, text=self.getName())
        self.enabledBox = CheckableEntryLine(
            self, fixedText="enabled : ", defaultState=currentCfg.enabled)
        self.colorEntry = TextEntryLine(self, "color (empty/name/#hex): ", (currentCfg.color or ""))
        self.lineStyle = OptionMenuLine(
            self, "line style: ", get_args_LiteralString(_LineStyle), currentCfg.lineStyle)
        self.pointStyleSelector = OptionMenuLine(
            self, "point style: ", get_args_LiteralString(_PointStyle), currentCfg.pointStyle)
        self.lineWidthSelector = TextEntryLine(self, "line width (float): ", str(currentCfg.lineWidth))
        self.emaCoeffEntry = TextEntryLine(
            self, "ema coeff (empty/float): ", str(currentCfg.emaCoeff or ""))
        self.removeButton = tkinter.Button(self, text="remove line", command=self.destroy)
        # place everything
        self.metricLabel.pack(side="top", fill="x")
        self.enabledBox.pack(side="top", fill="x")
        self.colorEntry.pack(side="top", fill="x")
        self.lineStyle.pack(side="top", fill="x")
        self.pointStyleSelector.pack(side="top", fill="x")
        self.lineWidthSelector.pack(side="top", fill="x")
        self.emaCoeffEntry.pack(side="top", fill="x")
        self.removeButton.pack(side="top", fill="x")
    
    def getName(self)->str:
        if isinstance(self.cfgName, str): 
            return self.cfgName
        metric, dfKey = self.cfgName
        return f"{metric}~{dfKey}"
    
    def getLineConfig(self)->"LineConfig":
        with ErrorHandlerWithMessage("invalid line config", self):
            colorStr = self.colorEntry.getEntryText()
            try: color = (None if colorStr == "" else colorStr)
            except: raise ValueError(f"invalid convertion of the emaCoeff from: {colorStr}")
            try: lineWidth = float(self.lineWidthSelector.getEntryText())
            except: raise ValueError(f"invalid convertion of the lineWidth from: {self.lineWidthSelector.getEntryText()}")
            emaCoeffStr = self.emaCoeffEntry.getEntryText()
            try: emaCoeff = (None if emaCoeffStr == "" else float(emaCoeffStr))
            except: raise ValueError(f"invalid convertion of the emaCoeff from: {emaCoeffStr}")
        return LineConfig(
            enabled=self.enabledBox.getState(),
            color=color, lineWidth=lineWidth, emaCoeff=emaCoeff,
            lineStyle=self.lineStyle.getSelected(), 
            pointStyle=self.pointStyleSelector.getSelected())
    
    @override
    def destroy(self)->None:
        self.lineCfgEditor.removeFrame(self.frameID)
        super().destroy()
        


class LineConfigsEditor(tkinter.Frame, Generic[_Field]):
    def __init__(self, master:"tkinter.Misc", metricsChoices:"set[_Field]", dfKeys:"set[_DfKey]|None", 
                 applieFunc:"Callable[[_Configs[_Field]], None]")->None:
        """applieFunc will be given all the configs, and should destroy the LineConfigsEditor"""
        super().__init__(master)
        if len(metricsChoices) == 0:
            raise IndexError("there are no metrics to add configs to, please add some metrics first")        
        self.applieFunc: "Callable[[_Configs[_Field]], None]" = applieFunc
        self.__lastFrameID: int = 0
        self.dfKeysChoice:"set[_DfKey]|None" = dfKeys
        self.configFrames: "dict[int, LineConfigFrame]" = {}
        self.linesFramesHolder = ScrollableFrame(
            self, scrollSides="horizontal", width=350, height=250)
        # buttons
        self.addConfigWidgets = WidgetsLine(self)
        
        self.metricsSelector = OptionMenuLine(
            self.addConfigWidgets, "metric to add config to: ", sorted(metricsChoices))
        self.addConfigWidgets.addWidgets(self.metricsSelector)
        
        self.dfKeySelector: "OptionMenuLine|None" = None
        if self.dfKeysChoice is not None:
            self.dfKeySelector = OptionMenuLine[Any](
                self.addConfigWidgets, fixedText="dfs to add config to: ", 
                values=[DfsMetricsConfigurator.ALL_DFS] + sorted(self.dfKeysChoice))
            self.addConfigWidgets.addWidgets(self.dfKeySelector)
        
        self.addConfigWidgets.addWidgets(
            tkinter.Button(self.addConfigWidgets, text="add new lineConfig", command=self.addEmptyFrame))
        
        self.validateChanges = tkinter.Button(
            self, text="validate all changes", command=self.saveConfig)
        # place each entry line
        self.addConfigWidgets.placeWidgets("pack")
        self.linesFramesHolder.pack(fill="both", side="top")
        self.addConfigWidgets.pack(fill="y", side="bottom")
        self.validateChanges.pack(fill="y", side="bottom")
        self.placeFrames()
    
    
    def getSelectedMetricName(self)->"_Field":
        return self.metricsSelector.getSelected()
    
    def getSelectedDfKey(self)->"str|None":
        if self.dfKeySelector is None:
            return None
        return self.dfKeySelector.getSelected()
    
    def saveConfig(self)->None:
        newLinesConfigs: "_Configs[_Field]" = [
            (frame.cfgName, frame.getLineConfig())
            for frame in self.configFrames.values()]
        self.applieFunc(newLinesConfigs)
        # => applieFunc will destroy self
    
    def placeFrames(self)->None:
        renderOrder: "list[int]" = sorted(self.configFrames.keys())
        for index, frameID in enumerate(renderOrder):
            self.configFrames[frameID].grid(column=index, row=0, padx=5)
    
    def addEmptyFrame(self)->None:
        cfgName: "_CFG_NAME[_Field]" = self.getSelectedMetricName()
        if self.dfKeySelector is not None:
            # => a dfKey is selected, add it to the cfgName
            dfKey = self.getSelectedDfKey()
            assert dfKey is not None
            cfgName = (cfgName, dfKey)
        self.addFrame(cfgName, LineConfig.default())
        
    
    def addFrame(self, cfgName:"_CFG_NAME[_Field]", cfg:"LineConfig")->None:
        self.__lastFrameID = newFrameID = (self.__lastFrameID + 1)
        self.configFrames[newFrameID] = LineConfigFrame(
            lineCfgEditor=self, frameID=newFrameID, 
            cfgName=cfgName, currentCfg=cfg)
        self.placeFrames()
    
    def removeFrame(self, frameID:int)->None:
        self.configFrames.pop(frameID)
        self.placeFrames()
    
    def clearFrames(self)->None:
        self.configFrames.clear()
        self.placeFrames()



class ErrorHandlerWithMessage(SupportsContext):
    """show an error message with the error and it's traceback, don't suppress the error !"""
    
    def __init__(self, title:str, master:"tkinter.Misc")->None:
        self.master: "tkinter.Misc" = master
        self.title: str = title
    
    def __enter__(self)->Self:
        return self
    def __exit__(self, exc_type:"type[Exception]|None", exc_value:"Exception|None",
                 traceback:"TracebackType|None")->"None|bool":
        if exc_value is None:
            # => no error !
            return None
        # => error
        # compute the text of the err
        exceptionText: str
        if __debug__:
            strIo = StringIO()
            print_exception(exc_value.with_traceback(traceback), file=strIo)
            exceptionText = strIo.getvalue()
            del strIo
        else: exceptionText = '\n'.join(exc_value.args)
        # show the error
        message = (f"\t{self.title}\n\n"
                   f"the following error happend:\n\t{exceptionText}")
        showError(self.master, self.title, message)
        return False # -> don't supress the error


