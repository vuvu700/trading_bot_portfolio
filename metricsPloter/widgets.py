import attrs
import functools

import tkinter
import tkinter.ttk
import tkinter.messagebox
from tkinter.font import Font

from holo.__typing import (
    TYPE_CHECKING, Generic, TypeVar, 
    LiteralString, Iterable, Literal, 
    Sequence, Callable, Protocol, 
    Any, Dict, )
from holo import getDuplicated

if TYPE_CHECKING:
    from .app import App

_Field = TypeVar("_Field", bound=LiteralString)
_Field2 = TypeVar("_Field2", bound=LiteralString)
_Kwargs = Dict[str, Any]

_ScollSide = Literal["vertical", "horizontal", "both"]
    

class CustomTopLevel(tkinter.Toplevel, Generic[_Field, _Field2]):
    def __init__(self, master:"tkinter.Misc", application:"App[_Field, _Field2]", title:str,
                 resizeable:bool=False, posDelta:"tuple[int, int]"=(100, 100))->None:
        super().__init__(master=master, bg="lightskyblue")
        self.title(title)
        self.application: "App[_Field, _Field2]" = application
        self.resizable(resizeable, resizeable)
        # determine the position of the 
        selfX = self.master.winfo_x() + posDelta[0]
        selfY = self.master.winfo_y() + posDelta[1]
        self.geometry(f"+{selfX}+{selfY}")


class WidgetsLine(tkinter.Frame):
    def __init__(self, master:tkinter.Misc) -> None:
        super().__init__(master, bg="lightgoldenrod1")
        self.widgets: "list[tkinter.Widget]" = []
        
    def addWidgets(self, *widgets:"tkinter.Widget")->None:
        self.widgets.extend(widgets)
    
    def placeWidgets(self, packing:"Literal['grid', 'pack']", placementKwargs:"_Kwargs|None"=None)->None:
        for index, widget in enumerate(self.widgets):
            assert widget.master == self
            if packing == "pack":
                widget.pack(side=tkinter.LEFT, cnf=placementKwargs)
            elif packing == "grid":
                widget.grid(column=index, row=0, sticky="we", cnf=placementKwargs)
            else: raise ValueError(f"invalide packing methode: {packing}")

class ButtonsLine(WidgetsLine):
    def __init__(self, master:"tkinter.Misc", 
                 buttonsInfos:"list[tuple[str, Callable[[], None]]]",
                 placement:"Literal['grid', 'pack']", 
                 placementKwargs:"_Kwargs|None"=None)->None:
        super().__init__(master)
        self.addWidgets(*(
            tkinter.Button(master=self, text=text, command=command)
            for (text, command) in buttonsInfos))
        self.placeWidgets(placement, placementKwargs)


    



class ComboboxLine(tkinter.Frame):
    def __init__(self, master:tkinter.Misc, fixedText:str, values:"list[str]")->None:
        # set the attributs
        super().__init__(master, bg="lightskyblue")
        
        self.var = tkinter.StringVar(self, value=values[0])
        
        # create the widgets
        self.fixedLabel = tkinter.Label(self, text=fixedText, bg="cyan")
        self.comboBox = tkinter.ttk.Combobox(self, textvariable=self.var, values=values)
        self.comboBox["state"] = "readonly"
        
        # configur widgets placement
        self.grid_columnconfigure(0, weight=1) # label
        self.grid_columnconfigure(1, weight=0) # comboBox
        self.fixedLabel.grid(column=0, row=0, sticky="w")
        self.comboBox.grid(column=1, row=0, sticky="e")
        
    def getEntryText(self)->str:
        return self.var.get()

    def setValues(self, values:"list[str]")->None:
        if self.getEntryText() not in self.comboBox["values"]:
            # => old selected value ins't in the new values
            self.var.set(values[0])
        self.comboBox["values"] = values


class OptionMenuLine(tkinter.Frame, Generic[_Field]):
    def __init__(self, master:tkinter.Misc, fixedText:str, values:"Sequence[_Field]",
                 initialValue:"_Field|None"=None, 
                 selectionCallback:"Callable[[_Field], None]|None"=None)->None:
        super().__init__(master, bg="lightskyblue")
        self.selectionCallback: "Callable[[_Field], None]|None" = selectionCallback
        self.fields: "set[_Field]" = set(values)
        assert len(self.fields) == len(values), \
            ValueError(f"there are duplicates fields in `values`: {getDuplicated(values)}")
        self.fixedLabel = tkinter.Label(self, text=fixedText, bg="cyan")
        self.var = tkinter.StringVar(self)
        self.optionMenu = tkinter.OptionMenu(self, self.var, *values, command=self.__selectionCallback)
        if initialValue is None: initialValue = values[0]
        assert initialValue in self.fields, "the value isn't in the `values`"
        self.var.set(initialValue)
        
        # configur widgets placement
        self.grid_columnconfigure(0, weight=1) # label
        self.grid_columnconfigure(1, weight=0) # OptionMenu
        self.fixedLabel.grid(column=0, row=0, sticky="w")
        self.optionMenu.grid(column=1, row=0, sticky="e")
    
    def setValues(self, values:"list[_Field]")->None:
        if self.getSelected() not in self.optionMenu["values"]:
            # => old selected value ins't in the new values
            self.var.set(values[0])
        self.optionMenu['menu'].delete(0, 'end')
        for choice in values:
            self.optionMenu['menu'].add_command(
                label=choice, command=tkinter._setit(self.var, choice))
    
    def __assertValue(self, v:str)->"_Field":
        assert v in self.fields
        return v
    
    def getSelected(self)->"_Field":
        return self.__assertValue(self.var.get())
    
    def __selectionCallback(self, var:"tkinter.StringVar")->None:
        if self.selectionCallback is None: return
        self.selectionCallback(self.__assertValue(var.get()))
        



class CheckableManyFrame(tkinter.Frame, Generic[_Field]):
    
    def __init__(self, master:tkinter.Misc, items:"set[_Field]", 
                 nbRows:"int|None", checkedCallback:"None|Callable[[_Field, bool], None]"=None)->None:
        super().__init__(master=master, bg=master["background"])
        self.__nbRowsTarget: "int|None" = nbRows
        self.__checkedCallback: "None|Callable[[_Field, bool], None]" = checkedCallback
        
        self.textLabel = tkinter.Label(
            self, text="select the options: ", bg=self["background"])
        self.buttonsFrame = tkinter.Frame(self, bg=self["background"])
        self.textLabel.pack(fill="x", anchor="center", side="top")
        self.buttonsFrame.pack(fill="both", anchor="center", side="top")
        
        self.checkBoxes: "dict[_Field, tkinter.Checkbutton]" = {}
        self.boxesValue: "dict[_Field, tkinter.BooleanVar]" = {}
        self.setFields(items)
    
    @property
    def fields(self)->"set[_Field]":
        return set(self.boxesValue.keys())
    
    def setState(self, field:"_Field", newState:bool)->None:
        self.boxesValue[field].set(newState)
    
    def getSelected(self)->"set[_Field]":
        return {field for field in self.fields if self.isChecked(field)}
    
    def isChecked(self, field:"_Field")->bool:
        return self.boxesValue[field].get()
    
    def _getNbRow(self, nbFields:int)->int:
        if self.__nbRowsTarget is None:
            return int(nbFields ** 0.5)
        elif self.__nbRowsTarget > nbFields:
            return nbFields
        return self.__nbRowsTarget
    
    def setFields(self, fields:"Iterable[_Field]")->None:
        """generate the buttons of the activities (keep their state)"""
        if not isinstance(fields, set):
            fields = set(fields)
        NB_ROWS: int = self._getNbRow(len(fields))
        selectedFields: "set[_Field]" = self.getSelected()
        currentFields: "set[_Field]" = self.fields
        fieldsToRemove: "set[_Field]" = currentFields.difference(fields)
        """all activity in `currentActivities` but not in `activities`"""
        # clear the current activities items
        for field in fieldsToRemove:
            button = self.checkBoxes.pop(field)
            self.boxesValue.pop(field)
            button.destroy()
            del field, button
        # add the new activities
        for index, field in enumerate(sorted(fields)):
            # create the tkinter elements
            if field not in self.boxesValue:
                var = tkinter.BooleanVar(value=(field in selectedFields))
                button = tkinter.Checkbutton(
                    self.buttonsFrame, text=field,
                    bg=self.buttonsFrame["background"],
                    variable=var, onvalue=True, offvalue=False, 
                    command=self.__getCheckedCommand(field))
                # bind them to the structure
                self.boxesValue[field] = var
                self.checkBoxes[field] = button
                del button, var
            # place the button
            column, row = divmod(index, NB_ROWS)
            self.checkBoxes[field].grid(
                row=row, column=column, sticky="nw")
            del field, row, column
    
    def __runCheckedCallback(self, field)->None:
        assert self.__checkedCallback is not None
        return self.__checkedCallback(field, self.isChecked(field))
         
    def __getCheckedCommand(self, field:"_Field")->"Literal['']|Callable[[], None]":
        if self.__checkedCallback is None: return ""
        return functools.partial(self.__runCheckedCallback, field)

class CheckableFrameTextEditor(tkinter.Frame, Generic[_Field]):
    
    def __init__(self, master:"tkinter.Misc", 
                 checkableFrame:"CheckableManyFrame[_Field]", fixedtext:str)->None:
        super().__init__(master=master, bg=master["background"])
        self.checkableFrame: "CheckableManyFrame[_Field]" = checkableFrame
        self.textEntry = TextEntryLine(
            self, fixedText=fixedtext, defaultEntryText=self.__getSelectionText())
        self.controlButtons = WidgetsLine(self)
        self.controlButtons.addWidgets(
            tkinter.Button(self.controlButtons, text="generate the selection", command=self.generate),
            tkinter.Button(self.controlButtons, text="applie the selection", command=self.applie))
        self.controlButtons.placeWidgets("grid", )
        self.textEntry.grid_configure(row=0, column=0)
        self.controlButtons.grid_configure(row=1, column=0)
    
    def generate(self)->None:
        """set the selection text with all the selected fields"""
        self.textEntry.var.set(self.__getSelectionText())
    
    def applie(self)->None:
        allFields: "set[_Field]" = self.checkableFrame.fields
        def assertIsAField(field:str)->"_Field":
            if field not in allFields:
                raise ValueError(f"the field: {repr(field)} don't exist")
            return field
        fieldsSelection: "set[_Field]" = \
            {assertIsAField(field.strip()) 
             for field in self.textEntry.getEntryText().split(",")
             if field != ''}
        for field in allFields:
            self.checkableFrame.setState(field, newState=(field in fieldsSelection))
        
    def __getSelectionText(self)->str:
        return ", ".join(self.checkableFrame.getSelected())


class TextEntryLine(tkinter.Frame):
    def __init__(self, master:"tkinter.Misc",
                 fixedText:str, defaultEntryText:str)->None:
        # set the attributs
        super().__init__(master, bg="lightskyblue")
        
        self.fixedText: str = fixedText
        self.var = tkinter.StringVar(self, value=defaultEntryText)
        
        # create the widgets
        self.fixedLabel = tkinter.Label(self, text=self.fixedText, bg="cyan")
        self.entryLabel = tkinter.Entry(self, textvariable=self.var, bg="yellow")
        
        # configur widgets placement
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0) # start
        self.fixedLabel.grid(column=0, row=0, sticky="w")
        self.entryLabel.grid(column=1, row=0, sticky="w")
        
    def getEntryText(self)->str:
        return self.var.get()

class CheckableEntryLine(tkinter.Frame):
    def __init__(self, master:"tkinter.Misc",
                 fixedText:str, defaultState:bool,
                 checkBoxText:str="")->None:
        # set the attributs
        super().__init__(master, bg="lightskyblue")
        
        self.fixedText: str = fixedText
        self.checkBoxText: str = checkBoxText
        self.var = tkinter.BooleanVar(self, value=defaultState)
        
        # create the widgets
        self.fixedLabel = tkinter.Label(self, text=self.fixedText, bg="cyan")
        self.entryLabel = tkinter.Checkbutton(
            self, text=self.checkBoxText, bg=self["background"],
            variable=self.var, onvalue=True, offvalue=False)
        
        # configur widgets placement
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0) # start
        self.fixedLabel.grid(column=0, row=0, sticky="w")
        self.entryLabel.grid(column=1, row=0, sticky="w")
        
    def getState(self)->bool:
        return self.var.get()


class MultipleTextEntryLine(tkinter.Frame, Generic[_Field]):
    def __init__(self, master:"tkinter.Misc", 
                 fixedTextToDefaultText:"dict[_Field, tuple[str, str]]")->None:
        super().__init__(master, bg="lightskyblue")
        self.entryLines: "dict[_Field, TextEntryLine]" = {
            field: TextEntryLine(master=self, fixedText=fixedText,
                          defaultEntryText=defaultText)
            for field, (fixedText, defaultText) in fixedTextToDefaultText.items()}
        # place each lines
        self.grid_columnconfigure(0, weight=1)
        for row, entryLine in enumerate(self.entryLines.values()):
            entryLine.grid(column=0, row=row, sticky="we")
    
    def getSingleEntryText(self, field:"_Field")->str:
        return self.entryLines[field].getEntryText()
    
    def getEntryTexts(self)->"dict[_Field, str]":
        return {field: entryLine.getEntryText()
                for field, entryLine in self.entryLines.items()}
    
    def __getitem__(self, field:"_Field")->str:
        return self.getSingleEntryText(field=field)

class FontsManger():
    def __init__(self, master:"tkinter.Misc", normal:"None|Font",
                 small:"None|Font"=None, big:"None|Font"=None) -> None:
        self.normal: Font = normal or Font(root=master)
        self.small: Font = small or self.getResizedCopy(self.normal, 0.75)
        self.big: Font = big or self.getResizedCopy(self.normal, 1.25)
        self.smaller: Font = self.getResizedCopy(self.small, 0.75)
        self.biger: Font = self.getResizedCopy(self.big, 1.25)
    
    @staticmethod
    def getResizedCopy(font: Font, newSize:"float")->Font:
        newFont: Font = font.copy()
        newFont.configure(size=int(newSize*font.cget("size")))
        return newFont



class ScrollableFrame(tkinter.ttk.Frame):
    def __init__(self, master:"tkinter.Misc", scrollSides:"_ScollSide",
                 width:int, height:int):
        super().__init__(master, height=height, width=width)
        self.__sides: "_ScollSide" = scrollSides
        self._canvas = tkinter.Canvas(self)
        self.scrollable_frame = tkinter.ttk.Frame(self._canvas)
        """where to add the childrens"""
        self.scrollable_frame.bind(
            "<Configure>", 
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        if self.scrollSides in ("vertical", "both"):
            self._verticalScrollbar = tkinter.Scrollbar(
                self, orient="vertical", command=self._canvas.yview)
            self._canvas.configure(xscrollcommand=self._verticalScrollbar.set)
            self._verticalScrollbar.pack(side="right", fill="y")
        if self.scrollSides in ("horizontal", "both"):
            self._horizontalScrollbar = tkinter.Scrollbar(
                self, orient="horizontal", command=self._canvas.xview)
            self._canvas.configure(xscrollcommand=self._horizontalScrollbar.set)
            self._horizontalScrollbar.pack(side="bottom", fill="x")
        self._canvas.pack(anchor="nw", fill="both", expand=True)
    
    @property
    def scrollSides(self)->"_ScollSide":
        return self.__sides
    