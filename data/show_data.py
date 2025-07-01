from typing import Callable, Any
import pandas as pd
import matplotlib.pyplot as plt



def plot_DataFrame(dataset:pd.DataFrame, columns:"list[str]|pd.Index|None"=["Open", "Close", "High", "Low"],
                   plotsize:"tuple[int|float, int|float]|None"=None, useDefaultColor:bool=True,
                   normFunctions:"dict[str, Callable]"=dict(), linewidth:float=1.0,
                   hlines:"None|list[float|int]|float|int"=None )->None:
    COLORS = {"Open":"blue", "Close":"orange", "High":"green", "Low":"red", "Volume":"gray"}
    COLORS2 = ["red", "green", "blue", "orange", "darkviolet", "lime", "yellow", "cyan", "deepskyblue", "chocolate", "navy", "hotpink"]

    if columns is None:
        columns = dataset.columns
    if not isinstance(columns, list):
        columns = list(map(str, columns))

    if hlines is not None:
        if isinstance(hlines, (float, int)):
            plt.axhline(hlines, linewidth=linewidth)
        else:
            for yHline in hlines:
                plt.axhline(yHline, linewidth=linewidth)

    for index, col in enumerate(columns):
        if col in dataset.keys():
            func = normFunctions.get(col, lambda x:x)
            color = COLORS.get(col, COLORS2[index]) if useDefaultColor is True else COLORS2[index]
            func(dataset[col]).plot(color=color, legend=col, figsize=plotsize, linewidth=linewidth)



def plot_dataFrame_multipleSubplots(dataset:pd.DataFrame, listeColumns:"list[list[str]]",
                   listLimitsY:"list[None|tuple[float|None, float|None]]|None"=None,
                   normFunctions:"dict[str, Callable]"=dict(), linewidth:float=1.,
                   listHlines:"None|list[None|float|int|list[float|int]]"=None,
                   plotsize:"tuple[int|float, int|float]|None"=None)->None:
    if len(listeColumns) > 0:

        ax1 = plt.subplot(100 * len(listeColumns) + 11)
        if (listLimitsY is not None) and (listLimitsY[0] is not None):
            ax1.set_ylim( *listLimitsY[0] ) # type:ignore because the error is due to incorect implementation
        hlines = None if ((listHlines is None) or (listHlines[0] is None)) else listHlines[0]
        plot_DataFrame(dataset=dataset, columns=listeColumns[0], plotsize=plotsize, useDefaultColor=False,
                       normFunctions=normFunctions, linewidth=linewidth, hlines=hlines)
        ID = 1

        for columns in listeColumns[1:]:
            axN = plt.subplot(100 * len(listeColumns) + 11 + ID , sharex=ax1)
            hlines = None if ((listHlines is None) or (listHlines[ID] is None)) else listHlines[ID]
            if (listLimitsY is not None) and (listLimitsY[ID] is not None):
                axN.set_ylim( *listLimitsY[ID] ) # type:ignore because the error is due to incorect implementation
            plot_DataFrame(dataset=dataset, columns=columns, plotsize=plotsize, useDefaultColor=False,
                           normFunctions=normFunctions, linewidth=linewidth, hlines=hlines)
            ID+=1

