import sys, pathlib; sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

import numpy
import random
import metricsPloter
from AI.senti3.datas.types_config import AllMetrics

from holo.__typing import Literal


_SingleMetrics = Literal["loss", "loss_val", "nbIters", "time"]
_DfsMetrics = Literal["loss", "loss_val", "nbIters"]

customLineConf1 = metricsPloter.LineConfig(
    color="red", lineStyle="dotted", pointStyle="square", 
    lineWidth=1.0, emaCoeff=0.05, enabled=True)

customLineConf2 = metricsPloter.LineConfig(
    color="#ff00ff", lineStyle="dashed", pointStyle="circle", 
    lineWidth=2.0, emaCoeff=None, enabled=True)

customLineConf3 = metricsPloter.LineConfig(
    color=None, lineStyle="solid", pointStyle="None", 
    lineWidth=5.0, emaCoeff=0.5, enabled=True)

figureConfig = metricsPloter.FigureConfig(
    figureID="fig", nbCols=2, nbRows=2, plotSize=(10, 10))
config = metricsPloter.PlotConfig[_SingleMetrics, _DfsMetrics].empty(figureConfig)

config.addAxisConfig(metricsPloter.AxisConfig[_SingleMetrics, _DfsMetrics](
    name="ax1", yLabel="loss", scale="log", hlines={0.1, 0.5, 0.9}, indexs=(1, 2),
    metricsConfigs=metricsPloter.AxisMetricsConfigs[_SingleMetrics, _DfsMetrics](
        disabledSingleMetrics=set(), disabledDfsMetrics=set(), dfsLinesConfigs={},
        singleLinesConfigs={"loss": metricsPloter.SingleMetricConfig([customLineConf1]), 
                            "loss_val":metricsPloter.SingleMetricConfig(
                                [customLineConf1, customLineConf2])}),
    yLimits=metricsPloter.Limits(mini=0, maxi=1)))

config.addAxisConfig(metricsPloter.AxisConfig[_SingleMetrics, _DfsMetrics](
    name="ax2", yLabel="time", scale="linear",
    metricsConfigs=metricsPloter.AxisMetricsConfigs[_SingleMetrics, _DfsMetrics](
        disabledSingleMetrics=set(), disabledDfsMetrics={"loss_val"},
        singleLinesConfigs={},
        dfsLinesConfigs={"nbIters": metricsPloter.DfsMetricConfig([customLineConf3], {}), 
                         "loss_val": metricsPloter.DfsMetricConfig(
                             [customLineConf1, customLineConf2], {})}),
    hlines=set(), indexs=(3, 4), yLimits=None))


datas = AllMetrics[_SingleMetrics, _DfsMetrics](
    {"loss", "loss_val", "nbIters", "time"}, {"loss", "loss_val", "nbIters"}, None)

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


app = metricsPloter.App(baseConfig=config, datas=datas, verbose=False)
app.mainloop()