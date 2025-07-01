if __name__ == "__main__":
    raise ImportError("the file must be imported from main dir")

import keras
import keras.callbacks


from ..datas.convergence import ConvergeConfig
from ..datas.types_config import TrainCallHist
from ..datas.datas_types import (
    Datas_training_Generator, Datas_dataset, _T_DfKey, )
from .baseAI import (
    BaseAI, _MetricNameSingle, _MetricNameDfs, 
    _AI_Verbose, convertVerbose, )
from ..training.callbacks import MesureFittingTimes

from holo.__typing import (
    Literal, Union, get_args, cast, getLiteralArgs,
    assertListSubType, assertIsinstance, )



_baseTrainSingleMetrics = Literal["loss_train", "loss_val", "nbTrainSteps", "nbValSteps",
                                  "trainTime", "valTime", ]
_baseTrainDfsMetrics = Literal["loss_train", "loss_val", "nbTrainSteps", "nbValSteps", ]
class AI_SupportsBaseTrain(
        BaseAI[Union[_MetricNameSingle, _baseTrainSingleMetrics],
               Union[_MetricNameDfs, _baseTrainDfsMetrics]]):
    
    @classmethod
    def getRequiredSingleMetrics(cls)->"set[_baseTrainSingleMetrics]":
        return set(get_args(_baseTrainSingleMetrics))
    @classmethod
    def getRequiredDfsMetrics(cls)->"set[_baseTrainDfsMetrics]":
        return set(get_args(_baseTrainDfsMetrics))
    
    

    def autoSetLearningRate(self)->float:
        metrics = self.metricsHistory.singleMetrics.ensureMetrics(
            self.getRequiredSingleMetrics())
        loss: float = metrics.getValue(
            epochID=self.nbEpochesFinished, metricName="loss_train", assertValueType=float)
        newLr: float = self.lossToLearningRate_table.getLr(loss)
        self.learningRate = newLr
        return newLr

    def _BaseTrain(self,
            datasGenerator:"Datas_training_Generator", autoUpdateLr:bool, 
            verbose:"_AI_Verbose")->"keras.callbacks.History":
        assert self.__isCompiled is True
        if datasGenerator.nbTrainingSteps == 0:
            raise ValueError(f"there is no training datas")
        if datasGenerator.nbValidationSteps == 0:
            raise ValueError(f"there is no validation datas")
        if (self.kerasConfig.tf_v1_compat is True) and (datasGenerator.nbEpoches != 1):
            raise ValueError(f"when the TF_V1_compat is acctivated you cant' do more than one epoche")
        
        ###train on the datas
        timeMesures = MesureFittingTimes()
        history: "keras.callbacks.History|None" = \
            self.model.fit(
                datasGenerator.getTrainingGenerator(),
                steps_per_epoch=datasGenerator.nbTrainingSteps,
                validation_data=datasGenerator.getValidationGenerator(),
                validation_steps=datasGenerator.nbValidationSteps,
                initial_epoch=self.nbEpochesFinished,
                epochs=self.nbEpochesFinished+datasGenerator.nbEpoches,
                callbacks=timeMesures,
                use_multiprocessing=False, workers=1,
                verbose=cast(str, convertVerbose(verbose))) # beacuse bad typing from keras
        assert isinstance(history, keras.callbacks.History)
        # TODO print the all posssible thing of the history to have more infos
        self.nbEpochesFinished += datasGenerator.nbEpoches
        
        ### register the fit call
        self.fittingHistory.addCall(history)
        
        ### add the metrics
        singleMetrics = self.metricsHistory.singleMetrics
        lossValues: "list[float]" = assertListSubType(float, history.history["loss"])
        lossValValues: "list[float]" = assertListSubType(float, history.history["val_loss"])
        for index in range(len(history.epoch)):
            epochID: int = history.epoch[index] + 1
            singleMetrics.addSingle(epochID, "loss_train", lossValues[index])
            singleMetrics.addSingle(epochID, "loss_val", lossValValues[index])
        # dict[callbackEpoch -> time]
        trainStepTimes: "dict[int, float]" = timeMesures.values["train_batch"]
        testStepTimes: "dict[int, float]" = timeMesures.values["test_batch"] 
        for callbackEpoch in timeMesures.epochs:
            epochID: int = callbackEpoch + 1
            # time metrics
            singleMetrics.addSingle(epochID, "trainTime", trainStepTimes[callbackEpoch])
            singleMetrics.addSingle(epochID, "valTime", testStepTimes[callbackEpoch])
            # nb steps metrics
            singleMetrics.addSingle(epochID, "nbTrainSteps", datasGenerator.nbTrainingSteps)
            singleMetrics.addSingle(epochID, "nbValSteps", datasGenerator.nbValidationSteps)
        
        if autoUpdateLr is True:
            self.autoSetLearningRate()
        return history
    
    def evaluate(self, datasGenerator:"Datas_training_Generator", verbose:"_AI_Verbose")->float:
        """evaluate the model on the given datas (use the training generator, val is ignored)"""
        return assertIsinstance(float, self.model.evaluate(
            x=datasGenerator.getTrainingGenerator(),
            steps=datasGenerator.nbTrainingSteps,
            batch_size=self.batch_size,
            use_multiprocessing=False, workers=1,
            verbose=cast(str, convertVerbose(verbose)))) # beacuse bad typing from keras

    def printTraingResume(self)->None:
        singleMetrics = self.metricsHistory.singleMetrics
        for epoch in range(1, self.nbEpochesFinished+1):
            nbTrainSteps = singleMetrics.getValue(epoch, "nbTrainSteps", assertValueType=int)
            nbValSteps = singleMetrics.getValue(epoch, "nbValSteps", assertValueType=int)
            trainTime = singleMetrics.getValue(epoch, "trainTime", assertValueType=float)
            valTime = singleMetrics.getValue(epoch, "valTime", assertValueType=float)
            loss_train = singleMetrics.getValue(epoch, "loss_train", assertValueType=float)
            loss_val = singleMetrics.getValue(epoch, "loss_val", assertValueType=float)
            print(f"Epoch {epoch} - steps: {nbTrainSteps}|{nbValSteps} - "
                  f"{trainTime*1000:.0f}ms|{valTime*1000:.0f}ms - "
                  f"loss: {loss_train:.4f} - val_loss: {loss_val:.4f}")











_convTrainSingleMetrics = Literal["nbConvStep_train", "nbConvStep_val"]
_backtestsSingleMetrics = Literal["avgWinRate_train", "avgWinRate_val", 
                                  "avgTradesRate_train", "avgTradesRate_val",
                                  "avgTradesDuration_train", "avgTradesDuration_val"]
_convTrainDfsMetrics = Literal["nbConvStep_train", "nbConvStep_val"]
_backtestsDfsMetrics = Literal["winRate_train", "winRate_val", "tradesRate_train", "tradesRate_val",
                               "tradesDuration_train", "tradesDuration_val"]


### handle the metrics concatenation required for AI_SupportsConvergenceTrain
_SupportsConvergenceTrain_reqSingleMetrics = Union[
    _baseTrainSingleMetrics, _convTrainSingleMetrics, _backtestsSingleMetrics]
_SupportsConvergenceTrain_reqDfsMetrics = Union[
    _baseTrainDfsMetrics, _convTrainDfsMetrics, _backtestsDfsMetrics]

class AI_SupportsConvergenceTrain(
        AI_SupportsBaseTrain[Union[_MetricNameSingle, _convTrainSingleMetrics, _backtestsSingleMetrics],
                             Union[_MetricNameDfs, _convTrainDfsMetrics, _backtestsDfsMetrics]]):

    @classmethod
    def getRequiredSingleMetrics(cls)->"set[_SupportsConvergenceTrain_reqSingleMetrics]":
        newMetrics1: "set[_convTrainSingleMetrics]" = set(get_args(_convTrainSingleMetrics))
        newMetrics2: "set[_backtestsSingleMetrics]" = set(get_args(_backtestsSingleMetrics))
        return (newMetrics1 | newMetrics2 | super().getRequiredSingleMetrics())
    @classmethod
    def getRequiredDfsMetrics(cls)->"set[_SupportsConvergenceTrain_reqDfsMetrics]":
        newMetrics1: "set[_convTrainDfsMetrics]" = set(get_args(_convTrainDfsMetrics))
        newMetrics2: "set[_backtestsDfsMetrics]" = set(get_args(_backtestsDfsMetrics))
        return (newMetrics1 | newMetrics2 | super().getRequiredDfsMetrics())

    def convergenceTraining(
            self, datas:"set[Datas_dataset[_T_DfKey]]", 
            nbEpoches:int, convergenceConfig:"ConvergeConfig", 
            runBacktests:bool=True, autoUpdateLR:bool=False, 
            verbose:"Literal['disable', 'normal', 'high']"="normal")->None:
        self.trainingCallsHistory.append(
            TrainCallHist.create(
                self.convergenceTraining, 
                funcsKwargs={"datas": [df.getInfos().toJson() for df in datas],
                             "nbEpoches": nbEpoches, "runBacktests": runBacktests,
                             "convergenceConfig": convergenceConfig.toJson(), 
                             "autoUpdateLR": autoUpdateLR, "verbose": verbose}))
        ... # TODO
