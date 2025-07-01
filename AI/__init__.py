import numpy
import attrs
import math

from holo.__typing import Iterable, Self
from holo.types_ext import _Serie_Float, _Serie_Integer, _Serie_Boolean

from modules.numbaJit import fastJitter, floatArray, floating, boolean, void
from calculationLib import _SerieName, validSerieNames
from calculationLib.rerange import (reRange, reRange2, reRange3, _moveRange)

from save_formats import AsJson_ValuesRange, AsJson_RegularizeConfig

### idées pour la V1 du logiciel
# rajouter un principe de positions a tailles variables 
# (peut etre une mauvaise idée par raport au calc de positions optimales)
#   avec un seuil d'achat un seuil de rachat
#       tel que: si over TBuy puis under TReBuy puis over TBuy
#           => buy more
#   donc determiner une facon de repatir son argent
#       idée: amoutRebuy(ttArgent, nbRebuy) = 
#           ttArgent * ((1/alpha)**(nbRebuy)  * (alpha-1))
# changer l'indexage de senti2: on ne predit plus T+0 mais T+1
#   en assumant que on connant T+0 => les index de toutes les series sont allignés
#   C'est réaliste car en simulation, on start avec 0.5 et on converge vers la vrais serie
# pour le calc de la simu: 
#   on calc toute la serie en 1 coup, on prend les resultats et on les décales en T+1,
#   puis on calc l'erreur avec le calc precedant
#   et on répete le procédé tant que l'erreur est au dessu d'un seuil fixé
# ajouter des transformersBlocks dans Senti2_V5

### idées pour la V2 du logiciel
# l'ia vas etre train avec du Reinforcement Learning
# l'ia auras 3 outputs: 
#       - senti2_c
#       - le niveaux de confiance
#       - la proportion du capital a engager (ou a posserder, approfondire)
#   ce niveau de confiance vas permetre de determiner
#   si l'ia doit trader ou s'abstenir (gestion des riques)
# le RL vas se baser sur les critéres surivants
#   ++++ si le trade current dans le positif
#   + si le trade est plus profitable que avant
#   - si le trade est moins profitable que avant
#   ++ si le trade n'as pas été pris, et qu'il aurait été negatif
#       (ie: l'ia s'est abstenu et a eut raison, cf niveaux de confiance)
#   -- si le trade n'as pas été pris, mais qu'il aurait été positif
#       (ie: l'ia s'est abstenu et a eut tord)


# the series that are suported throug the regularization process
REGULARIZABLE_SERIES: "set[_SerieName]" = {
    ... # REMOVED
}
"""all the series that can be regularized"""

REQUIRE_REGULARIZE_SERIES: "set[_SerieName]" = {
    ... # REMOVED
    
}
"""the series that need need to be rescaled to be suported != series that can be rescaled"""

UNREGULARIZABLE_SERIES: "set[_SerieName]" = \
    validSerieNames.difference(REGULARIZABLE_SERIES)
"""the series that can't be regularized""" 


def moveRange(array:"_Serie_Float", valuesRange:"ValuesRange")->_Serie_Float:
    """move, inplace, the range of the values from [0, 1] to [mini, maxi]"""
    if math.isinf(valuesRange.mini) or math.isinf(valuesRange.maxi):
        raise ValueError(f"impossible to move the range to {valuesRange}, it has some +-inf")
    _moveRange(array, valuesRange.mini, valuesRange.maxi)
    return array

########################## regularization of the datas

@attrs.frozen
class ValuesRange():
    mini: float
    maxi: float
    
    def toJson(self)->"AsJson_ValuesRange":
        return AsJson_ValuesRange(
            cls=self.__class__.__name__,
            mini=self.mini, maxi=self.maxi)
    @classmethod
    def fromJson(cls, datas:"AsJson_ValuesRange")->"Self":
        assert datas["cls"] == cls.__name__
        self = ValuesRange.__new__(cls)
        ValuesRange.__init__(self=self, mini=datas["mini"], maxi=datas["maxi"])
        return self

_DefaultValuesRange = ValuesRange(mini=0.0, maxi=1.0)



@attrs.frozen
class RegularizeConfig():
    """standardise les kwargs authorisés par regularize_datas(...) pour les reglages"""
    rescale: bool
    preferReRangePlus: bool
    valuesRange: "ValuesRange"
    
    def toJson(self)->"AsJson_RegularizeConfig":
        return AsJson_RegularizeConfig(
            cls=self.__class__.__name__,
            rescale=self.rescale,
            preferReRangePlus=self.preferReRangePlus,
            valuesRange=self.valuesRange.toJson())
    @classmethod
    def fromJson(cls, datas:"AsJson_RegularizeConfig")->"Self":
        assert datas["cls"] == cls.__name__
        self = RegularizeConfig.__new__(cls)
        RegularizeConfig.__init__(
            self=self, rescale=datas["rescale"], 
            preferReRangePlus=datas["preferReRangePlus"],
            valuesRange=ValuesRange.fromJson(datas["valuesRange"]))
        return self
        



def regularize_datas(
        seriesToRegularize:"dict[_SerieName, _Serie_Float]", 
        seriesSelection:"Iterable[_SerieName]|None",
        rescale:bool=True, preferReRangePlus:bool=True,
        valuesRange:"ValuesRange"=_DefaultValuesRange)->"dict[_SerieName, _Serie_Float]":
    """return a dict containg all the regularized series, the regularized range is [0, 1]\n
    `seriesToRegularize` are the series that will be regularized\n
    `seriesSelection` is a filter of the of the series to be regularized \
        if set to None it will do all the series it can\n"""
    regularized_datas:"dict[_SerieName, _Serie_Float]" = {}

    if seriesSelection is None:
        seriesSelection = set(seriesToRegularize.keys())
    elif not isinstance(seriesSelection, set):
        seriesSelection = set(seriesSelection)
    # => seriesSelection is a set[_SerieName_ext]
        
    if not seriesSelection.issubset(REGULARIZABLE_SERIES):
        raise ValueError(f"the following series aren't regularizable: {seriesSelection.difference(REGULARIZABLE_SERIES)}")

    # assert that all the requested series are in `seriesToRegularize`
    if not seriesSelection.issubset(seriesToRegularize.keys()):
        raise ValueError(
            "not all the series in `seriesSelection` are available in `seriesToRegularize`, "
            f"the following series are missing: {seriesSelection.difference(seriesToRegularize.keys())}")

    RESCALE_COEFS:"dict[str, float]" = {
        # all the series in SUPPORTED_SERIES_WITH_RESCALE need to ne inside
        ... # REMOVED
        }

    RERANGE2_ALPHA:"dict[str, float]" = {
        # need to contain all the series in RESCALE_COEFS
        ... # REMOVED
        }

    RERANGE2_KARGS:"dict[str, dict[str, bool]]" = {
        ... # REMOVED
        }
    
    RERANGE3_COEFF:"dict[str, float]" = {
        ... # REMOVED
        }
    RERANGE3_KARGS:"dict[str, dict[str, bool]]" = {
        ... # REMOVED
        }

    if len(set(RERANGE2_ALPHA.keys()).intersection(RERANGE3_COEFF)) > 0:
        raise ValueError("intersection betwin rerange2's and rerange3's dict")
        # so if one elt is in one it must use this methode


    for serie_name in seriesSelection:
        oldSerie:_Serie_Float = seriesToRegularize[serie_name]
        newSerie:_Serie_Float
        applieRescale:bool = rescale and (serie_name in RESCALE_COEFS)
        
        # select the rerange methode
        applieReRange:int = 1
        if preferReRangePlus is True:
            if serie_name in RERANGE2_ALPHA:
                applieReRange = 2
            elif serie_name in RERANGE3_COEFF:
                applieReRange = 3


         #rescale the serie to a more appropriate value (with a (true) range that is more comprehensible for the ai)
        if applieRescale is True:
            newSerie = oldSerie * RESCALE_COEFS[serie_name]
        else: newSerie = oldSerie.copy()

        # the series that are alredy the the correct range: [0, 1]
        if serie_name in (
                ... # REMOVED
                ):
            pass

        # the series that are in range: [-1, 1]
        elif serie_name in (
                ... # REMOVED
                ):
            newSerie = (newSerie + 1) / 2

        # the series that are in range: [0, 100]
        elif serie_name in (
                ... # REMOVED
                ):
            newSerie = newSerie / 100


        # strictly limit the range to [0, 1]
        if applieReRange == 1:
            regularized_datas[serie_name] = reRange(newSerie)
        elif applieReRange == 2:
            regularized_datas[serie_name] = reRange2(newSerie, RERANGE2_ALPHA[serie_name], **RERANGE2_KARGS.get(serie_name, dict()))
        elif applieReRange == 3:
            regularized_datas[serie_name] = reRange3(newSerie, RERANGE3_COEFF[serie_name], **RERANGE3_KARGS.get(serie_name, dict()))
        else: raise ValueError(f"bad dev: unkown `applieReRange`: {applieReRange}")
        
        if valuesRange != _DefaultValuesRange:
            moveRange(regularized_datas[serie_name], valuesRange)

    return regularized_datas





