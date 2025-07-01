import keras.activations
import tensorflow as tf
import keras
import keras.engine.keras_tensor

from holo.__typing import Literal, Union

from AI import ValuesRange

KerasTensor = Union[keras.engine.keras_tensor.KerasTensor, tf.Tensor]

_ActivationName = Literal["linear", "sigmoid", "tanh", "relu",
                          "hard_sigmoid", "gelu", "swish", "elu", 
                          "selu", "softplus", "softmax", "leaky_relu"]
_ModelDtype = Literal["float16", "float32", "float64"]


activation_to_valuesRange: "dict[_ActivationName, ValuesRange]" = {
    "linear": ValuesRange(mini=float("-inf"), maxi=float("+inf")),
    "sigmoid": ValuesRange(mini=0.0, maxi=1.0),
    "hard_sigmoid": ValuesRange(mini=0.0, maxi=1.0),
    "tanh": ValuesRange(mini=-1.0, maxi=1.0),
    "relu": ValuesRange(mini=0.0, maxi=float("+inf")),
    "gelu": ValuesRange(mini=-0.1700, maxi=float("+inf")), # range = [-0.169971..., +inf]
    "swish": ValuesRange(mini=-0.2785, maxi=float("+inf")), # range = [-0.27846..., +inf]
    "elu": ValuesRange(mini=-1.0, maxi=float("+inf")), # range = [-a, +inf] | base a in keras is 1.0
    "selu": ValuesRange(mini=-1.7581, maxi=float("+inf")), # range = [-1.75809..., +inf]
    "softplus": ValuesRange(mini=0.0, maxi=float("+inf")),
    "softmax": ValuesRange(mini=0.0, maxi=1.0),
    "leaky_relu": ValuesRange(mini=float("-inf"), maxi=float("+inf")),
}
