import keras
import keras.layers as layers

class _TransformerBlock_GPT(layers.Layer):
    def __init__(self,
            nb_heads:int, embeded_dims:int, feedForward_size:int,
            nbDims_heads:int, dropoutRate:"float|None"=0.1)->None:
        super().__init__()
        # LayerNorm
        self.layerNorm_1 = layers.LayerNormalization(epsilon=1e-6)
        # Linear, Attention heads
        self.heads_attention = layers.MultiHeadAttention(
            nb_heads, key_dim=embeded_dims, value_dim=nbDims_heads, 
            dropout=(dropoutRate if isinstance(dropoutRate, float) else 0.0),
        )
        # Linear
        self.linear_1 = layers.Dense(embeded_dims)
        # Dropout
        self.dropout_1 = layers.Dropout(dropoutRate)
        # LayerNorm
        self.layerNorm_2 = layers.LayerNormalization(epsilon=1e-6)
        # FeedForward Linear
        self.linear_2 = keras.Sequential([
            layers.Dense(feedForward_size, activation="relu"),
            layers.Dense(embeded_dims),
        ])
        # Dropout
        self.dropout_2 = layers.Dropout(dropoutRate)

    def call(self, inputs, training):
        normalizedInputs = self.layerNorm_1(inputs)

        attention_outputs = self.heads_attention(normalizedInputs, inputs)
        attention_outputs = self.linear_1(attention_outputs)
        attention_outputs = self.dropout_1(attention_outputs, training=training)

        post_attention_outputs = self.layerNorm_2(attention_outputs + inputs)

        feedForward_outputs = self.linear_2(post_attention_outputs)
        feedForward_outputs = self.dropout_2(feedForward_outputs, training=training)

        return feedForward_outputs


class _Transformers_GPT(layers.Layer):
  def __init__(self,
            nb_trasformers:int, nb_heads:int, embeded_dims:int, nbDims_heads:int,
            feedForward_size:int, dropoutRate:"float|None"=0.1, *args, **kwargs)->None:
    super().__init__(*args, **kwargs)
    self.nb_trasformers = nb_trasformers
    self.transformerBlock_layers = [
        _TransformerBlock_GPT(
            nb_heads=nb_heads,
            embeded_dims=embeded_dims,
            nbDims_heads=nbDims_heads,
            feedForward_size=feedForward_size,
            dropoutRate=dropoutRate,
        )
        for _ in range(self.nb_trasformers)
    ]

  def call(self, inputs, training):
    for i in range(self.nb_trasformers):
      inputs = self.transformerBlock_layers[i](inputs, training)
    return inputs