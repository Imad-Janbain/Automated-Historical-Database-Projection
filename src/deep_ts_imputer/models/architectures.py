"""Model architectures for time-series imputation.

All models share the same input/output contract::

    inputs:  (batch, look_back, n_features_in)
    outputs: (batch, n_outputs)   # = horizon * n_features_out

Available architectures:

* ``lstm``        — stacked LSTM
* ``bilstm``      — stacked bidirectional LSTM (default in the Seine paper)
* ``gru``         — stacked GRU
* ``bigru``       — stacked bidirectional GRU
* ``cnn_bilstm``  — 1-D Conv front-end feeding into BiLSTM, optional
                    self-attention block before the dense head

Use :func:`build_model` (in ``factory.py``) rather than calling these
constructors directly so that configuration is the single source of truth.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Layer,
    Permute,
    multiply,
)


def _stacked_recurrent(
    layer_cls,
    units: int,
    num_layers: int,
    dropout: float,
    look_back: int,
    n_features_in: int,
    bidirectional: bool,
):
    """Build a stacked (Bi)RNN trunk as a Sequential model (no head)."""
    model = Sequential()
    input_shape = (look_back, n_features_in)

    def wrap(layer):
        return Bidirectional(layer) if bidirectional else layer

    if num_layers == 1:
        model.add(wrap(layer_cls(units=units, input_shape=input_shape)))
        model.add(Dropout(dropout))
        return model

    model.add(
        wrap(
            layer_cls(
                units=units,
                return_sequences=True,
                input_shape=input_shape,
            ),
        ),
    )
    model.add(Dropout(dropout))
    for _ in range(num_layers - 2):
        model.add(wrap(layer_cls(units=units, return_sequences=True)))
        model.add(Dropout(dropout))
    model.add(wrap(layer_cls(units=units)))
    model.add(Dropout(dropout))
    return model


def build_lstm(look_back, n_features_in, n_outputs, units, num_layers, dropout):
    model = _stacked_recurrent(
        LSTM, units, num_layers, dropout, look_back, n_features_in, bidirectional=False,
    )
    model.add(Dense(n_outputs))
    return model


def build_bilstm(look_back, n_features_in, n_outputs, units, num_layers, dropout):
    model = _stacked_recurrent(
        LSTM, units, num_layers, dropout, look_back, n_features_in, bidirectional=True,
    )
    model.add(Dense(n_outputs))
    return model


def build_gru(look_back, n_features_in, n_outputs, units, num_layers, dropout):
    model = _stacked_recurrent(
        GRU, units, num_layers, dropout, look_back, n_features_in, bidirectional=False,
    )
    model.add(Dense(n_outputs))
    return model


def build_bigru(look_back, n_features_in, n_outputs, units, num_layers, dropout):
    model = _stacked_recurrent(
        GRU, units, num_layers, dropout, look_back, n_features_in, bidirectional=True,
    )
    model.add(Dense(n_outputs))
    return model


def _attention_block(inputs):
    """Simple per-timestep soft-attention head, as in the Paper-2 code."""
    input_dim = int(inputs.shape[2])
    a = Dense(input_dim, activation="softmax")(inputs)
    a_probs = Permute((1, 2), name="attention_vec")(a)
    return multiply([inputs, a_probs], name="attention_mul")


def build_cnn_bilstm(
    look_back,
    n_features_in,
    n_outputs,
    units,
    num_layers,
    dropout,
    cnn_filters: int = 32,
    use_attention: bool = False,
):
    """1-D CNN front-end + stacked BiLSTM, optionally followed by attention."""
    inputs = Input(shape=(look_back, n_features_in))
    x = Conv1D(filters=cnn_filters, kernel_size=1, activation="relu")(inputs)
    x = Dropout(dropout)(x)

    return_seq = use_attention or num_layers > 1
    x = Bidirectional(LSTM(units, return_sequences=return_seq))(x)
    x = Dropout(dropout)(x)
    for i in range(num_layers - 1):
        last = i == num_layers - 2 and not use_attention
        x = Bidirectional(LSTM(units, return_sequences=not last))(x)
        x = Dropout(dropout)(x)

    if use_attention:
        x = _attention_block(x)
        x = Flatten()(x)

    outputs = Dense(n_outputs)(x)
    return Model(inputs=inputs, outputs=outputs)
