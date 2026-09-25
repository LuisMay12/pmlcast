#!/usr/bin/env python3
"""Define, train and apply the LSTM forecasters."""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from pmlcast import config
from pmlcast import preprocess

DEFAULT_UNITS = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.001
HUBER_DELTA = 1.0
SHUFFLE_BUFFER = 10000
HEAD_UNITS = 64


def set_seeds(seed=config.SEED):
    """Seed numpy, TensorFlow and Keras for reproducible runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    K.utils.set_random_seed(seed)


def build_model(
    seq_shape,
    n_cal,
    n_level,
    n_meta=0,
    n_zones=0,
    zone_dim=4,
    n_nodes=0,
    node_dim=4,
    units=DEFAULT_UNITS,
    dropout=DEFAULT_DROPOUT,
    loss="huber",
    learning_rate=DEFAULT_LEARNING_RATE,
):
    """Build and compile the sequence-to-vector LSTM.

    seq -> LSTM(units, return_sequences) -> Dropout -> LSTM(units // 2)
    -> Dropout; concat[encoder, cal, level (, meta) (, zone embedding)
    (, node embedding)] -> Dense(64, relu) -> Dense(24) in z-units.

    Args:
        seq_shape: Shape of one input sequence, e.g. ``(168, 9)``.
        n_cal: Number of target-day calendar features.
        n_level: Number of level features.
        n_meta: Number of static metadata features (0 = history-only).
        n_zones: Load-zone vocabulary size (0 = no zone embedding).
        zone_dim: Embedding size for the load zone.
        n_nodes: Node vocabulary size (0 = no node embedding; ablation).
        node_dim: Embedding size for the node identity.
        units: Units of the first LSTM layer.
        dropout: Dropout rate after each LSTM layer.
        loss: ``"huber"`` or ``"mae"``.
        learning_rate: Adam learning rate.

    Returns:
        A compiled ``keras.Model`` taking a dict of named inputs.
    """
    if loss not in ("huber", "mae"):
        raise ValueError("loss must be 'huber' or 'mae'")
    inputs = {
        "seq": K.Input(shape=tuple(seq_shape), name="seq"),
        "cal": K.Input(shape=(n_cal,), name="cal"),
        "level": K.Input(shape=(n_level,), name="level"),
    }
    encoded = K.layers.LSTM(units, return_sequences=True)(inputs["seq"])
    encoded = K.layers.Dropout(dropout)(encoded)
    encoded = K.layers.LSTM(units // 2)(encoded)
    encoded = K.layers.Dropout(dropout)(encoded)
    parts = [encoded, inputs["cal"], inputs["level"]]

    if n_meta:
        inputs["meta"] = K.Input(shape=(n_meta,), name="meta")
        parts.append(inputs["meta"])
    if n_zones:
        inputs["zone"] = K.Input(shape=(1,), dtype="int32", name="zone")
        zone = K.layers.Embedding(n_zones, zone_dim)(inputs["zone"])
        parts.append(K.layers.Flatten()(zone))
    if n_nodes:
        inputs["node"] = K.Input(shape=(1,), dtype="int32", name="node")
        node = K.layers.Embedding(n_nodes, node_dim)(inputs["node"])
        parts.append(K.layers.Flatten()(node))

    hidden = K.layers.Concatenate()(parts)
    hidden = K.layers.Dense(HEAD_UNITS, activation="relu")(hidden)
    output = K.layers.Dense(config.TARGET_HOURS, name="y")(hidden)

    model = K.Model(inputs=inputs, outputs=output)
    loss_fn = K.losses.Huber(delta=HUBER_DELTA) if loss == "huber" else "mae"
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["mae"],
    )
    return model


def prepare_inputs(inputs):
    """Cast input arrays for Keras: floats to float32, ids to (n, 1) int32."""
    prepared = {}
    for name, values in inputs.items():
        values = np.asarray(values)
        if name in ("zone", "node"):
            values = values.astype(np.int32).reshape(len(values), 1)
        else:
            values = values.astype(np.float32)
        prepared[name] = values
    return prepared


def make_tf_dataset(
    inputs, y, batch_size, shuffle=False, seed=config.SEED, zone_dropout=0.0
):
    """Create a ``tf.data`` pipeline from input arrays and targets.

    Args:
        inputs: Dict of input arrays (see ``dataset.inputs_dict``).
        y: Targets in z-units, shape (n, 24).
        batch_size: Batch size.
        shuffle: Shuffle every epoch (training only).
        seed: Shuffle seed.
        zone_dropout: Probability of replacing a sample's zone by UNKNOWN
            (index 0) so the fallback embedding is learned.

    Returns:
        A batched, prefetched ``tf.data.Dataset``.
    """
    features = prepare_inputs(inputs)
    targets = np.asarray(y, dtype=np.float32)
    data = tf.data.Dataset.from_tensor_slices((features, targets))
    if shuffle:
        data = data.shuffle(
            min(len(targets), SHUFFLE_BUFFER),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    if zone_dropout > 0 and "zone" in features:

        def drop_zone(feats, target):
            keep = tf.random.uniform([]) >= zone_dropout
            feats = dict(feats)
            feats["zone"] = tf.where(
                keep, feats["zone"], tf.zeros_like(feats["zone"])
            )
            return feats, target

        data = data.map(drop_zone)
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_callbacks(patience, checkpoint_path=None):
    """Return early stopping, LR reduction and optional checkpointing."""
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-6,
        ),
    ]
    if checkpoint_path:
        callbacks.append(
            K.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True
            )
        )
    return callbacks


def train_model(
    model,
    train_inputs,
    y_train,
    val_inputs,
    y_val,
    epochs=50,
    patience=5,
    batch_size=256,
    checkpoint_path=None,
    seed=config.SEED,
    zone_dropout=0.0,
    verbose=1,
):
    """Fit the model with early stopping and return the Keras history."""
    set_seeds(seed)
    train_data = make_tf_dataset(
        train_inputs,
        y_train,
        batch_size,
        shuffle=True,
        seed=seed,
        zone_dropout=zone_dropout,
    )
    val_data = make_tf_dataset(val_inputs, y_val, batch_size)
    return model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        shuffle=False,  # the tf.data pipeline already shuffles
        callbacks=get_callbacks(patience, checkpoint_path),
        verbose=verbose,
    )


def predict_z(model, inputs, batch_size=1024):
    """Return predictions in z-units, shape (n, 24)."""
    return model.predict(
        prepare_inputs(inputs), batch_size=batch_size, verbose=0
    )


def predict_prices(model, inputs, mu, sigma, batch_size=1024):
    """Return predictions in MXN/MWh, shape (n, 24)."""
    return preprocess.invert(predict_z(model, inputs, batch_size), mu, sigma)
