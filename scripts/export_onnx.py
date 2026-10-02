#!/usr/bin/env python3
"""Export the trained Keras model to ONNX for the serving container."""

import argparse
import os

import numpy as np

from pmlcast import serving

INPUTS = ("seq", "cal", "level")
TOLERANCE = 1e-3  # z-units; a float32 round trip stays well inside this
CHECK_SAMPLES = 64


def _random_feeds(model, n, seed=0):
    """Return a dict of random float32 inputs shaped like the model's."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for tensor in model.inputs:
        name = tensor.name.split(":")[0]
        shape = [n] + [int(d) for d in tensor.shape[1:]]
        feeds[name] = rng.standard_normal(shape).astype(np.float32)

    return feeds


def export(keras_path, onnx_path):
    """Write the ONNX file and return the loaded Keras model."""
    import keras

    model = keras.models.load_model(keras_path)
    # A loaded model has not traced its call yet, and export needs that.
    model(_random_feeds(model, 1))
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    # Keras 3 converts through tf2onnx and keeps the named inputs, which is
    # what serving.Forecaster feeds: seq, cal and level.
    model.export(onnx_path, format="onnx")

    return model


def check(model, onnx_path, seed=0):
    """Compare Keras and ONNX on random inputs; return the largest gap."""
    import onnxruntime as ort

    feeds = _random_feeds(model, CHECK_SAMPLES, seed)
    expected = np.asarray(
        model.predict({k: feeds[k] for k in INPUTS}, verbose=0)
    )
    session = ort.InferenceSession(onnx_path)
    names = [i.name for i in session.get_inputs()]
    if sorted(names) != sorted(INPUTS):
        raise SystemExit("unexpected ONNX inputs: {}".format(names))
    got = session.run(None, {k: feeds[k] for k in INPUTS})[0]

    return float(np.max(np.abs(expected - got)))


def main():
    """Export, then refuse to leave a model behind that disagrees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", default=serving.KERAS_MODEL)
    parser.add_argument("--out", default=serving.DEFAULT_MODEL)
    args = parser.parse_args()

    model = export(args.keras, args.out)
    gap = check(model, args.out)
    print("exported {} (max |Keras - ONNX| = {:.2e})".format(args.out, gap))
    if gap > TOLERANCE:
        raise SystemExit(
            "ONNX disagrees with Keras beyond {}".format(TOLERANCE)
        )


if __name__ == "__main__":
    main()
