"""Export the trained CNN (and ten sample test digits) for the browser demo in web_demo/.

    python -m src.export_web      # writes web_demo/model_weights.js, then open web_demo/index.html
"""
import base64
import json

import numpy as np
from tensorflow import keras

from . import config, data


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def main():
    model = keras.models.load_model(config.CNN_PATH)
    w = model.get_weights()   # conv1 kernel/bias, conv2 kernel/bias, dense kernel/bias (Keras order)
    names = ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "dense_w", "dense_b"]
    weights = {n: _b64(a, "<f4") for n, a in zip(names, w)}
    _, (x_test, y_test) = data.load_mnist()
    samples = [{"label": d, "px": _b64(x_test[int(np.where(y_test == d)[0][0])], np.uint8)} for d in range(10)]
    path = config.ROOT_DIR / "web_demo" / "model_weights.js"
    path.write_text("window.MODEL_WEIGHTS = " + json.dumps(weights) + ";\nwindow.MNIST_SAMPLES = " + json.dumps(samples) + ";\n")
    print(f"Wrote {path} ({path.stat().st_size / 1e3:.0f} kB). Open web_demo/index.html in a browser.")


if __name__ == "__main__":
    main()
