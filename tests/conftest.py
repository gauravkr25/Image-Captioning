import numpy as np
import pytest


@pytest.fixture
def synthetic_features_and_captions():
    features = np.random.randn(6, 49, 8).astype("float32")
    captions = [
        "a dog runs in grass",
        "a dog sits near a tree",
        "two cats watch the sky",
        "a person holds a ball",
        "small red train on tracks",
        "people walk by a street",
    ]
    return features, captions
