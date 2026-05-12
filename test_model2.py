
import pytest
import joblib
import numpy as np


@pytest.fixture
def model():
    return joblib.load("gi_model.pkl")


# =========================================================
# TEST 1
# Check model loaded
# =========================================================

def test_model_loaded(model):
    assert model is not None


# =========================================================
# TEST 2
# Check prediction shape
# =========================================================

def test_prediction_shape(model):

    sample_review = ["this game is so bad,i dont even wanna playit anymore,i might uninstall it"]

    prediction = model.predict(sample_review)

    assert len(prediction) == 1


# =========================================================
# TEST 3
# Zero-table test
# Known prediction
# =========================================================

def test_known_prediction(model):

    sample_data = ["this game is so bad,i dont even wanna playit anymore,i might uninstall it"]

    prediction = model.predict(sample_data)

    # Expected Setosa = class 0
    assert prediction[0] == 1


# =========================================================
# TEST 4
# Probabilities sum to 1
# =========================================================

def test_probability_sum(model):

    sample_data = ["this game is so bad,i dont even wanna playit anymore,i might uninstall it"]

    probabilities = model.predict_proba(sample_data)

    total_prob = probabilities[0].sum()

    assert total_prob == pytest.approx(1.0, abs=1e-6)


# =========================================================
# TEST 5
# Parameterized Testing
# =========================================================

@pytest.mark.parametrize(
    "sample_review, expected_review",
    [
        (["this game is so bad and cheapsakte,not recommended"], 1),
        (["the game is good, but need more update"], 1), 
        (["this game is so good and beautiful, i like the game system and the gacha is so blessful"], 5),
    ]
)
def test_multiple_predictions(model, sample_review, expected_review):
    prediction = model.predict(sample_review)
    assert prediction[0] == expected_review
