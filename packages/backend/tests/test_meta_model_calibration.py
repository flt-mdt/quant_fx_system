import numpy as np

from quant_fx_system.quant.meta_model.calibration import make_calibrator
from quant_fx_system.quant.meta_model.diagnostics import expected_calibration_error


def test_calibration_improves_ece_on_train():
    rng = np.random.default_rng(7)
    scores = rng.uniform(0.05, 0.95, size=200)
    y_true = (scores > 0.6).astype(float)
    y_true = np.clip(y_true + rng.normal(scale=0.1, size=200), 0, 1)

    ece_before = expected_calibration_error(y_true, scores)

    calibrator = make_calibrator("isotonic")
    calibrator.fit(scores, y_true)
    calibrated = calibrator.predict(scores)

    ece_after = expected_calibration_error(y_true, calibrated)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert ece_after <= ece_before
