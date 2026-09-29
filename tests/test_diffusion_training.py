from sparsa.train_diffusion import validation_due


def test_validation_cadence_retries_only_incomplete_milestones():
    assert not validation_due(0, 30000, 10000)
    assert not validation_due(9999, 30000, 10000)
    assert validation_due(10000, 30000, 10000)
    assert not validation_due(10000, 30000, 10000, complete=True)
    assert validation_due(30000, 30000, 0)
    assert not validation_due(30000, 30000, 0, complete=True)
