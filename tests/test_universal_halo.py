from neurooptai.core.universal_halo import UniversalHALO


def test_universal_halo_should_intervene():
    halo = UniversalHALO()
    assert halo.should_intervene(3, 10) is True


def test_universal_halo_score():
    halo = UniversalHALO()
    assert halo.score(3, 10) == 10 / 3


def test_universal_halo_classify():
    halo = UniversalHALO()
    assert halo.classify(3, 10) == "intervene"


def test_universal_halo_expected_avoided_cost():
    halo = UniversalHALO()
    assert halo.expected_avoided_cost(0.8, 100) == 80


def test_universal_halo_probabilistic_intervention():
    halo = UniversalHALO()
    assert halo.should_intervene_probabilistic(20, 0.8, 100) is True


def test_universal_halo_rejects_zero_intervention_cost():
    halo = UniversalHALO()

    try:
        halo.score(0, 10)
    except ValueError as error:
        assert "greater than 0" in str(error)
    else:
        assert False


def test_universal_halo_rejects_invalid_probability():
    halo = UniversalHALO()

    try:
        halo.expected_avoided_cost(1.5, 100)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        assert False
