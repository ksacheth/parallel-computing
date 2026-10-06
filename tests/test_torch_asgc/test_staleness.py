from torch_asgc.staleness import StalenessConfig, decide_staleness


def test_staleness_policy_bounds():
    config = StalenessConfig(s_low=2, s_max=8, beta=0.5)

    # Within S_low: full weight
    dec0 = decide_staleness(0, config)
    assert dec0.accept and dec0.weight == 1.0 and dec0.kind == "accepted"
    dec2 = decide_staleness(2, config)
    assert dec2.accept and dec2.weight == 1.0

    # Between S_low and S_max: downweighted
    dec4 = decide_staleness(4, config)
    assert dec4.accept and dec4.kind == "downweighted"
    assert abs(dec4.weight - 1.0 / (1.0 + 0.5 * 4)) < 1e-5

    # Exactly at S_max: downweighted
    dec8 = decide_staleness(8, config)
    assert dec8.accept and dec8.kind == "downweighted"

    # Beyond S_max: rejected
    dec9 = decide_staleness(9, config)
    assert not dec9.accept and dec9.weight == 0.0 and dec9.kind == "rejected"
