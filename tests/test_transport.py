from asgc.transport import PushUpdate, push_delay_s


def _update(payload_bytes: int) -> PushUpdate:
    return PushUpdate(worker_id=0, version=1, payload=[], payload_bytes=payload_bytes)


def test_bandwidth_charges_bytes_proportionally():
    # 8 Mbit = 1 MB at 8 Mbps takes exactly one second
    assert push_delay_s(_update(1_000_000), comm_delay_s=0.0, bandwidth_bps=1_000_000.0) == 1.0


def test_flat_delay_and_bandwidth_add_up():
    delay = push_delay_s(_update(2_000_000), comm_delay_s=0.5, bandwidth_bps=4_000_000.0)
    assert delay == 0.5 + 0.5


def test_unlimited_bandwidth_charges_nothing():
    assert push_delay_s(_update(44_000_000), comm_delay_s=0.0, bandwidth_bps=0.0) == 0.0
