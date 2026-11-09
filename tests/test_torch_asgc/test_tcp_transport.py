import socket
import time
import torch
from torch_asgc.transport.base import BucketPayload, FetchRequest, PushUpdate
from torch_asgc.transport.tcp import TCPTransportClient, TCPTransportServer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_tcp_transport_push_and_fetch():
    port = _find_free_port()
    address = ("127.0.0.1", port)
    authkey = b"test_secret"

    server = TCPTransportServer(address, authkey=authkey)
    time.sleep(0.1)  # allow listener to start

    client = TCPTransportClient(address, authkey=authkey)

    try:
        # Test push
        update = PushUpdate(
            worker_id="worker_tcp_1",
            version=1,
            buckets=[
                BucketPayload(
                    bucket_index=0,
                    payload=[1, 2, 3],
                    payload_bytes=12,
                    residual_norm=0.1,
                )
            ],
            loss_val=0.42,
        )

        # Worker pushes in background thread
        import threading
        ack_received = []

        def client_push():
            ack = client.push(update)
            ack_received.append(ack)

        t = threading.Thread(target=client_push)
        t.start()

        # Server polls
        server_update = None
        for _ in range(50):
            server_update = server.poll_push()
            if server_update is not None:
                break
            time.sleep(0.02)

        assert server_update is not None
        assert server_update.worker_id == "worker_tcp_1"
        assert server_update.loss_val == 0.42

        # Server replies
        from torch_asgc.transport.base import PushAck
        server.send_push_ack(
            "worker_tcp_1",
            PushAck(status="accepted", weight=1.0, current_version=2),
        )

        t.join(timeout=2.0)
        assert len(ack_received) == 1
        assert ack_received[0].status == "accepted"
        assert ack_received[0].current_version == 2

    finally:
        client.close()
        server.close()


def test_bad_authkey_does_not_stop_accepting_workers():
    """A wrong-key client (or port scanner) must not kill the accept loop."""
    import pytest

    port = _find_free_port()
    address = ("127.0.0.1", port)
    server = TCPTransportServer(address, authkey=b"right_key")
    try:
        with pytest.raises(Exception):
            TCPTransportClient(address, authkey=b"wrong_key", connect_timeout=5.0)

        # Raw garbage connection that never completes the handshake
        junk = socket.create_connection(address)
        junk.sendall(b"GET / HTTP/1.0\r\n\r\n")

        client = TCPTransportClient(address, authkey=b"right_key", connect_timeout=5.0)
        client.close()
        junk.close()
    finally:
        server.close()


def test_server_signals_stop_and_codec_to_workers():
    import threading

    import torch.nn as nn

    from torch_asgc import ParameterServer, connect

    port = _find_free_port()
    server = ParameterServer(
        model=nn.Linear(3, 1),
        optimizer=torch.optim.SGD(nn.Linear(3, 1).parameters(), lr=0.1),
        bind_address=("127.0.0.1", port),
        authkey="secret",
        compression_mode="quantize",
        initial_strength=4,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = connect(f"tcp://127.0.0.1:{port}", worker_id="w", authkey="secret", reply_timeout=5.0)
    try:
        resp = client.fetch_parameters()
        assert resp.compression_mode == "quantize"
        assert resp.compression_setting == 4.0
        assert not client.stop_requested

        server.stop_training = True
        client.fetch_parameters()
        assert client.stop_requested
    finally:
        client.close()
        server.close()


def test_payload_tensors_are_moved_to_cpu_for_the_wire():
    from torch_asgc.compression import SparsePiece
    from torch_asgc.transport.tcp import _to_cpu

    piece = SparsePiece(indices=torch.tensor([1], dtype=torch.int32), values=torch.tensor([2.0]), total_numel=4)
    update = PushUpdate(worker_id="w", version=0, buckets=[BucketPayload(0, piece, 8)])
    moved = _to_cpu(("push", update))
    assert moved[0] == "push"
    assert moved[1].buckets[0].payload.total_numel == 4
    assert moved[1].buckets[0].payload.values.device.type == "cpu"


def test_wire_format_carries_tensor_data_by_value():
    """Tensors must not be pickled as shared-memory fd handles (they only resolve on the sending host)."""
    from torch_asgc.transport.base import FetchResponse
    from torch_asgc.transport.tcp import decode_message, encode_message

    resp = FetchResponse(version=3, params={"w": torch.arange(6.0).reshape(2, 3)})
    data = encode_message(resp)
    assert b"torch.multiprocessing" not in data
    assert b"rebuild_storage_fd" not in data
    restored = decode_message(data)
    assert torch.equal(restored.params["w"], resp.params["w"])

    # Contrast: the ForkingPickler path that Connection.send uses does embed fd handles
    from multiprocessing.reduction import ForkingPickler

    assert b"torch.multiprocessing" in bytes(ForkingPickler.dumps(resp))
