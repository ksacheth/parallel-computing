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
