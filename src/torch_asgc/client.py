"""Client communication manager for worker nodes."""
from __future__ import annotations

import uuid
from typing import Any

from torch_asgc.transport.base import (
    BucketPayload,
    FetchRequest,
    FetchResponse,
    PushAck,
    PushUpdate,
    TransportClient,
)
from torch_asgc.transport.local import LocalTransportClient, LocalTransportHub
from torch_asgc.transport.tcp import TCPTransportClient


class ASGCClient:
    """Manages communication between a worker node and the parameter server."""

    def __init__(self, transport: TransportClient, worker_id: str | None = None) -> None:
        self.transport = transport
        self.worker_id = worker_id if worker_id is not None else str(uuid.uuid4())[:8]
        self.known_version = -1

    def fetch_parameters(self) -> FetchResponse:
        """Request the latest model parameters from the server."""
        req = FetchRequest(worker_id=self.worker_id, known_version=self.known_version)
        resp = self.transport.fetch(req)
        if resp.version >= 0:
            self.known_version = resp.version
        return resp

    def push_step(
        self,
        version: int,
        buckets: list[BucketPayload],
        loss_val: float | None = None,
    ) -> PushAck:
        """Push a unified logical step containing all buckets to the server."""
        update = PushUpdate(
            worker_id=self.worker_id,
            version=version,
            buckets=buckets,
            loss_val=loss_val,
        )
        return self.transport.push(update)

    def push_update(
        self,
        bucket_index: int,
        version: int,
        payload: Any,
        payload_bytes: int,
        loss_val: float | None = None,
        residual_norm: float = 0.0,
    ) -> PushAck:
        """Push a single compressed gradient bucket (convenience method)."""
        bucket = BucketPayload(
            bucket_index=bucket_index,
            payload=payload,
            payload_bytes=payload_bytes,
            residual_norm=residual_norm,
        )
        return self.push_step(version=version, buckets=[bucket], loss_val=loss_val)

    def close(self) -> None:
        self.transport.close()


def connect(
    address: str | tuple[str, int] | LocalTransportHub,
    worker_id: str | None = None,
    authkey: bytes = b"torch_asgc",
) -> ASGCClient:
    """Convenience helper to initialize and connect an ASGCClient."""
    if isinstance(address, LocalTransportHub):
        client_transport = LocalTransportClient(address, worker_id=worker_id or str(uuid.uuid4())[:8])
        return ASGCClient(client_transport, worker_id=worker_id)

    if isinstance(address, tuple):
        host, port = address
        return ASGCClient(TCPTransportClient((host, port), authkey=authkey), worker_id=worker_id)

    if isinstance(address, str):
        # Format "tcp://host:port" or "host:port"
        cleaned = address.replace("tcp://", "")
        host, port_str = cleaned.split(":")
        return ASGCClient(TCPTransportClient((host, int(port_str)), authkey=authkey), worker_id=worker_id)

    raise ValueError(f"unsupported address format: {address!r}")
