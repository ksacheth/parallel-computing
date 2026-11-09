"""Client communication manager for worker nodes."""
from __future__ import annotations

import uuid
from typing import Any

import torch
import torch.nn as nn

from torch_asgc.transport.base import (
    BucketPayload,
    FetchRequest,
    FetchResponse,
    PushAck,
    PushUpdate,
    QuantizedDelta,
    TransportClient,
    parse_tcp_address,
)
from torch_asgc.transport.local import LocalTransportClient, LocalTransportHub
from torch_asgc.transport.tcp import TCPTransportClient


class ASGCClient:
    """Manages communication between a worker node and the parameter server."""

    def __init__(self, transport: TransportClient, worker_id: str | None = None) -> None:
        self.transport = transport
        self.worker_id = worker_id if worker_id is not None else str(uuid.uuid4())[:8]
        self.known_version = -1
        self.stop_requested = False

    def fetch_parameters(self) -> FetchResponse:
        """Request the latest model parameters or delta from the server."""
        req = FetchRequest(worker_id=self.worker_id, known_version=self.known_version)
        resp = self.transport.fetch(req)
        if resp.version >= 0:
            self.known_version = resp.version
        self.stop_requested = self.stop_requested or resp.stop
        return resp

    def sync_model(self, model: nn.Module) -> FetchResponse:
        """Fetch latest parameters/deltas from server and apply them to ``model``."""
        resp = self.fetch_parameters()
        self.apply_fetch_response(model, resp)
        return resp

    @staticmethod
    def apply_fetch_response(model: nn.Module, resp: FetchResponse) -> None:
        """Apply a FetchResponse (either quantized/dense delta or full snapshot) to ``model``."""
        if resp.is_delta and resp.delta is not None:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in resp.delta:
                        raw_d = resp.delta[name]
                        if isinstance(raw_d, QuantizedDelta):
                            d = raw_d.decode(device=p.device, dtype=p.dtype)
                        else:
                            d = raw_d.to(device=p.device, dtype=p.dtype)
                        p.add_(d)
                if resp.buffers:
                    for buf_name, buf in model.named_buffers():
                        if buf_name in resp.buffers:
                            buf.copy_(resp.buffers[buf_name].to(device=buf.device, dtype=buf.dtype))
        elif resp.params is not None:
            model.load_state_dict(resp.params)

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

    def close(self) -> None:
        self.transport.close()


def connect(
    address: str | tuple[str, int] | LocalTransportHub,
    worker_id: str | None = None,
    authkey: bytes | str | None = None,
    **tcp_options: Any,
) -> ASGCClient:
    """Convenience helper to initialize and connect an ASGCClient.

    ``authkey`` falls back to the ``ASGC_AUTHKEY`` environment variable, then to the
    library default. ``tcp_options`` (``connect_timeout``, ``connect_retries``,
    ``retry_delay``, ``reply_timeout``) are forwarded to ``TCPTransportClient``.
    """
    resolved_id = worker_id if worker_id is not None else str(uuid.uuid4())[:8]
    if isinstance(address, LocalTransportHub):
        client_transport = LocalTransportClient(address, worker_id=resolved_id)
        return ASGCClient(client_transport, worker_id=resolved_id)

    host, port = parse_tcp_address(address)
    return ASGCClient(TCPTransportClient((host, port), authkey=authkey, **tcp_options), worker_id=resolved_id)
