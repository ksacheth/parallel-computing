"""Standalone ParameterServer daemon for ASGC."""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torch_asgc.compression import make_compressor
from torch_asgc.controller import AdaptiveConfig, AdaptiveController, TelemetryWindow
from torch_asgc.engine.bucketing import create_buckets
from torch_asgc.engine.optimizer import StatefulServerOptimizer
from torch_asgc.staleness import StalenessConfig, decide_staleness
from torch_asgc.transport.base import FetchResponse, PushAck, TransportServer
from torch_asgc.transport.local import LocalTransportHub, LocalTransportServer
from torch_asgc.transport.tcp import TCPTransportServer


class ParameterServer:
    """Authoritative Parameter Server managing model weights and applying updates."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        transport: TransportServer | None = None,
        bind_address: str | tuple[str, int] | LocalTransportHub | None = None,
        staleness_config: StalenessConfig | None = None,
        compression_mode: str = "topk",
        initial_strength: float = 0.1,
        adaptive_config: AdaptiveConfig | None = None,
        bucket_cap_mb: float = 25.0,
    ) -> None:
        self.model = model
        self.version = 0
        self.running = True

        # Initialize transport
        if transport is not None:
            self.transport = transport
        elif isinstance(bind_address, LocalTransportHub):
            self.transport = LocalTransportServer(bind_address)
        elif isinstance(bind_address, tuple):
            host, port = bind_address
            self.transport = TCPTransportServer((host, port))
        elif isinstance(bind_address, str):
            cleaned = bind_address.replace("tcp://", "")
            host, port_str = cleaned.split(":")
            self.transport = TCPTransportServer((host, int(port_str)))
        else:
            raise ValueError("must provide transport or valid bind_address")

        # Configuration
        self.staleness_config = staleness_config or StalenessConfig()
        self.compression_mode = compression_mode
        self.current_strength = initial_strength
        self.compressor = make_compressor(
            compression_mode,
            rho=initial_strength,
            bits=int(initial_strength),
        )

        # Optimization & Bucketing
        self.trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.server_optimizer = StatefulServerOptimizer(optimizer, self.trainable_params)
        self.buckets = create_buckets(model, bucket_cap_mb=bucket_cap_mb)

        # Controller
        self.adaptive_config = adaptive_config
        self.controller = (
            AdaptiveController(adaptive_config, compression_mode=compression_mode)
            if adaptive_config and adaptive_config.enabled
            else None
        )
        self.telemetry = TelemetryWindow()
        self._last_control_time = time.perf_counter()

    def process_one(self) -> bool:
        """Poll and handle up to one fetch request and one push update.

        Returns True if any work was performed.
        """
        worked = False

        # 1. Handle Fetch requests
        fetch_req = self.transport.poll_fetch(timeout=0.0)
        if fetch_req is not None:
            worked = True
            if fetch_req.known_version == self.version:
                # Client already holds latest parameters
                resp = FetchResponse(
                    version=self.version,
                    params=None,
                    compression_setting=self.current_strength,
                    s_max=self.staleness_config.s_max,
                )
            else:
                # Snapshot parameters
                snapshot = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                resp = FetchResponse(
                    version=self.version,
                    params=snapshot,
                    compression_setting=self.current_strength,
                    s_max=self.staleness_config.s_max,
                )
            self.transport.send_fetch_response(fetch_req.worker_id, resp)

        # 2. Handle Push updates
        push_up = self.transport.poll_push(timeout=0.0)
        if push_up is not None:
            worked = True
            tau = max(0, self.version - push_up.version)
            decision = decide_staleness(tau, self.staleness_config)

            if decision.accept:
                # Decode update onto appropriate bucket
                bucket = self.buckets[push_up.bucket_index]
                decoded_flat = self.compressor.decode(push_up.payload, target_device=bucket.device)
                unflattened = bucket.unflatten_to_dict(decoded_flat)

                # Apply to server model with staleness downweighting
                self.server_optimizer.step_with_gradients(unflattened, weight=decision.weight)
                self.version += 1

            # Telemetry logging
            self.telemetry.add_update(
                staleness=tau,
                accepted=decision.accept,
                num_bytes=push_up.payload_bytes,
            )
            if push_up.loss_val is not None:
                self.telemetry.add_loss(push_up.loss_val)

            # Check adaptive controller
            if self.controller and (time.perf_counter() - self._last_control_time) >= self.adaptive_config.interval_s:
                ctrl_decision = self.controller.decide(
                    self.telemetry,
                    current_strength=self.current_strength,
                    current_s_max=self.staleness_config.s_max,
                )
                self.current_strength = ctrl_decision.new_compression_strength
                self.staleness_config.s_max = ctrl_decision.new_s_max
                self.compressor = make_compressor(
                    self.compression_mode,
                    rho=self.current_strength,
                    bits=int(self.current_strength),
                )
                self.telemetry = TelemetryWindow()
                self._last_control_time = time.perf_counter()

            # Acknowledge push
            ack = PushAck(
                status=decision.kind,
                weight=decision.weight,
                current_version=self.version,
                compression_setting=self.current_strength,
                s_max=self.staleness_config.s_max,
            )
            self.transport.send_push_ack(push_up.worker_id, ack)

        return worked

    def serve_forever(self, poll_interval_s: float = 0.001) -> None:
        """Run non-blocking server polling loop indefinitely."""
        while self.running:
            worked = self.process_one()
            if not worked and poll_interval_s > 0:
                time.sleep(poll_interval_s)

    def close(self) -> None:
        self.running = False
        self.transport.close()
