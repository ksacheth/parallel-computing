"""Standalone ParameterServer daemon for ASGC."""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torch_asgc.compression import CompressionConfig, make_compressor
from torch_asgc.controller import AdaptiveConfig, AdaptiveController, TelemetryWindow
from torch_asgc.engine.bucketing import create_buckets
from torch_asgc.engine.optimizer import StatefulServerOptimizer
from torch_asgc.staleness import StalenessConfig, decide_staleness
from torch_asgc.transport.base import (
    FetchResponse,
    PushAck,
    QuantizedDelta,
    TransportServer,
    parse_tcp_address,
)
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
        compression: CompressionConfig | None = None,
    ) -> None:
        self.model = model
        self.version = 0
        self.running = True

        # Initialize transport
        if isinstance(transport, LocalTransportHub):
            self.transport: TransportServer = LocalTransportServer(transport)
        elif transport is not None:
            self.transport = transport
        elif isinstance(bind_address, LocalTransportHub):
            self.transport = LocalTransportServer(bind_address)
        elif bind_address is not None:
            host, port = parse_tcp_address(bind_address)
            self.transport = TCPTransportServer((host, port))
        else:
            raise ValueError("must provide transport or valid bind_address")

        # Configuration
        self.staleness_config = staleness_config or StalenessConfig()
        if compression is not None:
            self.compression_config = compression
        else:
            norm_mode = "quantize" if compression_mode.lower() == "qsgd" else compression_mode.lower()
            rho_val = float(initial_strength) if norm_mode == "topk" else 0.1
            bits_val = (
                int(round(initial_strength))
                if norm_mode == "quantize" and initial_strength >= 2.0
                else 8
            )
            self.compression_config = CompressionConfig(
                mode=norm_mode,
                rho=rho_val,
                bits=bits_val,
            )
        self.compressor = make_compressor(self.compression_config)

        # Optimization & Bucketing
        self.trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.server_optimizer = StatefulServerOptimizer(optimizer, self.trainable_params)
        self.buckets = create_buckets(model, bucket_cap_mb=bucket_cap_mb)

        # Controller
        self.adaptive_config = adaptive_config
        self.controller = (
            AdaptiveController(adaptive_config, compression_mode=self.compression_config.mode)
            if adaptive_config and adaptive_config.enabled
            else None
        )
        self.telemetry = TelemetryWindow()
        self._last_control_time = time.perf_counter()

        # Downlink Delta History Buffer & Server-Side Error Feedback Residuals
        self.enable_delta_broadcast = True
        self.max_delta_history = max(8, self.staleness_config.s_max * 2)
        self._delta_history: dict[int, dict[str, torch.Tensor]] = {}
        self._delta_residuals: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, dtype=torch.float32) for name, p in self.trainable_params
        }

    @property
    def compression_mode(self) -> str:
        return self.compression_config.mode

    @property
    def current_strength(self) -> float:
        return self.compression_config.strength

    @current_strength.setter
    def current_strength(self, value: float) -> None:
        self.compression_config = self.compression_config.with_strength(value)
        self.compressor = make_compressor(self.compression_config)

    def _make_snapshot_response(self) -> FetchResponse:
        snapshot = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return FetchResponse(
            version=self.version,
            params=snapshot,
            is_delta=False,
            compression_setting=self.current_strength,
            s_max=self.staleness_config.s_max,
        )

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
                    is_delta=False,
                    compression_setting=self.current_strength,
                    s_max=self.staleness_config.s_max,
                )
            elif (
                self.enable_delta_broadcast
                and fetch_req.known_version >= 0
                and fetch_req.known_version in self._delta_history
            ):
                # Worker can be caught up using accumulated quantized deltas
                accumulated_delta: dict[str, torch.Tensor] | None = {}
                for v in range(fetch_req.known_version, self.version):
                    v_delta = self._delta_history.get(v)
                    if v_delta is None:
                        accumulated_delta = None
                        break
                    for name, d_tensor in v_delta.items():
                        if name not in accumulated_delta:
                            accumulated_delta[name] = d_tensor.clone()
                        else:
                            accumulated_delta[name].add_(d_tensor)

                if accumulated_delta is not None:
                    quantized_deltas: dict[str, QuantizedDelta | torch.Tensor] = {
                        name: QuantizedDelta.from_tensor(acc_t)[0]
                        for name, acc_t in accumulated_delta.items()
                    }
                    buffers = {
                        k: v.detach().cpu().clone() for k, v in self.model.named_buffers()
                    }
                    resp = FetchResponse(
                        version=self.version,
                        delta=quantized_deltas,
                        buffers=buffers if buffers else None,
                        is_delta=True,
                        compression_setting=self.current_strength,
                        s_max=self.staleness_config.s_max,
                    )
                else:
                    resp = self._make_snapshot_response()
            else:
                resp = self._make_snapshot_response()
            self.transport.send_fetch_response(fetch_req.worker_id, resp)

        # 2. Handle Push updates
        push_up = self.transport.poll_push(timeout=0.0)
        if push_up is not None:
            worked = True
            tau = max(0, self.version - push_up.version)
            decision = decide_staleness(tau, self.staleness_config)

            if decision.accept:
                old_params = None
                if self.enable_delta_broadcast:
                    old_params = {name: p.detach().clone() for name, p in self.trainable_params}

                # Decode and combine gradients from all buckets in this logical step
                unflattened_step_grads: dict[str, torch.Tensor] = {}
                for b_item in push_up.buckets:
                    bucket = self.buckets[b_item.bucket_index]
                    decoded_flat = self.compressor.decode(b_item.payload, target_device=bucket.device)
                    unflattened_bucket = bucket.unflatten_to_dict(decoded_flat)
                    unflattened_step_grads.update(unflattened_bucket)

                # Apply all bucket gradients together with staleness downweighting
                self.server_optimizer.step_with_gradients(unflattened_step_grads, weight=decision.weight)

                if self.enable_delta_broadcast and old_params is not None:
                    step_delta: dict[str, torch.Tensor] = {}
                    with torch.no_grad():
                        for name, p in self.trainable_params:
                            raw_delta = (p.detach() - old_params[name]).to(torch.float32) + self._delta_residuals[name]
                            _, reconstructed = QuantizedDelta.from_tensor(raw_delta)
                            self._delta_residuals[name] = (raw_delta - reconstructed).detach()
                            # Snap authoritative parameter to reconstructed INT8 step so worker & server never drift
                            p.copy_(old_params[name] + reconstructed.to(device=p.device, dtype=p.dtype))
                            step_delta[name] = reconstructed.cpu().to(dtype=p.dtype)

                    self._delta_history[self.version] = step_delta
                    min_keep = self.version - self.max_delta_history
                    self._delta_history = {v: d for v, d in self._delta_history.items() if v >= min_keep}

                # Increment server version exactly once for the entire logical step
                self.version += 1

            # Telemetry logging for the logical step
            self.telemetry.add_update(
                staleness=tau,
                accepted=decision.accept,
                num_bytes=push_up.total_bytes,
            )
            if push_up.loss_val is not None:
                self.telemetry.add_loss(push_up.loss_val)

            # Check adaptive controller
            now = time.perf_counter()
            if self.controller and (now - self._last_control_time) >= self.adaptive_config.interval_s:
                self.telemetry.duration_s = max(now - self._last_control_time, 1e-6)
                ctrl_decision = self.controller.decide(
                    self.telemetry,
                    current_strength=self.current_strength,
                    current_s_max=self.staleness_config.s_max,
                )
                self.current_strength = ctrl_decision.new_compression_strength
                self.staleness_config.s_max = ctrl_decision.new_s_max
                self.telemetry = TelemetryWindow()
                self._last_control_time = now

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
