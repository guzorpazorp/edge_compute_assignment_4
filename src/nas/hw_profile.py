"""Hardware profiling of a candidate model on the CPU it runs on.

* latency_ms   : median batch-1 CPU latency over `runs` timed forward passes (after warm-up),
                 with a fixed intra-op thread count so numbers are comparable across trials.
* size_mb      : bytes of the serialised fp32 state_dict (what would be stored on device).
* peak_mem_mb  : peak CPU memory allocated during one inference, from the PyTorch profiler's
                 allocation/free events ("where supported"; None if the profiler is unavailable).
* params, mflops (MACs x 2 approximated via forward hooks on conv/linear layers).
"""
import io
import platform
import statistics
import time

import torch
import torch.nn as nn


def _cpu_copy(model):
    import copy
    return copy.deepcopy(model).cpu().float().eval()


def measure_latency(model, resolution, threads=1, warmup=10, runs=50):
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, resolution, resolution)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(prev)
    return statistics.median(times)


def measure_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def measure_peak_memory_mb(model, resolution):
    try:
        from torch.profiler import ProfilerActivity, profile
        x = torch.randn(1, 3, resolution, resolution)
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            model(x)
        events = sorted((e for e in prof.events() if e.cpu_memory_usage != 0),
                        key=lambda e: e.time_range.start)
        cur = peak = 0
        for e in events:
            cur += e.cpu_memory_usage
            peak = max(peak, cur)
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        return (peak + weights) / 2**20  # activations + resident weights
    except Exception:
        return None


def measure_mflops(model, resolution):
    flops = 0

    def hook(m, inp, out):
        nonlocal flops
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            flops += 2 * k * out.numel()
        elif isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.inference_mode():
        model(torch.randn(1, 3, resolution, resolution))
    for h in hooks:
        h.remove()
    return flops / 1e6


def profile_model(model, resolution, threads=1, runs=50):
    m = _cpu_copy(model)
    return {
        "params": sum(p.numel() for p in m.parameters()),
        "mflops": round(measure_mflops(m, resolution), 2),
        "size_mb": round(measure_size_mb(m), 4),
        "latency_ms": round(measure_latency(m, resolution, threads, runs=runs), 4),
        "peak_mem_mb": (lambda v: None if v is None else round(v, 3))(measure_peak_memory_mb(m, resolution)),
    }


def check_constraints(hw, constraints):
    """Returns a list of human-readable violations (empty list = deployable)."""
    checks = [("latency_ms", "max_latency_ms"), ("size_mb", "max_size_mb"), ("peak_mem_mb", "max_peak_mem_mb")]
    out = []
    for metric, limit in checks:
        if constraints.get(limit) is not None and hw.get(metric) is not None and hw[metric] > constraints[limit]:
            out.append(f"{metric}={hw[metric]:.3f}>{constraints[limit]}")
    return out


def machine_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": cpu, "platform": platform.platform(), "torch": torch.__version__}
