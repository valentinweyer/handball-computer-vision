"""The SAM2 accelerations must reproduce the eager path bit for bit, or step aside."""
import os
from pathlib import Path
import sys

import pytest
import torch

from handball_cv.tracking import sam2_acceleration as accel

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def transformer(monkeypatch):
    """Upstream SAM2's transformer module, with apply_rotary_enc restored afterwards."""
    upstream = Path(os.environ.get("SAM2_UPSTREAM_DIR", Path(__file__).resolve().parents[2] / "sam2-upstream"))
    monkeypatch.syspath_prepend(str(upstream))
    module = pytest.importorskip("sam2.modeling.sam.transformer")
    pytest.importorskip("triton")
    monkeypatch.setattr(module, "apply_rotary_enc", module.apply_rotary_enc)
    return module


def rope_inputs(memory_frames):
    from sam2.modeling.position_encoding import compute_axial_cis

    torch.manual_seed(0)
    freqs = compute_axial_cis(dim=256, end_x=64, end_y=64, theta=10000.0).cuda()
    q = torch.randn(1, 1, 4096, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 4096 * memory_frames, 256, device="cuda", dtype=torch.bfloat16)
    return q, k, freqs


@cuda
@pytest.mark.filterwarnings("error::RuntimeWarning")  # a self-disabled kernel must fail, not pass silently
@pytest.mark.parametrize("memory_frames,repeat", [(1, False), (1, True), (7, True)])
def test_fused_rope_is_bit_identical_to_upstream(transformer, memory_frames, repeat):
    upstream = transformer.apply_rotary_enc
    q, k, freqs = rope_inputs(memory_frames)
    expected = upstream(q, k, freqs, repeat_freqs_k=repeat)

    assert accel.install_fused_rope()
    fused = transformer.apply_rotary_enc
    assert fused is not upstream
    for _ in range(2):  # first call is self-checked, the second is the plain fast path
        got = fused(q, k, freqs, repeat_freqs_k=repeat)
        assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])


@cuda
def test_fused_rope_defers_what_it_cannot_reproduce(transformer):
    calls = []

    def previous(xq, xk, freqs_cis, repeat_freqs_k=False):
        calls.append(xq.dtype)
        return xq, xk

    transformer.apply_rotary_enc = previous
    assert accel.install_fused_rope()
    q, k, freqs = rope_inputs(1)
    transformer.apply_rotary_enc(q.float(), k.float(), freqs)  # FP32: not this kernel's job
    transformer.apply_rotary_enc(q, k[..., ::2, :], freqs, repeat_freqs_k=True)  # non-contiguous
    assert calls == [torch.float32, torch.bfloat16]


@cuda
def test_fused_rope_disables_itself_when_not_identical(transformer):
    upstream = transformer.apply_rotary_enc

    def drifted(xq, xk, freqs_cis, repeat_freqs_k=False):
        xq_out, xk_out = upstream(xq, xk, freqs_cis, repeat_freqs_k)
        return xq_out, xk_out + 1  # stands in for a stack whose FMA order differs

    transformer.apply_rotary_enc = drifted
    assert accel.install_fused_rope()
    q, k, freqs = rope_inputs(1)
    with pytest.warns(RuntimeWarning, match="not bit-identical"):
        got = transformer.apply_rotary_enc(q, k, freqs)
    expected = drifted(q, k, freqs)
    assert torch.equal(got[1], expected[1])
    # Disabled for good: later calls, new shapes included, take the previous path.
    q2, k2, _ = rope_inputs(2)
    assert torch.equal(transformer.apply_rotary_enc(q2, k2, freqs, True)[1],
                       drifted(q2, k2, freqs, True)[1])


class Small(torch.nn.Module):
    """Launch-heavy stand-in: many tiny kernels, a dict-of-list output."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, x, scale, flag=False):
        y = x
        for _ in range(20):
            y = torch.nn.functional.gelu(self.proj(y)) * scale
        return {"out": y, "parts": [y.sum(-1), x + 1 if flag else x * 2]}


class Branchy(torch.nn.Module):
    def forward(self, x):
        return x * 2 if bool(x.sum() > 0) else x * 3  # host sync: not capturable


def run(module, *args, **kwargs):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return module(*args, **kwargs)


@cuda
def test_graph_replay_matches_eager_and_returns_fresh_tensors():
    torch.manual_seed(0)
    module = Small().cuda().eval()
    eager = module.forward
    module.forward = graphs = accel.SmallModuleGraphs(module)
    scale = torch.tensor(0.5, device="cuda")
    outputs = []
    for step in range(5):
        x = torch.randn(64, 16, device="cuda") + step
        got = run(module, x, scale, flag=bool(step % 2))
        want = run(eager, x, scale, flag=bool(step % 2))
        assert torch.equal(got["out"], want["out"])
        assert all(torch.equal(a, b) for a, b in zip(got["parts"], want["parts"]))
        outputs.append(got)
    assert graphs.counts["captured"] == 2  # one graph per flag value
    assert graphs.counts["replays"] >= 3
    # Returned tensors are clones: later replays must not overwrite earlier results.
    first = outputs[0]["out"].clone()
    run(module, torch.randn(64, 16, device="cuda"), scale)
    assert torch.equal(outputs[0]["out"], first)


@cuda
def test_graphs_fall_back_to_eager():
    module = Small().cuda().eval()
    module.forward = graphs = accel.SmallModuleGraphs(module)
    x = torch.randn(4, 16, device="cuda")
    scale = torch.tensor(1.0, device="cuda")
    for _ in range(3):
        with torch.no_grad():  # not inference mode
            module(x, scale)
    assert graphs.counts["eager"] == 3 and not graphs.entries

    branchy = Branchy().cuda().eval()
    eager = branchy.forward
    branchy.forward = graphs = accel.SmallModuleGraphs(branchy)
    with pytest.warns(RuntimeWarning, match="fell back to eager"):
        for _ in range(3):
            got = run(branchy, -x)
    assert torch.equal(got, run(eager, -x))
    assert graphs.counts["capture_failed"] == 1 and not graphs.entries


def test_install_skips_other_predictors():
    assert accel.install_small_module_graphs(object()) == {}
