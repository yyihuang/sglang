"""Cake Mamba2 SSD combined prefill through sglang.kernels.

Prepared-runner pattern: ``cake_ssd_combined(...)`` constructs the FlashInfer
``SSDCombined(backend="cake")`` runner; ``run()`` is launched, then launched
again with new activations in the same caller-owned ``out`` buffer. Checks
that the registry resolves the explicit FlashInfer backends; that the facade
output and final states are bitwise identical to constructing the FlashInfer
runner directly and to the functional ``cake_ssd_combined_fwd``; and that
output, final states and selective checkpoint rows (PR #35444's compact
checkpoints) match FlashInfer's own validated oracle for this kernel, in
batched mode and in packed-varlen mode (``seq_idx`` + chunk metadata, or the
engine's ``cu_seqlens`` + its chunk size: the route's call, CAKE-934 item 2).

CAKE-956 revision of the contract (FlashInfer ``flashinfer/mamba/cake_ssd_combined.py``):
any packed token count (the last physical chunk may be partial), BF16 / FP16 /
FP32 state, the caller's token-major ``[B, S, nheads, 64]`` buffer as ``out``
(the engine passes its own), and varlen without ``initial_states`` (the
sequence count travels as ``num_seqs``). The single-chunk batches that
returned NaN before (CAKE-950) are covered below.

Oracle and bound (FlashInfer ``tests/mamba/test_cake_ssd_combined.py``,
``_assert_cake_accuracy``, CAKE-956 revision): both the Cake route and
FlashInfer's CuTe SSD backend (``SSDCombined(backend="cute")``) are measured
on identical inputs against an fp64 token-by-token SSM recurrence
(``_fp64_reference``). Cake must be finite, within ``atol = rtol = 1e-2`` of
the recurrence on all but at most 1 % of the entries, have no more entries
outside that tolerance than CuTe (up to the Poisson noise ``2 * sqrt(CuTe's
count)``) and no larger maximum error (within one bf16 ulp). Elementwise
Cake-vs-CuTe parity is no longer the oracle: the Cake kernels store the
per-token ``delta`` in FP16 (CAKE-942) while CuTe stores it in BF16, so the
two kernels round differently and their sparse outlier sets (the
cancellation in ``C . state`` amplifying one operand rounding) no longer
coincide, although Cake is the more accurate kernel. Checkpoint states are
held to the same rules against the fp64 state at the boundary, with the CuTe
final state of the sequence prefix as the comparison arm.

Checkpoint contract (``flashinfer/mamba/ssd_combined.py`` docstring + the
generated kernels ``mamba_ssd_q_tmem_alias_*``): ``checkpoint_token_indices``
holds one *exclusive* token boundary per sequence -- sequence-relative in
batched mode, absolute in the packed token axis for varlen -- and the state
is captured only when that boundary is the end of a logical chunk
(``checkpoint_token == segment_end``): a multiple of ``chunk_size`` in
batched mode, or a boundary exposed through ``chunk_indices`` /
``chunk_offsets`` in varlen mode (the packed shape SGLang builds). Boundaries
anywhere else, negative entries and negative slots capture nothing and leave
the slot untouched.

The runner materializes strided inputs into graph-stable storage and owns its
workspaces, so one runner per stream; preparation (first ``run`` per shape)
must happen eagerly before any CUDA-graph capture. Skips with the reason when
FlashInfer lacks the module or the GPU is outside sm_100a / sm_103a.
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import mamba as cake_mamba
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.mamba.cake import cake_ssd_combined, cake_ssd_combined_fwd
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

CHUNK = cake_mamba.SSD_CHUNK_SIZE  # the kernels' internal tiling (triple form)
ENGINE_CHUNK = 256  # Nemotron-H ``mamba_chunk_size``: the route's chunk_size label
HEADDIM = cake_mamba.SSD_HEADDIM
DSTATE = cake_mamba.SSD_DSTATE
NHEADS, NGROUPS = 16, 8
# Unused checkpoint slot: must stay untouched (NaN-filled) after the launch.
UNTOUCHED_SLOT = 1
_ATOL = _RTOL = 1e-2
_MAX_OUTSIDE_FRACTION = 0.01


@pytest.mark.parametrize("op", ["mamba.ssd_combined", "mamba.ssd_combined_fwd"])
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.mamba:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake_mamba.FI_MODULE, cake_mamba.FI_SSD_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.mamba.cake_ssd_combined")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_mamba.ARCHS:
        pytest.skip(f"Cake SSDCombined is built for sm_100a/sm_103a, device is {cc}")


def _varlen_metadata(lengths, device, extra_boundaries=()):
    """``seq_idx`` plus the logical-chunk metadata of a packed batch.

    A logical chunk starts at every physical chunk start and at every
    sequence start; ``extra_boundaries`` (absolute packed token indices) add
    further logical boundaries, which is how a caller exposes a checkpoint
    inside a physical chunk.
    """
    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=torch.int32, device=device)
    starts, start = [], 0
    for seq, n in enumerate(lengths):
        seq_idx[0, start : start + n] = seq
        starts.append(start)
        start += n
    boundaries = set(starts) | set(int(b) for b in extra_boundaries)
    chunk_indices, chunk_offsets = [], []
    for chunk in range(-(-total // CHUNK)):
        lo, hi = chunk * CHUNK, (chunk + 1) * CHUNK
        for offset in sorted({0} | {b - lo for b in boundaries if lo < b < hi}):
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device=device),
        torch.tensor(chunk_offsets, dtype=torch.int32, device=device),
    )


def _case(device, *, varlen, seed, lengths=None, initial_states=True, state_dtype=None):
    """Inputs of one Cake run. ``lengths``: per-sequence token counts (packed
    for varlen; equal rows for batched). ``initial_states=False`` is the
    engine's "no prefix" call (``None`` plus ``num_seqs`` in varlen mode)."""
    torch.manual_seed(seed)
    if lengths is None:
        lengths = (96, 160) if varlen else (256, 256)
    if varlen:
        batch, seqlen = 1, sum(lengths)
    else:
        assert len(set(lengths)) == 1
        batch, seqlen = len(lengths), lengths[0]
    state_dtype = torch.bfloat16 if state_dtype is None else state_dtype
    x = torch.randn(batch, seqlen, NHEADS, HEADDIM, device=device).bfloat16()
    dt = torch.randn(batch, seqlen, NHEADS, device=device)
    A = -torch.rand(NHEADS, device=device) - 1.0
    B = torch.randn(batch, seqlen, NGROUPS, DSTATE, device=device).bfloat16()
    C = torch.randn_like(B)
    D = torch.randn(NHEADS, device=device).bfloat16()
    z = torch.randn_like(x)
    dt_bias = torch.rand(NHEADS, device=device) - 4.0
    if initial_states:
        # BF16-representable values: the BF16-state CuTe oracle then starts
        # from exactly the states an FP16- / FP32-state Cake run starts from.
        states = (
            (torch.randn(len(lengths), NHEADS, HEADDIM, DSTATE, device=device) * 0.1)
            .bfloat16()
            .to(state_dtype)
        )
    else:
        states = None
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, device)
    else:
        seq_idx = chunk_indices = chunk_offsets = None
    # The engine's token-major activation buffer is the kernel's out.
    out = torch.empty(
        batch, seqlen, NHEADS, HEADDIM, device=device, dtype=torch.bfloat16
    )
    return dict(
        tensors=(x, dt, A, B, C),
        lengths=lengths,
        run=dict(
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            initial_states=states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            num_seqs=len(lengths) if varlen and states is None else None,
            return_final_states=True,
        ),
        ctor=dict(
            chunk_size=CHUNK,
            nheads=NHEADS,
            headdim=HEADDIM,
            dstate=DSTATE,
            ngroups=NGROUPS,
            io_dtype=torch.bfloat16,
            state_dtype=state_dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=states is not None,
            has_varlen=varlen,
            has_z=True,
            seq_idx_dtype=torch.int32,
        ),
        out=out,
    )


def _cute_reference(case):
    """FlashInfer's validated oracle for the Cake route: its CuTe SSD backend.

    CuTe keeps the original contract (BF16 state, explicit initial states,
    ``seqlen % 128 == 0``), so the Cake case is mapped onto it: missing
    initial states become zeros, states are cast to BF16, and a packed token
    count off the 128 grid is padded with one extra zero sequence whose rows
    and final state are dropped from the comparison.
    """
    from flashinfer.mamba import SSDCombined

    x, dt, A, B, C = case["tensors"]
    run = {key: value for key, value in case["run"].items() if key != "num_seqs"}
    ctor = {**case["ctor"], "state_dtype": torch.bfloat16, "has_initial_states": True}
    num_seqs = len(case["lengths"])
    states = run["initial_states"]
    if states is None:
        states = torch.zeros(
            num_seqs, NHEADS, HEADDIM, DSTATE, device=x.device, dtype=torch.bfloat16
        )
    run["initial_states"] = states.bfloat16()
    seqlen = x.shape[1]
    pad = (-seqlen) % CHUNK
    if pad:
        assert case["ctor"]["has_varlen"], "batched padding is not modelled here"

        def pad_tokens(tensor):
            return torch.nn.functional.pad(
                tensor, (0, 0) * (tensor.ndim - 2) + (0, pad)
            )

        lengths = tuple(case["lengths"]) + (pad,)
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, x.device)
        run.update(
            seq_idx=seq_idx, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets
        )
        run["initial_states"] = torch.cat(
            [run["initial_states"], torch.zeros_like(run["initial_states"][:1])]
        )
        x, dt, B, C = (pad_tokens(t) for t in (x, dt, B, C))
        run["z"] = pad_tokens(run["z"])
    out, final = SSDCombined(**ctor, backend="cute").run(x, dt, A, B, C, **run)
    return out[:, :seqlen], final[:num_seqs]


def _fp64_reference(case, checkpoints=()):
    """fp64 token-by-token SSM recurrence, the oracle both backends are measured
    against (FlashInfer ``_fp64_reference``): ``dt' = clamp(softplus(dt +
    dt_bias), dt_limit)``, ``state = exp(dt' * A) * state + dt' * (x (x) B)``,
    ``y = C . state + D * x``, ``y *= z * sigmoid(z)``. Returns the token-major
    output with ``x``'s shape, the ``[num_seqs, nheads, 64, 128]`` final
    states (zero initial state when ``initial_states`` is ``None``) and the
    state after each absolute exclusive token boundary in ``checkpoints``."""
    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    lengths = tuple(case["lengths"])
    batch, seqlen, nheads, headdim = x.shape
    total = batch * seqlen
    assert sum(lengths) == total, (lengths, total)
    ngroups, dstate = B.shape[2], B.shape[3]
    rep = nheads // ngroups
    f64 = torch.float64
    xf = x.reshape(total, nheads, headdim).to(f64)
    dtf = dt.reshape(total, nheads).to(f64)
    if run.get("dt_bias") is not None:
        dtf = dtf + run["dt_bias"].to(f64)
    if run.get("dt_softplus", False):
        dtf = torch.nn.functional.softplus(dtf)
    dt_min, dt_max = run.get("dt_limit", (0.0, float("inf")))
    dtf = dtf.clamp(min=float(dt_min), max=float(dt_max))
    decay = torch.exp(A.to(f64)[None, :] * dtf)
    Bf = B.reshape(total, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    Cf = C.reshape(total, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    D = run.get("D")
    if D is None:
        Df = torch.zeros((nheads, 1), dtype=f64, device=x.device)
    else:
        Df = D.to(f64)
        Df = Df[:, None] if Df.ndim == 1 else Df
        if not case["ctor"]["d_has_hdim"]:
            Df = Df[:, :1]
    initial = run.get("initial_states")
    y = torch.empty((total, nheads, headdim), dtype=f64, device=x.device)
    states = torch.empty(
        (len(lengths), nheads, headdim, dstate), dtype=f64, device=x.device
    )
    captured = {}
    start = 0
    for sequence, length in enumerate(lengths):
        if initial is None:
            state = torch.zeros((nheads, headdim, dstate), dtype=f64, device=x.device)
        else:
            state = initial[sequence].to(f64).clone()
        for token in range(start, start + length):
            state = state * decay[token][:, None, None] + (
                (dtf[token][:, None] * xf[token])[:, :, None] * Bf[token][:, None, :]
            )
            y[token] = torch.einsum("hdn,hn->hd", state, Cf[token]) + Df * xf[token]
            if token + 1 in checkpoints:
                captured[token + 1] = state.clone()
        states[sequence] = state
        start += length
    z = run.get("z")
    if z is not None:
        zf = z.reshape(total, nheads, headdim).to(f64)
        y = y * (zf * torch.sigmoid(zf))
    return y.reshape(x.shape), states, captured


def _outside_tolerance(actual, reference):
    """``|actual - reference| > atol + rtol * |reference|`` elementwise."""
    difference = (actual.to(torch.float64) - reference).abs()
    return difference > _ATOL + _RTOL * reference.abs()


def _bf16_ulp(value):
    """Spacing of bf16 values at magnitude ``value`` (smallest normal below)."""
    magnitude = max(abs(value), torch.finfo(torch.bfloat16).tiny)
    return 2.0 ** math.floor(math.log2(magnitude)) * torch.finfo(torch.bfloat16).eps


def _assert_accuracy(name, cake, cute, reference):
    """FlashInfer's ``_assert_cake_accuracy`` rules for one returned tensor:
    Cake finite; at most ``_MAX_OUTSIDE_FRACTION`` of its entries outside
    ``atol = rtol = 1e-2`` of the fp64 recurrence; no more such entries than
    CuTe on the same inputs (slack ``2 * sqrt(CuTe's count)``); and a maximum
    absolute error no larger than CuTe's by more than one bf16 ulp at the
    magnitude of Cake's worst entry."""
    assert tuple(cake.shape) == tuple(reference.shape) == tuple(cute.shape), (
        name,
        tuple(cake.shape),
        tuple(cute.shape),
        tuple(reference.shape),
    )
    cake64 = cake.to(torch.float64)
    assert torch.isfinite(cake64).all(), f"{name}: Cake output is not finite"
    cake_outside = int(_outside_tolerance(cake, reference).sum())
    cute_outside = int(_outside_tolerance(cute, reference).sum())
    budget = _MAX_OUTSIDE_FRACTION * reference.numel()
    assert cake_outside <= budget, (
        f"{name}: {cake_outside} of {reference.numel()} Cake entries outside "
        f"atol=rtol={_ATOL} of the fp64 recurrence (limit {budget:.0f}, CuTe {cute_outside})"
    )
    slack_count = 2.0 * math.sqrt(cute_outside)
    assert cake_outside <= cute_outside + slack_count, (
        f"{name}: Cake has {cake_outside} entries outside atol=rtol={_ATOL} of the "
        f"fp64 recurrence, CuTe {cute_outside} on the same inputs (slack {slack_count:.0f})"
    )
    cake_error = (cake64 - reference).abs()
    cute_error = (cute.to(torch.float64) - reference).abs()
    worst = int(cake_error.reshape(-1).argmax())
    cake_max = float(cake_error.reshape(-1)[worst])
    cute_max = float(cute_error.max())
    slack = _bf16_ulp(float(reference.reshape(-1)[worst]))
    assert cake_max <= cute_max + slack, (
        f"{name}: Cake max abs error {cake_max:.4g} exceeds CuTe's {cute_max:.4g} "
        f"by more than one bf16 ulp ({slack:.4g}) at the worst entry"
    )


def _assert_matches_reference(case, actual, cute, reference=None):
    """``out`` and ``final_states`` of a Cake run against the fp64 recurrence,
    with the CuTe run on the same inputs as the comparison arm."""
    if reference is None:
        reference = _fp64_reference(case)
    for index, name in enumerate(("out", "final_states")):
        _assert_accuracy(name, actual[index], cute[index], reference[index])


def _cute_prefix_final_state(case, *, batch_index, start, length, sequence):
    """CuTe final state after ``length`` tokens of one sequence (batched run)."""
    from flashinfer.mamba import SSDCombined

    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    sl = slice(start, start + length)
    prefix = {
        **run,
        "z": run["z"][batch_index : batch_index + 1, sl].contiguous(),
        "initial_states": run["initial_states"][sequence : sequence + 1].contiguous(),
        "seq_idx": None,
        "chunk_indices": None,
        "chunk_offsets": None,
    }
    ctor = {**case["ctor"], "has_varlen": False}
    _, final = SSDCombined(**ctor, backend="cute").run(
        x[batch_index : batch_index + 1, sl].contiguous(),
        dt[batch_index : batch_index + 1, sl].contiguous(),
        A,
        B[batch_index : batch_index + 1, sl].contiguous(),
        C[batch_index : batch_index + 1, sl].contiguous(),
        **prefix,
    )
    return final[0]


def _supports(case, **extra):
    x, dt, A, B, C = case["tensors"]
    return cake_mamba.supports_ssd_combined(
        x,
        dt,
        A,
        B,
        C,
        D=case["run"]["D"],
        z=case["run"]["z"],
        dt_bias=case["run"]["dt_bias"],
        initial_states=case["run"]["initial_states"],
        seq_idx=case["run"]["seq_idx"],
        chunk_indices=extra.pop("chunk_indices", case["run"]["chunk_indices"]),
        chunk_offsets=extra.pop("chunk_offsets", case["run"]["chunk_offsets"]),
        num_seqs=case["run"]["num_seqs"],
        **extra,
    )


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_matches_flashinfer_functional_and_reference(varlen):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=varlen, seed=int(varlen))
    x, dt, A, B, C = case["tensors"]
    assert _supports(case, out=case["out"])
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(*case["tensors"], out=case["out"], **case["run"])
    assert out.untyped_storage().data_ptr() == case["out"].untyped_storage().data_ptr()
    from flashinfer.mamba import SSDCombined

    out_fi, final_fi = SSDCombined(**case["ctor"], backend="cake").run(
        *case["tensors"], **case["run"]
    )
    out_fn, final_fn = cake_ssd_combined_fwd(*case["tensors"], **case["run"])
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi) and torch.equal(final, final_fi)
    assert torch.equal(out, out_fn) and torch.equal(final, final_fn)
    assert torch.isfinite(out.float()).all() and torch.isfinite(final.float()).all()
    _assert_matches_reference(case, (out, final), _cute_reference(case))

    # Second run: new activations in the same storage, same out buffer.
    x.copy_(torch.randn_like(x.float()).bfloat16())
    case["run"]["z"].copy_(torch.randn_like(x.float()).bfloat16())
    out2, final2 = runner.run(*case["tensors"], out=case["out"], **case["run"])
    torch.cuda.synchronize()
    _assert_matches_reference(case, (out2, final2), _cute_reference(case))


@pytest.mark.parametrize(
    "lengths",
    [(128,), (128, 128), (72, 128), (1000,), (128, 900), (1,), (8,), (64, 60)],
    ids=[
        "one-chunk",
        "two-one-chunk-seqs",
        "partial-chunk",
        "long-partial",
        "two-partial",
        "short-1",
        "short-8",
        "short-64-60",
    ],
)
def test_runner_varlen_without_initial_states_any_length(lengths):
    """The engine's packed prefill call: no prefix (``initial_states=None``
    plus ``num_seqs``) on every sequence geometry, including the single-chunk
    batches that returned NaN before (CAKE-950), token counts off the 128
    grid and packed batches shorter than one chunk (CAKE-1063: the
    FlashInfer host zero-pads them to one chunk); the result lands in the
    caller's token-major buffer."""
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(
        device,
        varlen=True,
        seed=20 + len(lengths),
        lengths=lengths,
        initial_states=False,
    )
    assert _supports(case, out=case["out"])
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(*case["tensors"], out=case["out"], **case["run"])
    torch.cuda.synchronize()
    assert out.untyped_storage().data_ptr() == case["out"].untyped_storage().data_ptr()
    assert tuple(out.shape) == tuple(case["tensors"][0].shape)
    assert tuple(final.shape) == (len(lengths), NHEADS, HEADDIM, DSTATE)
    assert torch.isfinite(out.float()).all() and torch.isfinite(final.float()).all()
    _assert_matches_reference(case, (out, final), _cute_reference(case))


def test_runner_batched_shorter_than_one_chunk():
    """Batched ``2 x 50``: fewer than 128 tokens per row (CAKE-1063).  The
    FlashInfer host zero-pads x/B/C to one chunk and stages the output; the
    CuTe oracle runs the same tokens as one packed stream of two sequences."""
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=False, seed=31, lengths=(50, 50))
    x, dt, A, B, C = case["tensors"]
    batch, seqlen = x.shape[:2]
    assert seqlen < CHUNK
    assert _supports(case, out=case["out"])
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(*case["tensors"], out=case["out"], **case["run"])
    torch.cuda.synchronize()
    assert out.untyped_storage().data_ptr() == case["out"].untyped_storage().data_ptr()
    assert tuple(out.shape) == tuple(x.shape)
    assert tuple(final.shape) == (batch, NHEADS, HEADDIM, DSTATE)
    assert torch.isfinite(out.float()).all() and torch.isfinite(final.float()).all()

    def packed(tensor):
        return tensor.reshape(1, batch * seqlen, *tensor.shape[2:]).contiguous()

    seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(case["lengths"], device)
    packed_case = {
        **case,
        "tensors": (packed(x), packed(dt), A, packed(B), packed(C)),
        "run": {
            **case["run"],
            "z": packed(case["run"]["z"]),
            "seq_idx": seq_idx,
            "chunk_indices": chunk_indices,
            "chunk_offsets": chunk_offsets,
        },
        "ctor": {**case["ctor"], "has_varlen": True},
    }
    cute_out, cute_final = _cute_reference(packed_case)
    cute = (cute_out.reshape(x.shape), cute_final)
    _assert_matches_reference(case, (out, final), cute)


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_fp32_state(varlen):
    """``--mamba-ssm-dtype float32`` (the engine default): FP32 initial and
    final states; the CuTe oracle runs the same BF16-representable states."""
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(
        device, varlen=varlen, seed=30 + int(varlen), state_dtype=torch.float32
    )
    assert _supports(case, out=case["out"])
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(*case["tensors"], out=case["out"], **case["run"])
    torch.cuda.synchronize()
    assert final.dtype == torch.float32
    assert torch.isfinite(out.float()).all() and torch.isfinite(final).all()
    _assert_matches_reference(case, (out, final), _cute_reference(case))


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
def test_runner_writes_selective_checkpoint_states(varlen):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(device, varlen=varlen, seed=7 + int(varlen))
    lengths = case["lengths"]
    if varlen:
        # Packed [0, 96) + [96, 256). Sequence 0 checkpoints at its end (96,
        # a logical boundary because sequence 1 starts there); sequence 1
        # after 128 of its tokens (absolute 224), a boundary inside physical
        # chunk 1 that the caller exposes through the chunk metadata.
        boundaries = [96, 224]
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(
            lengths, device, extra_boundaries=(224,)
        )
        assert torch.equal(
            chunk_indices.cpu(), torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        )
        assert torch.equal(
            chunk_offsets.cpu(), torch.tensor([0, 96, 0, 96], dtype=torch.int32)
        )
        metadata = dict(chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
        prefix = [(0, 0, 96), (0, 96, 128)]  # (batch index, start, length)
    else:
        # Sequence-relative: sequence 0 after one chunk, sequence 1 at its end.
        boundaries = [128, 256]
        metadata = {}
        prefix = [(0, 0, 128), (1, 0, 256)]
    slots = [2, 0]
    token_indices = torch.tensor(boundaries, device=device, dtype=torch.int32)
    slot_indices = torch.tensor(slots, device=device, dtype=torch.int32)
    checkpoint_states = torch.full(
        (3, NHEADS, HEADDIM, DSTATE), float("nan"), device=device
    ).bfloat16()
    assert _supports(
        case,
        checkpoint_token_indices=token_indices,
        checkpoint_state_slots=slot_indices,
        checkpoint_states=checkpoint_states,
        **metadata,
    )
    runner = cake_ssd_combined(**case["ctor"])
    out, final = runner.run(
        *case["tensors"],
        checkpoint_token_indices=token_indices,
        checkpoint_state_slots=slot_indices,
        checkpoint_states=checkpoint_states,
        **{**case["run"], **metadata},
    )
    torch.cuda.synchronize()
    # The exposed logical boundary does not change the result.
    expected = _cute_reference(case)
    absolute = [
        (0 if varlen else b * case["tensors"][0].shape[1]) + start + length
        for (b, start, length) in prefix
    ]
    reference = _fp64_reference(case, checkpoints=set(absolute))
    _assert_matches_reference(case, (out, final), expected, reference)
    # Written slots: the fp64 state at the boundary (a checkpoint at the
    # sequence end is its final state), with CuTe's final state of the same
    # prefix as the comparison arm; the unused slot stays untouched.
    for seq, (slot, (b, start, length), boundary) in enumerate(
        zip(slots, prefix, absolute)
    ):
        assert torch.isfinite(checkpoint_states[slot].float()).all()
        seq_end = sum(lengths[: seq + 1]) if varlen else lengths[seq]
        if start + length == seq_end:
            cute_state = expected[1][seq]
        else:
            cute_state = _cute_prefix_final_state(
                case, batch_index=b, start=start, length=length, sequence=seq
            )
        _assert_accuracy(
            f"checkpoint[{slot}]",
            checkpoint_states[slot],
            cute_state,
            reference[2][boundary],
        )
    assert torch.isnan(checkpoint_states[UNTOUCHED_SLOT]).all()

    # A boundary that is not a logical chunk end (72) and a negative entry
    # capture nothing: the NaN fill survives in every slot.
    unaligned = torch.full_like(checkpoint_states, float("nan"))
    runner.run(
        *case["tensors"],
        checkpoint_token_indices=torch.tensor(
            [72, -1], device=device, dtype=torch.int32
        ),
        checkpoint_state_slots=slot_indices,
        checkpoint_states=unaligned,
        **{**case["run"], **metadata},
    )
    torch.cuda.synchronize()
    assert torch.isnan(unaligned).all()


def _cu_seqlens(lengths, device):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return torch.tensor(cu, dtype=torch.int32, device=device)


def _stock_triton_reference(case, cu_seqlens, *, chunk_size, state_dtype):
    """The engine's own call: stock Triton ``mamba_chunk_scan_combined`` at the
    model chunk size with ``cu_seqlens`` and, when a prefix is given, the
    engine's chunk-``chunk_size`` metadata (``Mamba2Metadata``). Returns
    ``(out, varlen_states)``."""
    from sglang.kernels.ops.mamba.triton_ops.ssd_combined import (
        mamba_chunk_scan_combined,
    )
    from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata

    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    chunk_indices = chunk_offsets = None
    if run["initial_states"] is not None:
        chunk_indices, chunk_offsets = (
            Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
                cu_seqlens, chunk_size, int(x.shape[1])
            )
        )
    out = torch.empty_like(case["out"])
    varlen_states = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=chunk_size,
        D=run["D"],
        z=run["z"],
        dt_bias=run["dt_bias"],
        initial_states=run["initial_states"],
        seq_idx=run["seq_idx"],
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        dt_softplus=run["dt_softplus"],
        dt_limit=run["dt_limit"],
        out=out,
        return_varlen_states=True,
        return_final_states=False,
        state_dtype=state_dtype,
    )
    return out, varlen_states


def _assert_within_fixed_tolerance(name, actual, reference, comparison=None):
    """The fixed tolerance against the fp64 recurrence -- every value finite,
    at most ``_MAX_OUTSIDE_FRACTION`` of the entries outside ``atol = rtol =
    1e-2`` -- for the Cake result and, when given, for the stock Triton(256)
    comparison arm.  ``_assert_accuracy``'s parity rule (no more outliers than
    the comparison arm up to Poisson slack) is a CuTe rule: CuTe runs the same
    bf16 algorithm, stock Triton keeps the per-token delta in fp32 where the
    Cake kernels store it in fp16 (CAKE-942: 0.66 % vs 0.62 % outliers on the
    realistic distribution), so Triton is the more accurate arm by
    construction and a Poisson slack does not cover that systematic gap.  Both
    counts are reported; the tolerance itself is the fixed one."""
    assert tuple(actual.shape) == tuple(reference.shape), (name, tuple(actual.shape))
    assert torch.isfinite(actual.to(torch.float64)).all(), f"{name}: Cake output is not finite"
    budget = _MAX_OUTSIDE_FRACTION * reference.numel()
    outside = int(_outside_tolerance(actual, reference).sum())
    detail = ""
    if comparison is not None:
        assert tuple(comparison.shape) == tuple(reference.shape), name
        comparison_outside = int(_outside_tolerance(comparison, reference).sum())
        assert comparison_outside <= budget, (
            f"{name}: stock Triton(256) has {comparison_outside} of {reference.numel()} "
            f"entries outside atol=rtol={_ATOL} of the fp64 recurrence (limit {budget:.0f})"
        )
        detail = f"; stock Triton(256): {comparison_outside}"
    assert outside <= budget, (
        f"{name}: {outside} of {reference.numel()} Cake entries outside "
        f"atol=rtol={_ATOL} of the fp64 recurrence (limit {budget:.0f}{detail})"
    )


@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["bf16-pool", "fp32-pool"]
)
@pytest.mark.parametrize("initial", [True, False], ids=["prefix", "no-prefix"])
@pytest.mark.parametrize(
    "lengths",
    [(96, 160), (1000,), (64, 60), (300, 700), (128, 896), (2048, 2048)],
    ids=["96+160", "1000", "64+60", "300+700", "128+896", "2048x2"],
)
def test_route_call_cu_seqlens_at_engine_chunk_size_matches_triple_and_stock(
    lengths, initial, state_dtype
):
    """The route's call (CAKE-934 item 2 option B): ``cu_seqlens`` + the
    engine's chunk size (256), no ``seq_idx`` / chunk-128 triple / ``num_seqs``.
    Admitted by the adapter at chunk 256 (the triple is not); bitwise equal
    to the chunk-128 triple call of the same inputs (the preprocess derives
    the same segment tables); within the fixed BF16 tolerance of the fp64
    recurrence, with stock Triton(256) -- the engine's own kernel -- held to
    the same tolerance on the same inputs, for a BF16 and an FP32 state pool,
    with and without a prefix state."""
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _case(
        device,
        varlen=True,
        seed=40 + sum(lengths) % 97 + int(initial),
        lengths=lengths,
        initial_states=initial,
        state_dtype=state_dtype,
    )
    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    cu = _cu_seqlens(lengths, device)
    common = dict(
        D=run["D"],
        z=run["z"],
        dt_bias=run["dt_bias"],
        initial_states=run["initial_states"],
        state_dtype=state_dtype,
    )
    # Admission: the cu_seqlens form at the engine's chunk size, no triple.
    assert cake_mamba.supports_ssd_combined(
        x, dt, A, B, C, cu_seqlens=cu, chunk_size=ENGINE_CHUNK, out=case["out"], **common
    )
    assert not cake_mamba.supports_ssd_combined(
        x,
        dt,
        A,
        B,
        C,
        seq_idx=run["seq_idx"],
        chunk_indices=run["chunk_indices"],
        chunk_offsets=run["chunk_offsets"],
        num_seqs=run["num_seqs"],
        chunk_size=ENGINE_CHUNK,
        out=case["out"],
        **common,
    )
    scan = dict(dt_softplus=True, dt_limit=run["dt_limit"], return_final_states=True)
    out_cu, final_cu = cake_ssd_combined_fwd(
        x, dt, A, B, C, cu_seqlens=cu, chunk_size=ENGINE_CHUNK, out=case["out"], **common, **scan
    )
    assert out_cu.untyped_storage().data_ptr() == case["out"].untyped_storage().data_ptr()
    triple_out = torch.empty_like(case["out"])
    out_tr, final_tr = cake_ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        seq_idx=run["seq_idx"],
        chunk_indices=run["chunk_indices"],
        chunk_offsets=run["chunk_offsets"],
        num_seqs=run["num_seqs"],
        out=triple_out,
        **common,
        **scan,
    )
    torch.cuda.synchronize()
    assert final_cu.dtype == state_dtype and tuple(final_cu.shape) == (
        len(lengths),
        NHEADS,
        HEADDIM,
        DSTATE,
    )
    assert torch.equal(out_cu, out_tr), "cu_seqlens form differs from the chunk-128 triple"
    assert torch.equal(final_cu, final_tr)
    stock_out, stock_final = _stock_triton_reference(
        case, cu, chunk_size=ENGINE_CHUNK, state_dtype=state_dtype
    )
    torch.cuda.synchronize()
    reference = _fp64_reference(case)
    _assert_within_fixed_tolerance("out", out_cu, reference[0], stock_out)
    _assert_within_fixed_tolerance(
        "final_states", final_cu, reference[1], stock_final.to(state_dtype)
    )


def test_route_call_cu_seqlens_engine_chunk_size_checkpoint_unaligned_start():
    """A radix-cache track row of a sequence that starts at a packed offset
    off the 128 grid (packed [0, 300) + [300, 1000), track position 256
    tokens into sequence 1 = absolute 556): with ``cu_seqlens`` the Cake
    preprocess makes the checkpoint token a segment boundary itself, so the
    route passes the checkpoint pair without any chunk metadata.  The slot
    holds the fp64 state after 556 tokens within the fixed tolerance (stock
    Triton's chunk-256 final state of the same prefix held to the same
    tolerance); an unused slot stays untouched."""
    _skip_unless_supported()
    device = torch.device("cuda")
    lengths = (300, 700)
    case = _case(device, varlen=True, seed=91, lengths=lengths)
    x, dt, A, B, C = case["tensors"]
    run = case["run"]
    cu = _cu_seqlens(lengths, device)
    boundary = 300 + 256
    token_indices = torch.tensor([-1, boundary], device=device, dtype=torch.int32)
    slot_indices = torch.tensor([-1, 2], device=device, dtype=torch.int32)
    checkpoint_states = torch.full(
        (3, NHEADS, HEADDIM, DSTATE), float("nan"), device=device
    ).bfloat16()
    common = dict(
        D=run["D"],
        z=run["z"],
        dt_bias=run["dt_bias"],
        initial_states=run["initial_states"],
        state_dtype=torch.bfloat16,
        checkpoint_token_indices=token_indices,
        checkpoint_state_slots=slot_indices,
        checkpoint_states=checkpoint_states,
    )
    assert cake_mamba.supports_ssd_combined(
        x, dt, A, B, C, cu_seqlens=cu, chunk_size=ENGINE_CHUNK, out=case["out"], **common
    )
    out, final = cake_ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens=cu,
        chunk_size=ENGINE_CHUNK,
        out=case["out"],
        dt_softplus=True,
        dt_limit=run["dt_limit"],
        return_final_states=True,
        **common,
    )
    torch.cuda.synchronize()
    reference = _fp64_reference(case, checkpoints={boundary})
    stock_out, stock_final = _stock_triton_reference(
        case, cu, chunk_size=ENGINE_CHUNK, state_dtype=torch.bfloat16
    )
    _assert_within_fixed_tolerance("out", out, reference[0], stock_out)
    _assert_within_fixed_tolerance("final_states", final, reference[1], stock_final)
    # The checkpoint: stock Triton's final state of sequence 1's first 256
    # tokens (a batched chunk-256 call on that prefix) is the comparison arm.
    prefix_case = {
        **case,
        "tensors": tuple(
            t[:, 300:boundary].contiguous() if t.ndim >= 2 and t.shape[1] == 1000 else t
            for t in case["tensors"]
        ),
        "lengths": (256,),
        "run": {
            **run,
            "z": run["z"][:, 300:boundary].contiguous(),
            "initial_states": run["initial_states"][1:2].contiguous(),
            # stock Triton requires seq_idx whenever cu_seqlens rides with
            # initial_states (ssd_state_passing: "continuous batching"); one
            # sequence of 256 tokens.
            "seq_idx": torch.zeros((1, 256), dtype=torch.int32, device=device),
            "chunk_indices": None,
            "chunk_offsets": None,
        },
        "out": torch.empty(1, 256, NHEADS, HEADDIM, device=device, dtype=torch.bfloat16),
    }
    _, prefix_final = _stock_triton_reference(
        prefix_case, _cu_seqlens((256,), device), chunk_size=ENGINE_CHUNK, state_dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    assert torch.isfinite(checkpoint_states[2].float()).all()
    _assert_within_fixed_tolerance(
        "checkpoint[2]", checkpoint_states[2], reference[2][boundary], prefix_final[0]
    )
    assert torch.isnan(checkpoint_states[UNTOUCHED_SLOT]).all()
    assert torch.isnan(checkpoint_states[0]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
