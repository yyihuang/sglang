from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.dsv4 import cake_backend


def test_rebased_indices_keep_swa_and_compressed_pools_separate():
    swa_source = torch.arange(256, dtype=torch.int32).view(2, 128)
    swa_source[1, 127] = -1
    compressed_source = torch.tensor(
        [[0, 1, -1, -1], [2, 3, 4, -1]], dtype=torch.int32
    )
    indices = cake_backend._rebased_indices(swa_source, compressed_source)

    assert indices.shape == (2, 132)
    assert torch.equal(indices[0, :128], torch.arange(128, dtype=torch.int32))
    assert torch.equal(indices[1, :127], torch.arange(128, 255, dtype=torch.int32))
    assert indices[1, 127].item() == -1
    assert torch.equal(indices[0, 128:], torch.tensor([0, 1, -1, -1]))
    assert torch.equal(indices[1, 128:], torch.tensor([4, 5, 6, -1]))


def test_decode_adapter_forces_cake_without_fallback(monkeypatch):
    dequant_calls = []
    flashinfer_calls = []

    def fake_dequant(cache, indices, page_size, out):
        dequant_calls.append((cache, indices.clone(), page_size))
        out.zero_()
        return out

    def fake_flashinfer(*args, **kwargs):
        flashinfer_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return kwargs["out"]

    monkeypatch.setattr(cake_backend, "dequantize_k_cache_paged", fake_dequant)
    monkeypatch.setattr(cake_backend, "_flashinfer_dsv4", lambda: fake_flashinfer)

    adapter = cake_backend.CakeDsv4DecodeWorkspace(torch.device("cpu"))
    q = torch.zeros((2, 1, 8, 512), dtype=torch.bfloat16)
    packed = torch.zeros((4, 4096), dtype=torch.uint8)
    swa_indices = torch.arange(256, dtype=torch.int32).view(2, 1, 128)
    compressed_indices = torch.tensor(
        [[[0, 1, -1, -1]], [[2, 3, -1, -1]]], dtype=torch.int32
    )

    result = adapter.run(
        q=q,
        packed_swa_cache=packed,
        swa_indices=swa_indices,
        swa_active_lens=torch.tensor([128, 127], dtype=torch.int32),
        swa_page_size=128,
        packed_compressed_cache=packed,
        compressed_indices=compressed_indices,
        compressed_active_lens=torch.tensor([2, 1], dtype=torch.int32),
        compressed_page_size=2,
        seq_lens=torch.tensor([256, 255], dtype=torch.int32),
        max_seq_len=256,
        softmax_scale=0.125,
        sinks=torch.zeros(8, dtype=torch.float32),
    )

    assert result.shape == q.shape
    assert len(dequant_calls) == 2
    assert len(flashinfer_calls) == 1
    call = flashinfer_calls[0]
    assert call.kwargs["backend"] == "cake"
    assert call.kwargs["enable_pdl"] is False
    assert call.kwargs["kv_layout"] == "HND"
    assert call.kwargs["sparse_topk_lens"].tolist() == [130, 129]
    assert call.kwargs["sparse_indices"].shape == (2, 132)
