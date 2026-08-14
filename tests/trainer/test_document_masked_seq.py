import torch

from astrai.dataset.dataset import SEQDataset
from astrai.trainer.strategy import (
    SEQStrategy,
    make_doc_boundary_mask,
    make_document_cu_seqlens,
)


class _CaptureModel(torch.nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        batch, length = kwargs["input_ids"].shape
        return {"logits": torch.zeros(batch, length, self.vocab_size)}


class _DocumentStore:
    keys = ["sequence", "position_ids"]
    token_count = 6

    def __len__(self):
        return 1

    def sample_window(self, index):
        assert index == 0
        return 0, 5

    def fetch(self, begin, end, key):
        values = {
            "sequence": torch.tensor([10, 11, 2, 12, 13, 2]),
            "position_ids": torch.tensor([0, 1, 2, 0, 1, 2]),
        }
        return values[key][begin:end]


def test_document_boundary_mask_blocks_cross_document_attention():
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    mask = make_doc_boundary_mask(position_ids)[0, 0]

    assert mask[2, 0]
    assert not mask[3, 2]
    assert mask[4, 3]
    assert not mask[2, 3]


def test_seq_dataset_masks_first_target_of_next_document():
    item = SEQDataset(_DocumentStore())[0]

    assert item["position_ids"].tolist() == [0, 1, 2, 0, 1]
    assert item["loss_mask"].tolist() == [True, True, False, True, True]


def test_document_cu_seqlens_include_row_and_document_boundaries():
    position_ids = torch.tensor([[4, 5, 0, 1], [0, 1, 2, 0]])

    cu_seqlens, max_seqlen = make_document_cu_seqlens(position_ids)

    assert cu_seqlens.tolist() == [0, 2, 4, 7, 8]
    assert max_seqlen == 3


def test_seq_strategy_masks_cross_document_target():
    model = _CaptureModel()
    strategy = SEQStrategy(model, "cpu", loss_backend="torch")
    batch = {
        "input_ids": torch.tensor([[4, 5, 2, 7]]),
        "target_ids": torch.tensor([[5, 2, 7, 8]]),
        "position_ids": torch.tensor([[0, 1, 2, 0]]),
        "loss_mask": torch.tensor([[True, True, False, True]]),
    }

    loss = strategy.compute_loss(batch)

    assert torch.isfinite(loss)
    assert model.last_kwargs["document_cu_seqlens"].tolist() == [0, 3, 4]
    assert model.last_kwargs["document_max_seqlen"] == 3
