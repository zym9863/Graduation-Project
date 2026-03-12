import torch

from src.models.nrms import NRMSModel


def test_nrms_forward_shape() -> None:
    model = NRMSModel(
        num_categories=8,
        num_subcategories=16,
        siglip_dim=8,
        news_value_dim=5,
        category_emb_dim=4,
        news_repr_dim=16,
        num_attention_heads=4,
        dropout=0.1,
        fusion="concat",
    )

    batch_size = 2
    history_len = 3
    candidate_count = 4

    scores = model(
        history_text=torch.randn(batch_size, history_len, 8),
        history_image=torch.randn(batch_size, history_len, 8),
        history_value=torch.randn(batch_size, history_len, 5),
        history_category=torch.randint(0, 8, (batch_size, history_len)),
        history_subcategory=torch.randint(0, 16, (batch_size, history_len)),
        history_mask=torch.tensor([[True, True, False], [True, False, False]]),
        candidate_text=torch.randn(batch_size, candidate_count, 8),
        candidate_image=torch.randn(batch_size, candidate_count, 8),
        candidate_value=torch.randn(batch_size, candidate_count, 5),
        candidate_category=torch.randint(0, 8, (batch_size, candidate_count)),
        candidate_subcategory=torch.randint(0, 16, (batch_size, candidate_count)),
    )

    assert scores.shape == (batch_size, candidate_count)