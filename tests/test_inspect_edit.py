import torch

from wsp.inspect_edit import summarize_delta


def test_rank_one_write_of_a_token_direction_is_identified():
    embedding = torch.eye(4)
    delta = torch.outer(0.3 * embedding[2], torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, 1.0]))
    summary = summarize_delta(delta, embedding, str, top=1)
    assert summary["rank1_energy_percent"] == 100.0
    assert summary["output_direction_tokens"] == [{"id": 2, "token": "2", "abs_cosine": 1.0}]


def test_two_independent_directions_are_not_reported_as_rank_one():
    embedding = torch.eye(4)
    delta = torch.outer(embedding[0], torch.ones(5)) + torch.outer(
        embedding[1], torch.tensor([1.0, -1.0, 1.0, -1.0, 0.0])
    )
    summary = summarize_delta(delta, embedding, str)
    assert summary["rank1_energy_percent"] < 60
    assert summary["rank2_energy_percent"] == 100.0
