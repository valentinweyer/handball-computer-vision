import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.teammate_reid import finetune


def test_untrained_adapter_is_identity():
    x = torch.nn.functional.normalize(torch.randn(5, 512), dim=-1)
    assert torch.allclose(finetune.make_adapter()(x), x, atol=1e-6)


def test_same_fragment_is_never_a_positive():
    y = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    ident = torch.tensor([0, 0, 1, 1])
    team = torch.tensor([0, 0, 0, 0])
    frame = torch.tensor([0, 500, 0, 500])
    one_fragment = finetune.contrastive_loss(y, ident, torch.tensor([7, 7, 8, 8]), frame, team, 0.1, 0.3)
    assert float(one_fragment) == 0.0          # no anchor has a positive
    two_fragments = finetune.contrastive_loss(y, ident, torch.tensor([7, 9, 8, 10]), frame, team, 0.1, 0.3)
    assert float(two_fragments) > 0.0


def test_positive_needs_min_separation():
    y = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    args = torch.tensor([0, 0]), torch.tensor([1, 2])
    close = finetune.contrastive_loss(y, *args, torch.tensor([0, 10]), torch.tensor([0, 0]), 0.1, 0.3)
    assert float(close) == 0.0


def test_time_blocks_drop_straddlers():
    s = finetune.SPLIT_FRAME
    assert finetune.block({"start": 0, "end": s - 1}) == "train"
    assert finetune.block({"start": s, "end": s + 100}) == "heldout"
    assert finetune.block({"start": s - 10, "end": s + 10}) is None


def test_seen_teams_follow_the_main_window_cluster():
    main = {"match": "Gummersbach-Loewen", "to_main": {0: 0, 1: 1}}
    extra = {"match": "Gummersbach-Loewen", "to_main": {0: 1, 1: 0}}   # clusters swapped in this window
    assert finetune.seen_team(main, 0) and not finetune.seen_team(main, 1)
    assert finetune.seen_team(extra, 1) and not finetune.seen_team(extra, 0)
    assert finetune.team_key(extra, 1) == "Gummersbach-Loewen|0"
    assert not finetune.seen_team({"match": "Eisenach-Hamburg", "to_main": {0: 0, 1: 1}}, 0)


def test_views_keep_shape():
    img = np.random.default_rng(0).integers(0, 255, (120, 50, 3), dtype=np.uint8)
    views = finetune._views(img, np.random.default_rng(1))
    assert len(views) == 3 and all(v.shape == img.shape for v in views)
    assert np.array_equal(views[1], img[:, ::-1])


def test_split_has_no_shared_match_between_groups():
    groups = {}
    for g, match, *_ in finetune.SOURCES:
        groups.setdefault(match, set()).add(g)
    assert all(len(v) == 1 for v in groups.values())
    train = {m for m, g in groups.items() if "A" in g}
    assert not train & {m for m, g in groups.items() if g & {"C", "D"}}
