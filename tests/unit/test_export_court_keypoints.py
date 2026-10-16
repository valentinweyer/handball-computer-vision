"""ema_checkpoint must write the requested epoch's EMA weights where rfdetr reads them.

rfdetr loads a .pth's `model` entry. The first version copied that entry from
the run's checkpoint_best_ema.pth template, so an "epoch 139" export silently
held epoch 166's weights.
"""
import torch

from scripts.export_court_keypoints import ema_checkpoint


def test_ema_weights_land_in_model_and_state_dict(tmp_path):
    ema_weight, template_weight = torch.ones(2), torch.zeros(2)
    torch.save({
        "epoch": 139, "global_step": 5880, "state_dict": {"model.w": torch.full((2,), 7.0)},
        "callbacks": {"RFDETREMACallback": {"average_model_state_dict": {
            "n_averaged": torch.tensor(5), "module.model.w": ema_weight}}},
    }, tmp_path / "checkpoint_139.ckpt")
    torch.save({
        "model": {"w": template_weight}, "state_dict": {"model.w": template_weight},
        "args": {"lr": 1}, "model_name": "RFDETRKeypointPreview", "model_config": {},
        "rfdetr_version": "1.9.4", "epoch": 166, "callbacks": {"stale": True},
    }, tmp_path / "checkpoint_best_ema.pth")

    out = torch.load(ema_checkpoint(tmp_path / "checkpoint_139.ckpt", tmp_path / "out.pth"),
                     weights_only=False)

    assert torch.equal(out["model"]["w"], ema_weight)
    assert torch.equal(out["state_dict"]["model.w"], ema_weight)
    assert out["epoch"] == 139 and out["model_name"] == "RFDETRKeypointPreview"
    assert "callbacks" not in out
