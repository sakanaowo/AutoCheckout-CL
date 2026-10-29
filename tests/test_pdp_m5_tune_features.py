"""M5: from task 2 on, the last backbone stage and the encoder can keep training, held to the teacher's encoder
output by a feature distillation loss."""

import pytest
import torch

from pdp_helpers import TINY_DETR, make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow
TUNE = "layer4,encoder"  # the tiny model's ResNet-18; ConvNeXt's last stage is stages_3


def m5_args(tmp_path, *extra):
    return pdp_args(tmp_path, "--tune_after_task1", TUNE, "--optim_groups", "prompt", *extra)  # as the configs


def trained(trainer, part):
    return {p.requires_grad for n, p in trainer.model.named_parameters() if part in n.split('.')}


@pytest.mark.parametrize("task_id", [1, 2])
def test_tuned_parts_train_from_task_2_only(tmp_path, monkeypatch, task_id):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(m5_args(tmp_path), task_id=task_id)
    tuned = task_id > 1
    assert trained(trainer, "layer4") == {tuned} and trained(trainer, "encoder") == {tuned}
    assert trained(trainer, "layer3") == {False} and trained(trainer, "decoder") == {False}
    assert trained(trainer, "prompts") == {True}
    groups = trainer.configure_optimizers()["optimizer"].param_groups
    old = {id(p) for p in groups[1]["params"]}
    encoder = [p for n, p in trainer.model.named_parameters() if "encoder" in n.split('.')]
    assert groups[1]["lr"] < groups[0]["lr"] and all((id(p) in old) == tuned for p in encoder)  # lr_old group


def test_feature_kd_is_zero_for_the_teacher_and_pulls_the_encoder_back(tmp_path, monkeypatch):
    import engine
    from models.configuration_deformable_detr import DeformableDetrConfig

    # without dropout, so that the training model starts exactly at the teacher
    monkeypatch.setattr(engine, "DeformableDetrConfig", lambda: DeformableDetrConfig(
        **TINY_DETR, dropout=0.0, activation_dropout=0.0, attention_dropout=0.0))
    trainer = make_trainer(m5_args(tmp_path, "--feature_kd", "1"), task_id=2)
    trainer.set_teacher()
    trainer.train()
    batch = make_batch([[3, 4], [3]])
    _, loss_dict = trainer.common_step(batch, 0)
    assert loss_dict["loss_feature_kd"].item() == pytest.approx(0, abs=1e-10)

    encoder = trainer.model.model.encoder.layers[0].fc1.weight
    with torch.no_grad():
        encoder.add_(0.1 * torch.randn_like(encoder))
    _, loss_dict = trainer.common_step(batch, 0)
    kd = loss_dict["loss_feature_kd"]
    assert kd.item() > 0
    kd.backward()
    assert encoder.grad is not None and encoder.grad.abs().sum() > 0


def test_feature_kd_off_by_default(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path), task_id=2)
    trainer.set_teacher()
    trainer.train()
    _, loss_dict = trainer.common_step(make_batch([[3]]), 0)
    assert "loss_feature_kd" not in loss_dict
