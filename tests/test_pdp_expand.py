"""Expansion of a trained checkpoint to more class slots (docs/project/task6-relearn-plan-2026-10-30.md, Task 4)."""

import math
from types import SimpleNamespace

import pytest
import torch

import runtime
from autocheckout.taskcfg import TaskConfig
from expand import expand_checkpoint
from pdp_helpers import make_batch, make_trainer, pdp_args, task_config_dict, use_tiny_detr
from tools.make_relearn_config import relearn_config

pytestmark = pytest.mark.slow

LEARNED = 5  # classes of tasks 1-2 of config A


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A's task-2 checkpoint (prompts of tasks 1-2 initialised, some prototype memory) and its expansion to B."""
    monkeypatch = pytest.MonkeyPatch()
    use_tiny_detr(monkeypatch)
    tmp = tmp_path_factory.mktemp("expand")
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    cfg_a = TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=1))         # n_classes 7
    cfg_b = relearn_config(cfg_a, [[0, 3], [1]], reserved=1, name="toy+relearn")  # tasks 3, 2, 2, 1, 1 -> 10
    args_a, args_b = pdp_args(tmp / "a", config=cfg_a), pdp_args(tmp / "b", config=cfg_b)
    trainer_a = make_trainer(args_a, task_id=2)
    trainer_a.model.model.prompts.init_task_prompts()
    trainer_a.class_query_cache[0] = [torch.ones(256)]
    trainer_a.class_prototypes[0] = torch.ones(256)
    trainer_a.class_cache_count[0] = 1
    path_a = str(tmp / "a" / "task_final.pth")
    trainer_a.save_task_final(path_a)
    old = torch.load(path_a, map_location="cpu")
    new = expand_checkpoint(old, cfg_a, cfg_b, learned=LEARNED)
    path_b = str(tmp / "b" / "expanded.pth")
    torch.save(new, path_b)
    yield SimpleNamespace(cfg_a=cfg_a, cfg_b=cfg_b, args_a=args_a, args_b=args_b, trainer_a=trainer_a,
                          old=old, new=new, path_a=path_a, path_b=path_b)
    monkeypatch.undo()


def two_pass(model, batch):
    first = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], train=False)
    return model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
                 query=first.last_hidden_state, train=False)


def test_learned_rows_are_copied_and_new_rows_start_untrained(toy):
    old, new = toy.old["model"], toy.new["model"]
    w_old, w_new = old["class_embed.0.weight"], new["class_embed.0.weight"]
    assert w_new.shape[0] == 10 and torch.equal(w_new[:5], w_old[:5]) and torch.equal(w_new[-1], w_old[-1])
    assert 0.01 < w_new[5:9].std() < 0.03  # N(0, init_std = 0.02)
    b_new = new["class_embed.0.bias"]
    assert torch.equal(b_new[:5], old["class_embed.0.bias"][:5]) and torch.equal(b_new[-1], old["class_embed.0.bias"][-1])
    assert torch.allclose(b_new[5:9], torch.full((4,), -math.log(99)))
    heads = sorted({k.split(".")[1] for k in new if k.startswith("class_embed.")})
    assert len(heads) == 2  # one per decoder layer of the tiny model
    assert all(torch.equal(new[f"class_embed.{i}.weight"], w_new) for i in heads)
    assert all(torch.equal(new[f"class_embed.{i}.bias"], b_new) for i in heads)

    p_old, p = old["model.prompts.private_k_0"], new["model.prompts.private_k_0"]
    assert p_old[:5].norm(dim=1).min() > 0.99  # tasks 1-2 prompts are set in the old checkpoint
    assert p.shape[0] == toy.cfg_b.num_slots and torch.equal(p[:5], p_old[:5]) and p[5:].abs().sum() == 0

    assert len(toy.new["class_prototypes"]) == toy.cfg_b.num_slots and toy.new["class_query_cache"][8] == []
    assert toy.new["class_cache_count"][0] == 1 and torch.equal(toy.new["class_prototypes"][0], torch.ones(256))
    assert toy.new["class_cache_count"][8] == 0 and toy.new["class_prototypes"][8].abs().sum() == 0
    assert toy.new["expanded_from"] == {"old_classes": 7, "new_classes": 10, "learned": 5, "seed": 0}

    runtime.validate_runtime(toy.args_b, toy.new["runtime"])
    with pytest.raises(ValueError):
        runtime.validate_runtime(toy.args_a, toy.new["runtime"])
    assert len(toy.old["runtime"]["model_config"]["id2label"]) == 7  # the input is not changed


def test_expanded_model_predicts_the_learned_classes_exactly_as_before(toy):
    trainer_b = make_trainer(toy.args_b, task_id=3)
    trainer_b.resume(toy.path_b)
    batch = make_batch([[0], [1, 2]])
    outputs = []
    for model in (toy.trainer_a.model, trainer_b.model):
        model.eval()
        model.model.prompts.set_task_id(1)
        with torch.no_grad():
            outputs.append(two_pass(model, batch))
    a, b = outputs
    assert torch.equal(b.last_hidden_state, a.last_hidden_state) and torch.equal(b.pred_boxes, a.pred_boxes)
    # same rows, but a linear layer of 10 outputs is not computed bit for bit like one of 7 (BLAS kernels)
    torch.testing.assert_close(b.logits[..., :5], a.logits[..., :5], rtol=0, atol=1e-5)
    torch.testing.assert_close(b.logits[..., -1], a.logits[..., -1], rtol=0, atol=1e-5)


def test_task_after_expansion_trains_on_alias_labels(toy):
    trainer = make_trainer(toy.args_b, task_id=3)
    trainer.resume(toy.path_b)
    trainer.set_teacher()
    trainer.model.model.prompts.init_task_prompts()
    trainer.train()
    loss, _ = trainer.common_step(make_batch([[5, 6], [6]]), 0)
    assert torch.isfinite(loss)
    loss.backward()
    grad = trainer.model.model.prompts.private_p_0.grad
    assert grad[:5].abs().sum() == 0 and grad[5:7].abs().sum() > 0


def test_unexpanded_checkpoint_does_not_load_into_the_larger_model(toy):
    trainer = make_trainer(toy.args_b, task_id=3)
    with pytest.raises(ValueError, match="n_classes"):  # runtime check, before load_state_dict
        trainer.resume(toy.path_a)


@pytest.mark.parametrize("break_it, message", [("rows", "rows"), ("trained_later_slot", "not zero"),
                                               ("changed_learned_class", "differs")])
def test_expansion_refuses_a_checkpoint_or_config_that_does_not_match(toy, break_it, message):
    old, cfg_a, cfg_b = dict(toy.old), toy.cfg_a, toy.cfg_b
    if break_it == "rows":  # the checkpoint was not trained with this config
        cfg_a = TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=2))
    elif break_it == "trained_later_slot":  # the slot after the learned ones has a prompt
        old["model"] = dict(old["model"])
        old["model"]["model.prompts.private_k_0"] = old["model"]["model.prompts.private_k_0"].clone()
        old["model"]["model.prompts.private_k_0"][5] = 1.0
    else:
        data = cfg_b.to_dict()
        data["tasks"][0]["classes"][2]["name"] = "sku_x"
        cfg_b = TaskConfig.from_dict(data)
    with pytest.raises(ValueError, match=message):
        expand_checkpoint(old, cfg_a, cfg_b, learned=LEARNED)
