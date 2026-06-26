import torch

from sl_atstgcn.schedule import build_scheduler


def make_optimizer():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([param], lr=0.01)


def test_triangular2_scheduler_decays_cycle_amplitude():
    optimizer = make_optimizer()
    scheduler = build_scheduler(
        optimizer,
        {"scheduler": "cyclic", "base_lr": 0.0, "max_lr": 1.0, "clr_step_size": 2, "clr_mode": "triangular2"},
        steps_per_epoch=1,
    )

    values = [scheduler.batch_step() for _ in range(10)]

    assert max(values[:4]) == 1.0
    assert max(values[4:8]) == 0.5


def test_cosine_scheduler_warms_up_then_decays():
    optimizer = make_optimizer()
    scheduler = build_scheduler(
        optimizer,
        {
            "scheduler": "cosine",
            "base_lr": 0.01,
            "max_lr": 0.3,
            "min_lr": 0.001,
            "epochs": 4,
            "warmup_epochs": 1,
        },
        steps_per_epoch=2,
    )

    values = [scheduler.batch_step() for _ in range(8)]

    assert values[0] == 0.01
    assert values[2] > values[1]
    assert values[-1] < values[2]


def test_scheduler_preserves_optimizer_group_lr_scales():
    param_a = torch.nn.Parameter(torch.tensor([1.0]))
    param_b = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [
            {"params": [param_a], "lr_scale": 0.5},
            {"params": [param_b], "lr_scale": 2.0},
        ],
        lr=0.1,
    )
    scheduler = build_scheduler(
        optimizer,
        {"scheduler": "cosine", "base_lr": 0.1, "max_lr": 0.1, "min_lr": 0.1, "epochs": 1, "warmup_epochs": 0},
        steps_per_epoch=1,
    )

    lr = scheduler.batch_step()

    assert lr == 0.1
    assert optimizer.param_groups[0]["lr"] == 0.05
    assert optimizer.param_groups[1]["lr"] == 0.2
