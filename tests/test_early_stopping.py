from neurofy.training.early_stopping import EarlyStopping


def test_patience_only_counts_from_start_epoch() -> None:
    stopper = EarlyStopping(patience=2, start_epoch=5)

    stopper.update(0.9, epoch=1)
    for epoch in range(2, 5):
        stopper.update(0.1, epoch=epoch)
    assert stopper.epochs_without_improvement == 0

    stopper.update(0.1, epoch=5)
    assert not stopper.should_stop
    stopper.update(0.1, epoch=6)
    assert stopper.should_stop


def test_improvement_resets_patience_and_tracks_best_epoch() -> None:
    stopper = EarlyStopping(patience=2, start_epoch=1)

    assert stopper.update(0.5, epoch=1)
    assert not stopper.update(0.4, epoch=2)
    assert stopper.update(0.8, epoch=3)

    assert stopper.epochs_without_improvement == 0
    assert stopper.best_epoch == 3
    assert stopper.best_score == 0.8
