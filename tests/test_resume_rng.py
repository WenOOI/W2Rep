import random

import numpy as np
import torch

from train import advance_iterator_preserving_rng, restore_rng_state, rng_state


class RandomIterator:
    def __init__(self, length: int) -> None:
        self.remaining = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            raise StopIteration
        self.remaining -= 1
        random.random()
        np.random.rand()
        torch.rand(())
        return self.remaining


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_replaying_consumed_batches_preserves_main_process_rng():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    saved = rng_state()

    advance_iterator_preserving_rng(RandomIterator(3), 2)
    actual = _draw()

    restore_rng_state(saved)
    expected = _draw()
    assert actual == expected
