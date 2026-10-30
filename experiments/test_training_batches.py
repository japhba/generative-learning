"""Sparse successful rollouts must not trigger singleton optimizer updates."""

import unittest

from train_planning import completed_minibatches


class AcceptedBatchTests(unittest.TestCase):
    def test_sparse_acceptance_preserves_order_and_full_batches_until_final_flush(self):
        incoming = [[0], [], [1, 2], [3], [], [4, 5, 6], [7, 8]]
        pending, trained, sizes = [], [], []
        for position, rows in enumerate(incoming):
            final = position == len(incoming) - 1
            ready, pending = completed_minibatches(pending, rows, 4, final)
            if ready: sizes.append(len(ready))
            if not final: self.assertEqual(len(ready) % 4, 0)
            trained.extend(ready)
        self.assertEqual(trained, list(range(9)))
        self.assertEqual(pending, [])
        self.assertEqual(sizes, [4, 5])


if __name__ == "__main__": unittest.main()
