from btdl.data.sampler import EpochSampler


def test_shuffle_false_is_ordered():
    sampler = EpochSampler(10, shuffle=False, seed=42, epoch_aware=False)
    assert list(sampler) == list(range(10))


def test_shuffle_false_ordered_regardless_of_epoch():
    sampler = EpochSampler(10, shuffle=False, seed=42, epoch_aware=False)
    sampler.set_epoch(5)
    assert list(sampler) == list(range(10))


def test_shuffle_true_is_a_permutation():
    sampler = EpochSampler(20, shuffle=True, seed=42, epoch_aware=False)
    order = list(sampler)
    assert sorted(order) == list(range(20))
    assert order != list(range(20))  # astronomically unlikely to coincide


def test_shuffle_true_reproducible_for_same_seed_and_epoch():
    a = list(EpochSampler(20, shuffle=True, seed=42, epoch_aware=False))
    b = list(EpochSampler(20, shuffle=True, seed=42, epoch_aware=False))
    assert a == b


def test_shuffle_true_differs_across_epochs():
    sampler = EpochSampler(20, shuffle=True, seed=42, epoch_aware=False)
    sampler.set_epoch(0)
    order0 = list(sampler)
    sampler.set_epoch(1)
    order1 = list(sampler)
    assert order0 != order1


def test_shuffle_true_differs_across_seeds():
    a = list(EpochSampler(20, shuffle=True, seed=42, epoch_aware=False))
    b = list(EpochSampler(20, shuffle=True, seed=43, epoch_aware=False))
    assert a != b


def test_epoch_aware_yields_index_epoch_tuples():
    sampler = EpochSampler(5, shuffle=False, seed=42, epoch_aware=True)
    sampler.set_epoch(3)
    items = list(sampler)
    assert items == [(i, 3) for i in range(5)]


def test_not_epoch_aware_yields_plain_ints():
    sampler = EpochSampler(5, shuffle=False, seed=42, epoch_aware=False)
    items = list(sampler)
    assert all(isinstance(i, int) for i in items)


def test_every_index_exactly_once_per_epoch():
    sampler = EpochSampler(37, shuffle=True, seed=7, epoch_aware=False)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        order = list(sampler)
        assert sorted(order) == list(range(37))


def test_len():
    sampler = EpochSampler(15, shuffle=True, seed=1, epoch_aware=False)
    assert len(sampler) == 15
