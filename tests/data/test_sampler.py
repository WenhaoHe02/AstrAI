from astrai.dataset import RDSampler


def test_random_sampler_consistency(random_dataset):
    """Test RandomSampler produces consistent results with same seed"""
    dataset = random_dataset

    # Create two samplers with same seed
    sampler1 = RDSampler(dataset, seed=42)
    sampler2 = RDSampler(dataset, seed=42)

    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))

    assert indices1 == indices2


def test_random_sampler_different_seeds(random_dataset):
    """Test RandomSampler produces different results with different seeds"""
    dataset = random_dataset

    # Create two samplers with different seeds
    sampler1 = RDSampler(dataset, seed=42)
    sampler2 = RDSampler(dataset, seed=123)

    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))

    # Very high probability they should be different
    assert indices1 != indices2


def test_sampler_across_epochs(random_dataset):
    """Test sampler behavior across multiple epochs"""
    dataset = random_dataset
    n = len(dataset)

    sampler = RDSampler(dataset, seed=42)

    # Get indices for first epoch
    epoch1_indices = list(iter(sampler))
    assert len(epoch1_indices) == n

    # Get indices for second epoch
    epoch2_indices = list(iter(sampler))
    assert len(epoch2_indices) == n

    # Check that epochs have different order (should be random)
    assert epoch1_indices != epoch2_indices

    # Check that all indices are present in each epoch
    assert set(epoch1_indices) == set(range(n))
    assert set(epoch2_indices) == set(range(n))


def test_resume_at_epoch_boundary_advances_epoch(random_dataset):
    n = len(random_dataset)
    resumed = RDSampler(random_dataset, start_epoch=0, start_iter=n, seed=42)
    epoch_two = RDSampler(random_dataset, start_epoch=1, seed=42)

    assert resumed.epoch == 1
    assert list(resumed) == list(epoch_two)


def test_resume_mid_later_epoch_preserves_offset(random_dataset):
    n = len(random_dataset)
    full_epoch = list(RDSampler(random_dataset, start_epoch=3, seed=42))
    resumed = RDSampler(
        random_dataset,
        start_epoch=3,
        start_iter=3 * n + 2,
        seed=42,
    )

    assert resumed.epoch == 3
    assert list(resumed) == full_epoch[2:]
