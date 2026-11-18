"""Distance meaning, pair order and degenerate-output safety, without inference."""

import numpy as np
import pytest

from src.evaluation.geometry import decision_distances, distance_agreement, encoder_distances


def test_pairs_have_same_order_and_known_distances():
    x = np.array([[1, 0], [0, 1], [1, 1]])
    np.testing.assert_allclose(encoder_distances(x), [1, 1 - 2**-.5, 1 - 2**-.5])
    np.testing.assert_allclose(decision_distances([0, .25, 1]), [.25, 1, .75])
    np.testing.assert_allclose(decision_distances([[1, 0], [.75, .25], [0, 1]]),
                               [.25, 1, .75])


def test_all_nine_probabilities_matter_even_when_means_agree():
    p = np.zeros((2, 9))
    p[0, 4] = 1
    p[1, [0, 8]] = .5
    assert np.all(p @ np.arange(-4, 5) == 0)
    assert decision_distances(p).item() == 1


def test_constant_output_is_undefined_and_pairs_are_not_independent():
    d = encoder_distances(np.eye(3))
    result = distance_agreement(d, decision_distances([.2, .2, .2]))
    assert result['spearman'] is None
    assert result['independent_pairs'] is False
    result = distance_agreement([.1, .2, .3], [.3, .2, .1])
    assert result['spearman'] == -1
    assert 'pvalue' not in result


@pytest.mark.parametrize('bad', [[[0, 0], [1, 0]], [[1, 0], [float('nan'), 1]]])
def test_invalid_embeddings_fail_without_silent_pair_dropping(bad):
    with pytest.raises(ValueError):
        encoder_distances(bad)


@pytest.mark.parametrize('bad', [[-0.1, .5], [.5, float('nan')], [[.5, .4], [.3, .7]]])
def test_invalid_probabilities_fail(bad):
    with pytest.raises(ValueError):
        decision_distances(bad)


def test_500_texts_have_124750_pairs_and_misaligned_lengths_fail():
    assert len(encoder_distances(np.ones((500, 4)))) == 124750
    with pytest.raises(ValueError):
        distance_agreement([0, 1], [0, 1, 2])
