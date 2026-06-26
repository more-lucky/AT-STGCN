from sl_atstgcn.extractor import _frame_bound_to_index


def test_wlasl_frame_bounds_are_one_based_and_negative_means_unbounded():
    assert _frame_bound_to_index(1, default=0) == 0
    assert _frame_bound_to_index(7, default=0) == 6
    assert _frame_bound_to_index(None, default=None) is None
    assert _frame_bound_to_index(-1, default=None) is None
