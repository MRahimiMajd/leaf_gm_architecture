import os

import pandas as pd
import pytest

from leaf_gm_architecture.functions import data_aggregation


def test_data_aggregation():
    """
    We store the aggregated dataset in the repo, so that it doesn't have
    to be recomputed every time. This regression test ensures that the
    `data_aggregation` function didn't change in the meantime.
    """
    # get the path of the directory that contains this script
    test_dir = os.path.dirname(os.path.realpath(__file__))
    path_input_file = os.path.join(test_dir, '..', 'gm_dataset_Knauer_et_al_2022.xlsx')
    test_input_df = pd.read_excel(path_input_file, sheet_name='data')

    path_expected_output_file = os.path.join(test_dir, '..', 'gm_dataset_Knauer_et_al_2022_aggregated.parquet')
    expected_output_df = pd.read_parquet(path_expected_output_file)

    actual_output_df = data_aggregation(test_input_df)
    pd.testing.assert_frame_equal(actual_output_df, expected_output_df)
