# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse

import pytest

from relax.utils.arguments import get_slime_extra_args_provider


def test_qsa_stage2_loss_coefficient_defaults_off_and_parses_positive():
    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).qsa_indexer_loss_coeff == 0.0
    assert parser.parse_args(["--qsa-indexer-loss-coeff", "0.01"]).qsa_indexer_loss_coeff == 0.01


def test_qsa_stage2_loss_coefficient_reuses_megatron_registered_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsa-indexer-loss-coeff", type=float, default=0.25)
    get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).qsa_indexer_loss_coeff == 0.0
    assert parser.parse_args(["--qsa-indexer-loss-coeff", "0.7"]).qsa_indexer_loss_coeff == 0.7
    with pytest.raises(SystemExit):
        parser.parse_args(["--qsa-indexer-loss-coeff", "nan"])


@pytest.mark.parametrize("value", ["-0.01", "nan", "inf", "-inf"])
def test_qsa_stage2_loss_coefficient_rejects_invalid_values(value: str):
    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([f"--qsa-indexer-loss-coeff={value}"])
