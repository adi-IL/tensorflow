# Copyright 2026 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for flags utility library."""

import os
from unittest import mock

from absl import flags
from absl.testing import absltest

from xla.benchmarks.core import flag_utils


class FlagsTest(absltest.TestCase):

  def test_parse_libtpu_init_args_empty(self):
    self.assertEqual(flag_utils.parse_libtpu_init_args(""), {})
    self.assertEqual(flag_utils.parse_libtpu_init_args("   "), {})
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(flag_utils.parse_libtpu_init_args(), {})

  def test_parse_libtpu_init_args_key_value(self):
    args = "--xla_tpu_dvfs_p_state=3 --xla_tpu_scoped_vmem_limit_kib=65536"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(
        parsed,
        {
            "xla_tpu_dvfs_p_state": "3",
            "xla_tpu_scoped_vmem_limit_kib": "65536",
        },
    )

  def test_parse_libtpu_init_args_space_separated(self):
    args = "--xla_tpu_dvfs_p_state 3 --other_flag hello"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(
        parsed,
        {
            "xla_tpu_dvfs_p_state": "3",
            "other_flag": "hello",
        },
    )

  def test_parse_libtpu_init_args_negative_numbers(self):
    args = "--xla_tpu_dvfs_p_state=-1 --other_flag -10.5"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(
        parsed,
        {
            "xla_tpu_dvfs_p_state": "-1",
            "other_flag": "-10.5",
        },
    )

    args_space = "--xla_tpu_dvfs_p_state -1"
    parsed_space = flag_utils.parse_libtpu_init_args(args_space)
    self.assertEqual(parsed_space, {"xla_tpu_dvfs_p_state": "-1"})

  def test_parse_libtpu_init_args_boolean_flag(self):
    args = "--enable_feature --noenable_other"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(parsed["enable_feature"], "true")
    self.assertEqual(parsed["noenable_other"], "true")

  def test_parse_libtpu_init_args_quotes_and_equals(self):
    args = "--str_flag=\"hello world\" --key_eq=a=b=c --single_quoted='foo bar'"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(
        parsed,
        {
            "str_flag": "hello world",
            "key_eq": "a=b=c",
            "single_quoted": "foo bar",
        },
    )

  def test_parse_libtpu_init_args_repeated_flags(self):
    args = "--xla_tpu_dvfs_p_state=1 --xla_tpu_dvfs_p_state=3"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(parsed["xla_tpu_dvfs_p_state"], "3")

  def test_parse_from_env_var(self):
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=5"}
    ):
      parsed = flag_utils.parse_libtpu_init_args()
      self.assertEqual(parsed, {"xla_tpu_dvfs_p_state": "5"})

  def test_get_flag_from_libtpu_init_args(self):
    args = (
        "--xla_tpu_dvfs_p_state=3 --str_val=hello --bool_val=true"
        " --float_val=2.5"
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "xla_tpu_dvfs_p_state", args_str=args, flag_type=int
        ),
        3,
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "--xla_tpu_dvfs_p_state", args_str=args, flag_type=int
        ),
        3,
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "xla-tpu-dvfs-p-state", args_str=args, flag_type=int
        ),
        3,
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "str_val", args_str=args, flag_type=str
        ),
        "hello",
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "bool_val", args_str=args, flag_type=bool
        ),
        True,
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "float_val", args_str=args, flag_type=float
        ),
        2.5,
    )
    self.assertIsNone(
        flag_utils.get_flag_from_libtpu_init_args(
            "nonexistent_flag", args_str=args
        )
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "nonexistent_flag", default=42, flag_type=int, args_str=args
        ),
        42,
    )

  def test_get_flag_from_libtpu_init_args_bool_variants(self):
    for truthy in ("true", "1", "t", "yes", "y"):
      self.assertTrue(
          flag_utils.get_flag_from_libtpu_init_args(
              "flag", args_str=f"--flag={truthy}", flag_type=bool
          )
      )
    for falsy in ("false", "0", "f", "no", "n"):
      self.assertFalse(
          flag_utils.get_flag_from_libtpu_init_args(
              "flag", args_str=f"--flag={falsy}", flag_type=bool
          )
      )

  def test_get_flag_value_fallback(self):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "test_p_state",
        -1,
        "Test p-state flag",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()

    # Flag is in flag_values with default -1 -> falls back to env var
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--test_p_state=3"}):
      val = flag_utils.get_flag_value(
          "test_p_state",
          default=None,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 3)

    # Flag is set in flag_values to non -1 -> uses flag_values
    test_flag_values.test_p_state = 7
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--test_p_state=3"}):
      val = flag_utils.get_flag_value(
          "test_p_state",
          default=None,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 7)

    # Flag does not exist in flag_values -> uses env var
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--unregistered_flag=42"}
    ):
      val = flag_utils.get_flag_value(
          "unregistered_flag",
          default=None,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 42)

    # Flag does not exist anywhere -> returns default
    with mock.patch.dict(os.environ, {}, clear=True):
      val = flag_utils.get_flag_value(
          "missing_flag",
          default=99,
          flag_type=int,
          flag_values=test_flag_values,
      )
      self.assertEqual(val, 99)

  def test_parse_libtpu_init_args_bare_dashes(self):
    args = "--flag1=1 -- - --flag2=2"
    parsed = flag_utils.parse_libtpu_init_args(args)
    self.assertEqual(parsed, {"flag1": "1", "flag2": "2"})

  def test_get_flag_from_libtpu_init_args_mixed_dashes(self):
    args = "--xla-tpu_dvfs-p_state=3"
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "xla_tpu_dvfs_p_state", args_str=args, flag_type=int
        ),
        3,
    )
    self.assertEqual(
        flag_utils.get_flag_from_libtpu_init_args(
            "--xla-tpu-dvfs-p-state", args_str=args, flag_type=int
        ),
        3,
    )

  def test_get_flag_from_libtpu_init_args_invalid_bool(self):
    with self.assertRaises(ValueError):
      flag_utils.get_flag_from_libtpu_init_args(
          "flag", args_str="--flag=invalid", flag_type=bool
      )

  def test_get_flag_value_zero_and_false(self):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "zero_flag",
        0,
        "Zero flag",
        flag_values=test_flag_values,
    )
    flags.DEFINE_bool(
        "false_flag",
        False,
        "False flag",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()

    # Zero and False values in flag_values should be respected and not fall back
    with mock.patch.dict(
        os.environ,
        {"LIBTPU_INIT_ARGS": "--zero_flag=5 --false_flag=true"},
    ):
      self.assertEqual(
          flag_utils.get_flag_value(
              "zero_flag", flag_type=int, flag_values=test_flag_values
          ),
          0,
      )
      self.assertFalse(
          flag_utils.get_flag_value(
              "false_flag", flag_type=bool, flag_values=test_flag_values
          )
      )

  def test_get_flag_value_xla_tpu_dvfs_p_state_defined_in_flags(self):
    test_flag_values = flags.FlagValues()
    flags.DEFINE_integer(
        "xla_tpu_dvfs_p_state",
        -1,
        "DVFS P-state",
        flag_values=test_flag_values,
    )
    test_flag_values.mark_as_parsed()

    # When set via flag
    test_flag_values.xla_tpu_dvfs_p_state = 7
    self.assertEqual(
        flag_utils.get_flag_value(
            "xla_tpu_dvfs_p_state",
            default=None,
            flag_type=int,
            flag_values=test_flag_values,
        ),
        7,
    )

    # When flag is -1 (default unset), fallback to LIBTPU_INIT_ARGS
    test_flag_values.xla_tpu_dvfs_p_state = -1
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=3"}
    ):
      self.assertEqual(
          flag_utils.get_flag_value(
              "xla_tpu_dvfs_p_state",
              default=None,
              flag_type=int,
              flag_values=test_flag_values,
          ),
          3,
      )

    # When flag is -1 and LIBTPU_INIT_ARGS has no p_state, return default
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertIsNone(
          flag_utils.get_flag_value(
              "xla_tpu_dvfs_p_state",
              default=None,
              flag_type=int,
              flag_values=test_flag_values,
          )
      )

  def test_get_flag_value_xla_tpu_dvfs_p_state_not_defined_in_flags(self):
    test_flag_values = flags.FlagValues()
    test_flag_values.mark_as_parsed()

    # Not defined in flags, fallback to LIBTPU_INIT_ARGS
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=5"}
    ):
      self.assertEqual(
          flag_utils.get_flag_value(
              "xla_tpu_dvfs_p_state",
              default=None,
              flag_type=int,
              flag_values=test_flag_values,
          ),
          5,
      )

    # Not defined in flags and not in env var
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertIsNone(
          flag_utils.get_flag_value(
              "xla_tpu_dvfs_p_state",
              default=None,
              flag_type=int,
              flag_values=test_flag_values,
          )
      )


if __name__ == "__main__":
  absltest.main()
