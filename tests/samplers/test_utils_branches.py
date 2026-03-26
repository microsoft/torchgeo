# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Additional branch coverage tests for sampler utilities."""

from torchgeo.samplers.utils import _to_tuple


class TestToTupleBranches:
    """Test branch coverage for _to_tuple function."""

    def test_to_tuple_with_int(self) -> None:
        """Test _to_tuple with int input - should take first branch."""
        result = _to_tuple(5)
        assert result == (5, 5)

    def test_to_tuple_with_float(self) -> None:
        """Test _to_tuple with float input - should take first branch."""
        result = _to_tuple(3.14)
        assert result == (3.14, 3.14)

    def test_to_tuple_with_int_tuple(self) -> None:
        """Test _to_tuple with int tuple - should take second branch."""
        result = _to_tuple((4, 8))
        assert result == (4, 8)

    def test_to_tuple_with_float_tuple(self) -> None:
        """Test _to_tuple with float tuple - should take second branch."""
        result = _to_tuple((2.5, 7.5))
        assert result == (2.5, 7.5)

    def test_to_tuple_preserves_identity(self) -> None:
        """Test that _to_tuple returns the same tuple when given a tuple."""
        input_tuple = (10, 20)
        result = _to_tuple(input_tuple)
        assert result is input_tuple  # Should be same object, not just equal
