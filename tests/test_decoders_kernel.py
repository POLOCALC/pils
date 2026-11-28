"""Tests for pils.decoders.KERNEL_utils module."""

import pytest

from pils.decoders import KERNEL_utils


class TestKernelMsg:
    """Test the KernelMsg class."""

    @pytest.fixture
    def kernel_msg(self):
        """Create a KernelMsg instance."""
        return KERNEL_utils.KernelMsg()

    def test_kernel_msg_init(self, kernel_msg):
        """Test KernelMsg initialization."""
        assert hasattr(kernel_msg, "msg_address")
        assert isinstance(kernel_msg.msg_address, list)
        assert len(kernel_msg.msg_address) > 0

    def test_decode_single_basic(self, kernel_msg):
        """Test decode_single returns a dict with Type field on valid input."""
        # Test that decode_single gracefully handles minimal message
        # and returns at least the Type field (try/except catches incomplete data)
        header = KERNEL_utils.HEADER
        msg_type = kernel_msg.msg_address[0]
        # Create message with sufficient data for parsing
        msg = header + b"\x00" + msg_type + b"\x00\x00" + b"\x00" * 50

        result = kernel_msg.decode_single(msg)
        assert isinstance(result, dict)
        # At minimum, Type field should be decoded
        assert "Type" in result
