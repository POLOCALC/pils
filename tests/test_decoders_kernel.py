"""
Tests for pils.decoders.KERNEL_utils module.

Following TDD methodology - these tests are written first and will fail initially.
"""

import logging

import pytest

from pils.decoders import KERNEL_utils


class TestChecksum:
    """Test the _checksum function."""

    def test_checksum_simple_message(self):
        """Test checksum computation for a simple message."""
        msg = b"\x01\x02\x03"
        result = KERNEL_utils._checksum(msg)
        # Sum of bytes: 1 + 2 + 3 = 6
        expected = (6).to_bytes(2, byteorder="little", signed=False)
        assert result == expected

    def test_checksum_with_header(self):
        """Test checksum computation strips header."""
        msg = b"\xaa\x55\x01\x02\x03"
        result = KERNEL_utils._checksum(msg)
        # Header is stripped, sum: 1 + 2 + 3 = 6
        expected = (6).to_bytes(2, byteorder="little", signed=False)
        assert result == expected

    def test_checksum_empty_after_header(self):
        """Test checksum with only header."""
        msg = b"\xaa\x55"
        result = KERNEL_utils._checksum(msg)
        expected = (0).to_bytes(2, byteorder="little", signed=False)
        assert result == expected

    def test_checksum_large_value(self):
        """Test checksum with large sum (overflow to 2 bytes)."""
        msg = b"\xff\xff\x01"
        result = KERNEL_utils._checksum(msg)
        # Sum: 255 + 255 + 1 = 511
        expected = (511).to_bytes(2, byteorder="little", signed=False)
        assert result == expected


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

    def test_decode_single_return_dict(self, kernel_msg):
        """Test decode_single return_dict parameter."""
        # The return_dict parameter exists but doesn't change behavior
        # This test just confirms the function accepts it
        header = KERNEL_utils.HEADER
        msg_type = kernel_msg.msg_address[0]
        msg = header + b"\x00" + msg_type + b"\x00\x00" + b"\x00" * 50

        # Just test that both call forms work
        result1 = kernel_msg.decode_single(msg, return_dict=False)
        result2 = kernel_msg.decode_single(msg, return_dict=True)
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_decode_multi_creates_dict(self, kernel_msg, tmp_path):
        """Test decode_multi returns a dictionary of decoded values."""
        # Create a test binary file with multiple messages
        test_file = tmp_path / "test_kernel.bin"
        header = KERNEL_utils.HEADER
        msg_type = kernel_msg.msg_address[0]
        msg = header + b"\x00" + msg_type + b"\x00\x00"

        # Write multiple messages
        with open(test_file, "wb") as f:
            f.write(msg * 3)

        result = kernel_msg.decode_multi(str(test_file))
        assert isinstance(result, dict)


class TestKernelMsgLogging:
    """Test that print() statements have been replaced with logging."""

    def test_decode_multi_uses_logging(self, tmp_path, caplog):
        """Test that decode_multi uses logger instead of print()."""
        kernel_msg = KERNEL_utils.KernelMsg()
        test_file = tmp_path / "test_kernel.bin"

        # Create a simple test message
        header = KERNEL_utils.HEADER
        msg_type = kernel_msg.msg_address[0]
        msg = header + b"\x00" + msg_type + b"\x00\x00"

        with open(test_file, "wb") as f:
            f.write(msg * 2)

        with caplog.at_level(logging.INFO):
            kernel_msg.decode_multi(str(test_file))

        # Check that logging was used (not print)
        # Look for log message about decoded values
        log_messages = [record.message for record in caplog.records]
        assert any("Decoded" in msg or "values" in msg.lower() for msg in log_messages)

    def test_logger_exists(self):
        """Test that logger is defined in the module."""
        assert hasattr(KERNEL_utils, "logger")
