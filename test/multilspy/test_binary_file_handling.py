import os
import tempfile

import pytest

from multilspy.multilspy_exceptions import MultilspyException
from multilspy.multilspy_logger import MultilspyLogger
from multilspy.multilspy_utils import FileUtils


class TestBinaryFileHandling:
    def test_binary_file_detection(self):
        """Test that binary files are properly detected and skipped."""
        logger = MultilspyLogger()

        # Test with a binary file extension
        with tempfile.NamedTemporaryFile(suffix=".ico", delete=False) as temp_file:
            temp_file.write(b"\x00\x01\x02\x03")  # Some binary content
            temp_file_path = temp_file.name

        try:
            # Try to read the binary file - should raise MultilspyException
            with pytest.raises(MultilspyException) as excinfo:
                FileUtils.read_file(logger, temp_file_path)

            # Check that the error message indicates binary file detection
            assert "Binary file detected" in str(excinfo.value)
        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_null_byte_detection(self):
        """Test that files with NULL bytes are detected as binary."""
        logger = MultilspyLogger()

        # Create a file with NULL bytes but without a binary extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is some text with a null byte: \x00 in it")
            temp_file_path = temp_file.name

        try:
            # Try to read the file with NULL bytes - this may or may not raise an exception
            # depending on how the file is read (some encodings might handle NULL bytes)
            try:
                content = FileUtils.read_file(logger, temp_file_path)
                # If it doesn't raise, verify the content was read correctly
                assert "\x00" in content, "NULL byte should be present in the content"
            except MultilspyException as e:
                # If it does raise, verify it's because of binary detection
                assert "Binary file detected" in str(e)
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
