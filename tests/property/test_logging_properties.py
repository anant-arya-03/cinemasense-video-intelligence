"""
Property-based tests for Error Logging Completeness.

Feature: cinemasense-stabilization
Property 20: Error Logging Completeness
Validates: Requirements 12.1, 12.3

Tests that for any caught exception in the system, the error SHALL be logged
with timestamp, error type, message, and stack trace.
"""

import sys
import logging
import re
from pathlib import Path
from typing import Type
from io import StringIO
from unittest.mock import patch, MagicMock

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings

from cinemasense.logging_setup import (
    LoggingConfig,
    log_exception,
    get_logger,
    DEFAULT_LOG_FORMAT,
    DEFAULT_DATE_FORMAT
)


# Strategy for generating various exception types
exception_types_strategy = st.sampled_from([
    ValueError,
    TypeError,
    RuntimeError,
    IOError,
    KeyError,
    AttributeError,
    FileNotFoundError,
    PermissionError,
    ZeroDivisionError,
    IndexError,
])

# Strategy for generating exception messages
exception_message_strategy = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S', 'Z'))
).filter(lambda x: x.strip())


class TestErrorLoggingCompleteness:
    """
    Property tests for error logging completeness.
    
    Feature: cinemasense-stabilization, Property 20: Error Logging Completeness
    Validates: Requirements 12.1, 12.3
    """
    
    def setup_method(self):
        """Reset logging configuration before each test."""
        LoggingConfig.reset()
    
    def teardown_method(self):
        """Clean up after each test."""
        LoggingConfig.reset()
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy
    )
    @settings(max_examples=100)
    def test_log_exception_includes_timestamp(
        self,
        exception_type: Type[Exception],
        error_message: str
    ):
        """
        Property 20: Error Logging Completeness (timestamp)
        
        For any caught exception, the logged output SHALL include a timestamp.
        
        Validates: Requirements 12.1, 12.3
        """
        # Create a string stream to capture log output
        log_stream = StringIO()
        
        # Setup logging with our stream handler
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        # Add a stream handler to capture output
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Create and log the exception
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, "Test error occurred")
        
        # Get the log output
        log_output = log_stream.getvalue()
        
        # Verify timestamp pattern exists (YYYY-MM-DD HH:MM:SS format)
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert re.search(timestamp_pattern, log_output), (
            f"Log output should contain timestamp in format YYYY-MM-DD HH:MM:SS\n"
            f"Log output: {log_output}"
        )
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy
    )
    @settings(max_examples=100)
    def test_log_exception_includes_error_type(
        self,
        exception_type: Type[Exception],
        error_message: str
    ):
        """
        Property 20: Error Logging Completeness (error type)
        
        For any caught exception, the logged output SHALL include the error type.
        
        Validates: Requirements 12.1, 12.3
        """
        log_stream = StringIO()
        
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, "Test error occurred")
        
        log_output = log_stream.getvalue()
        
        # Verify error type is in the output
        assert exception_type.__name__ in log_output, (
            f"Log output should contain error type '{exception_type.__name__}'\n"
            f"Log output: {log_output}"
        )
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy
    )
    @settings(max_examples=100)
    def test_log_exception_includes_message(
        self,
        exception_type: Type[Exception],
        error_message: str
    ):
        """
        Property 20: Error Logging Completeness (message)
        
        For any caught exception, the logged output SHALL include the error message.
        
        Validates: Requirements 12.1, 12.3
        """
        log_stream = StringIO()
        
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, "Test error occurred")
        
        log_output = log_stream.getvalue()
        
        # Verify error message is in the output
        # Note: Some exception types (like KeyError) may escape special characters
        # in their string representation, so we check for either the raw message
        # or its repr (escaped) form
        message_present = (
            error_message in log_output or
            repr(error_message) in log_output or
            error_message.encode('unicode_escape').decode('ascii') in log_output
        )
        assert message_present, (
            f"Log output should contain error message '{error_message}' "
            f"(or its escaped representation)\n"
            f"Log output: {log_output}"
        )
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy
    )
    @settings(max_examples=100)
    def test_log_exception_includes_stack_trace(
        self,
        exception_type: Type[Exception],
        error_message: str
    ):
        """
        Property 20: Error Logging Completeness (stack trace)
        
        For any caught exception, the logged output SHALL include a stack trace.
        
        Validates: Requirements 12.1, 12.3
        """
        log_stream = StringIO()
        
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, "Test error occurred", include_traceback=True)
        
        log_output = log_stream.getvalue()
        
        # Verify stack trace indicators are present
        # Stack traces contain "Traceback" and file references
        assert "Traceback" in log_output, (
            f"Log output should contain 'Traceback' for stack trace\n"
            f"Log output: {log_output}"
        )
        
        # Also verify it contains file reference pattern
        file_pattern = r'File ".*", line \d+'
        assert re.search(file_pattern, log_output), (
            f"Log output should contain file reference in stack trace\n"
            f"Log output: {log_output}"
        )
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy
    )
    @settings(max_examples=100)
    def test_log_exception_uses_error_level(
        self,
        exception_type: Type[Exception],
        error_message: str
    ):
        """
        Property 20: Error Logging Completeness (log level)
        
        For any caught exception, the log entry SHALL be at ERROR level.
        
        Validates: Requirements 12.1, 12.3
        """
        log_stream = StringIO()
        
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, "Test error occurred")
        
        log_output = log_stream.getvalue()
        
        # Verify ERROR level is in the output
        assert "ERROR" in log_output, (
            f"Log output should contain 'ERROR' level indicator\n"
            f"Log output: {log_output}"
        )
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy,
        custom_prefix=st.text(min_size=1, max_size=50).filter(lambda x: x.strip())
    )
    @settings(max_examples=100)
    def test_log_exception_includes_custom_message(
        self,
        exception_type: Type[Exception],
        error_message: str,
        custom_prefix: str
    ):
        """
        Property 20: Error Logging Completeness (custom message)
        
        For any caught exception with a custom message prefix, the logged
        output SHALL include the custom message.
        
        Validates: Requirements 12.1, 12.3
        """
        log_stream = StringIO()
        
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, custom_prefix)
        
        log_output = log_stream.getvalue()
        
        # Verify custom message prefix is in the output
        assert custom_prefix in log_output, (
            f"Log output should contain custom message prefix '{custom_prefix}'\n"
            f"Log output: {log_output}"
        )
    
    @given(
        exception_type=exception_types_strategy,
        error_message=exception_message_strategy
    )
    @settings(max_examples=100)
    def test_log_exception_without_traceback_still_logs_type_and_message(
        self,
        exception_type: Type[Exception],
        error_message: str
    ):
        """
        Property 20: Error Logging Completeness (minimal logging)
        
        For any caught exception logged without traceback, the output SHALL
        still include error type and message.
        
        Validates: Requirements 12.1, 12.3
        """
        log_stream = StringIO()
        
        LoggingConfig.reset()
        logger = LoggingConfig.setup(console_output=False)
        
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        try:
            raise exception_type(error_message)
        except Exception as e:
            log_exception(logger, e, "Test error", include_traceback=False)
        
        log_output = log_stream.getvalue()
        
        # Verify error type is present
        assert exception_type.__name__ in log_output, (
            f"Log output should contain error type '{exception_type.__name__}' "
            f"even without traceback\n"
            f"Log output: {log_output}"
        )
        
        # Verify error message is present
        # Note: Some exception types (like KeyError) may escape special characters
        # in their string representation, so we check for either the raw message
        # or its repr (escaped) form
        message_present = (
            error_message in log_output or
            repr(error_message) in log_output or
            error_message.encode('unicode_escape').decode('ascii') in log_output
        )
        assert message_present, (
            f"Log output should contain error message '{error_message}' "
            f"(or its escaped representation) even without traceback\n"
            f"Log output: {log_output}"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
