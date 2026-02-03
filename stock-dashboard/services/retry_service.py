"""
Retry service - automatic retry with exponential backoff for API calls.
"""

import time
import functools
from typing import Callable, Tuple, Type


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback(attempt, exception, delay) called before each retry

    Example:
        @retry(max_attempts=3, delay=1, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        # Final attempt failed, raise the exception
                        raise

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, e, current_delay)
                    else:
                        print(f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}")

                    # Wait before next attempt
                    time.sleep(current_delay)
                    current_delay *= backoff

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Async version of retry decorator."""
    import asyncio

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise

                    print(f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retrying a block of code.

    Example:
        with RetryContext(max_attempts=3) as retry:
            for attempt in retry:
                result = risky_operation()
    """
    def __init__(self, max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,)):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
        self.attempt = 0
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self.attempt >= self.max_attempts:
            if self.last_exception:
                raise self.last_exception
            raise StopIteration

        self.attempt += 1
        return self.attempt

    def record_failure(self, exception):
        """Record a failure and wait before next attempt."""
        self.last_exception = exception

        if self.attempt < self.max_attempts:
            wait_time = self.delay * (self.backoff ** (self.attempt - 1))
            time.sleep(wait_time)
            return True  # Will retry

        return False  # No more retries


def fetch_with_retry(fetch_func, *args, max_attempts=3, **kwargs):
    """
    Utility function to fetch data with retry.

    Args:
        fetch_func: The function to call
        *args: Arguments to pass to fetch_func
        max_attempts: Number of retry attempts
        **kwargs: Keyword arguments to pass to fetch_func

    Returns:
        Result of fetch_func or None if all attempts fail
    """
    delay = 1.0
    last_error = None

    for attempt in range(max_attempts):
        try:
            result = fetch_func(*args, **kwargs)
            return result
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")

            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= 2

    print(f"All {max_attempts} attempts failed. Last error: {last_error}")
    return None
