from datetime import datetime
import random

def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

def current_time() -> str:
    """Get the current time."""
    return str(datetime.now())

def random_number(min_value: int, max_value: int) -> int:
    """Get a random number between min_value and max_value."""
    return random.randint(min_value, max_value)