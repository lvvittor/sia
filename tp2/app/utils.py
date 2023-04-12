import func_timeout
from functools import wraps


def timeout(seconds: int):
    """
    A decorator that times out a function after a given amount of seconds
    and returns False if the function times out.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func_timeout.func_timeout(
                    seconds, func, args=args, kwargs=kwargs
                )
            except func_timeout.FunctionTimedOut:
                return False

        return wrapper

    return decorator
