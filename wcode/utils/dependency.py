from shutil import which

class MissingDependencyError(Exception):
    """Raised when a required dependency is missing."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def is_tool(name: str, error: bool = False) -> bool:
    """Checks whether ``name`` is on ``PATH`` and is marked as an executable.

    Source:
    https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    :param name: Name of program to check for execution ability.
    :type name: str
    :param error: Whether to raise an error.
    :type error: bool. Defaults to ``False``.
    :return: Whether ``name`` is on PATH and is marked as an executable.
    :rtype: bool
    :raises MissingDependencyError: If ``error`` is ``True`` and ``name`` is
        not on ``PATH`` or is not marked as an executable.
    """
    found = which(name) is not None
    if not found and error:
        raise MissingDependencyError(name)
    return found
