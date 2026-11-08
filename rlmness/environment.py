"""Minimal inherited environment for execution processes."""
import os

_INHERITED = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "PROGRAMDATA", "HOME", "USERPROFILE", "HOMEDRIVE",
    "HOMEPATH", "TMP", "TEMP", "TMPDIR", "LANG", "LANGUAGE", "TZ", "TERM",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "APPDATA", "LOCALAPPDATA",
})


def execution_environment(overrides=None):
    """Inherit OS essentials; credentials/control variables require explicit opt-in."""
    result = {key: value for key, value in os.environ.items()
              if key.upper() in _INHERITED or key.upper().startswith("LC_")}
    if overrides:
        for key, value in overrides.items():
            if value is None:
                result.pop(key, None)
            else:
                result[str(key)] = str(value)
    return result
