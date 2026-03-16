# Workaround: pytest with pythonpath=["."] can cause the installed
# "langgraph" namespace package to become unresolvable.  Ensure the
# real site-packages langgraph path is on the module's __path__.
import sys
from pathlib import Path

_site_lg = None
for _sp in sys.path:
    if "site-packages" not in _sp:
        continue
    _candidate = Path(_sp) / "langgraph"
    if _candidate.is_dir() and (_candidate / "constants.py").exists():
        _site_lg = str(_candidate)
        break

if _site_lg:
    if "langgraph" in sys.modules:
        _mod = sys.modules["langgraph"]
        if hasattr(_mod, "__path__"):
            if _site_lg not in list(_mod.__path__):
                _mod.__path__.insert(0, _site_lg)
        else:
            _mod.__path__ = [_site_lg]
    else:
        from types import ModuleType

        _mod = ModuleType("langgraph")
        _mod.__path__ = [_site_lg]
        _mod.__package__ = "langgraph"
        sys.modules["langgraph"] = _mod
