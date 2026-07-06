import importlib.util
import os
import sys


def load_pkg_module(pkg_name: str, pkg_dir: str, submodule: str):
    """
    Load <pkg_dir>/<submodule>.py under sys.modules key <pkg_name>.<submodule>.

    The parent package (<pkg_name>) is also registered so that relative imports
    inside the submodule (e.g. `from .data import ...`) resolve correctly.
    """
    full_name = f"{pkg_name}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name,
            os.path.join(pkg_dir, "__init__.py"),
            submodule_search_locations=[pkg_dir],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [pkg_dir]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        pkg_spec.loader.exec_module(pkg)

    file_path = os.path.join(pkg_dir, submodule + ".py")
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod
