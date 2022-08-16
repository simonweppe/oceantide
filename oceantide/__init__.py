"""Top-level package for oceantide."""

__version__ = "0.4.1"
__author__ = "Oceanum Developers"
__contact__ = "developers@oceanum.science"
__url__ = "http://github.com/wavespectra/oceantide"
__description__ = "Library for ocean tide prediction"
__keywords__ = "ocean tide prediction constituents xarray accessor"


def _import_read_functions(pkgname="input"):
    """Make read functions available at module level.

    Functions are imported here if:
        they are defined in a module wavespectra.input.{modname}
        they are named as read_{modname}

    """
    import os
    import glob
    from importlib import import_module

    here = os.path.dirname(os.path.abspath(__file__))
    for filename in glob.glob1(os.path.join(here, pkgname), "*.py"):
        module = os.path.splitext(filename)[0]
        if module == "__init__":
            continue
        func_name = f"read_{module}"
        try:
            globals()[func_name] = getattr(
                import_module(f"oceantide.{pkgname}.{module}"), func_name
            )
        except Exception as exc:
            print(f"Cannot import reading function {func_name}:\n{exc}")


_import_read_functions()
