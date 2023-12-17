"""Top-level package for oceantide."""

__version__ = "0.6.0"


def _import_read_functions(pkgname: str = "input"):
    """Make read functions available at module level.

    Parameters
    ----------
    pkgname (str)
        Name of the package to import functions from.

    Functions are imported here if:

    - they are defined in a module wavespectra.input.{modname}
    - they are named as read_{modname}

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
