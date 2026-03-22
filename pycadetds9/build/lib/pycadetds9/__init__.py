from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pycadetds9")
except PackageNotFoundError:
    __version__ = "unknown"
