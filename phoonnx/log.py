"""Logger used across phoonnx, falling back to the stdlib when ovos-utils is absent."""

try:
    from ovos_utils.log import LOG
except ImportError:
    import logging

    LOG = logging.getLogger("phoonnx")
