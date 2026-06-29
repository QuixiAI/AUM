__version__ = "0.1.0.dev0"

# Lazy top-level exports (PEP 562): importing a leaf submodule such as
# aum_ssm.modules.ssd_reference must not drag in the Triton kernels, so the
# package CPU-tests on machines without Triton/CUDA.
__all__ = ["AumConfig", "AumLMHeadModel"]


def __getattr__(name):
    if name == "AumConfig":
        from aum_ssm.models.config_aum import AumConfig
        return AumConfig
    if name == "AumLMHeadModel":
        from aum_ssm.models.aum_lm import AumLMHeadModel
        return AumLMHeadModel
    raise AttributeError(f"module 'aum_ssm' has no attribute {name!r}")
