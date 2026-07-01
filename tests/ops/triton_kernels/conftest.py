# The kernels under aum_ssm/ops/triton are the deferred NVIDIA path; their tests need Triton.
# Skip collecting them on machines without Triton (e.g. Apple Silicon dev boxes).
try:
    import triton.language  # noqa: F401  (the kernels need triton.language)
except ImportError:
    collect_ignore_glob = ["*"]
