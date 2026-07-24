"""Fail-fast environment checks for AstrAI's DeepEP V2 training backend."""

import re
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return ()
    return tuple(int(part or 0) for part in match.groups())


def main() -> None:
    errors = []
    torch_version = version_tuple(torch.__version__)
    cuda_version = version_tuple(torch.version.cuda or "")
    torch_nccl_version = (
        torch.cuda.nccl.version() if torch.cuda.is_available() else ()
    )
    try:
        runtime_nccl_text = version("nvidia-nccl-cu13")
        runtime_nccl_version = version_tuple(runtime_nccl_text)
    except PackageNotFoundError:
        runtime_nccl_text = ".".join(map(str, torch_nccl_version))
        runtime_nccl_version = tuple(torch_nccl_version)

    if torch_version < (2, 10, 0):
        errors.append(f"PyTorch >= 2.10 is required, found {torch.__version__}")
    if cuda_version < (12, 3, 0):
        errors.append(f"CUDA >= 12.3 is required, found {torch.version.cuda}")
    if runtime_nccl_version < (2, 30, 4):
        errors.append(
            f"NCCL runtime >= 2.30.4 is required, found {runtime_nccl_text}"
        )
    if not torch.cuda.is_available():
        errors.append("CUDA is not available to PyTorch")
    elif torch.cuda.get_device_capability() < (9, 0):
        errors.append(
            "DeepEP requires Hopper or newer; found compute capability "
            f"{torch.cuda.get_device_capability()}"
        )

    deep_ep_version = "unavailable"
    try:
        import deep_ep

        deep_ep_version = getattr(deep_ep, "__version__", "unknown")
        if not hasattr(deep_ep, "ElasticBuffer"):
            errors.append("Installed DeepEP does not expose the V2 ElasticBuffer API")
    except Exception as exc:  # Import validates linked/runtime NCCL consistency.
        errors.append(f"DeepEP import failed: {exc}")

    print(
        "DEEPEP_ENV",
        f"python={sys.version.split()[0]}",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"torch_nccl={torch_nccl_version}",
        f"runtime_nccl={runtime_nccl_text}",
        f"deepep={deep_ep_version}",
    )
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("DEEPEP_ENV_OK")


if __name__ == "__main__":
    main()
