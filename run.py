import functools
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import triton
import yaml
from torch._inductor.codecache import build_code_hash, torch_key
from tritonbench.utils.run_utils import tritonbench_run

try:
    from torch._inductor.runtime.triton_compat import triton_key
except ImportError:
    from torch._inductor.codecache import triton_key

logging.basicConfig(
    level=logging.INFO,
    format="[Run Runner] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkKeys:
    """Container for benchmark identification keys."""

    helion_key: str
    torch_key: str
    triton_key: str
    triton_version: str

    def to_dict(self) -> dict:
        return {
            "helion_key": self.helion_key,
            "torch_key": self.torch_key,
            "triton_key": self.triton_key,
            "triton_version": self.triton_version,
        }


@functools.cache
def get_device_name() -> str:
    """
    Return a normalized name for the current CUDA device.

    Handles ROCm GCN architecture names and normalizes NVIDIA H100 naming.
    Returns 'unknown' if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return "unknown"

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    name = torch.cuda.get_device_name(device_idx)

    arch = getattr(props, "gcnArchName", None)
    if torch.version.hip is not None and arch is not None:
        name = f"{name} {arch}"

    # Normalize H100 naming (inconsistent reporting across drivers)
    if name.startswith("NVIDIA H100"):
        name = "NVIDIA H100"

    # Sanitize for filesystem compatibility
    return re.sub(r"[^\w\-]", "_", name).replace("__", "_")


@functools.cache
def get_helion_key() -> str:
    """Generate Helion key matching base_cache.py logic."""
    try:
        import helion

        helion_path = os.path.dirname(helion.__file__)
        combined_hash = hashlib.sha256()
        build_code_hash([helion_path], "", combined_hash)
        return combined_hash.hexdigest()
    except ImportError:
        logger.debug("Helion not installed, using placeholder key")
        return "helion_not_found"


@functools.cache
def get_torch_key() -> str:
    """Get the PyTorch cache key as a hex string."""
    return torch_key().hex()


@functools.cache
def get_triton_key() -> str:
    """Get the Triton cache key as a SHA256 hex string."""
    full_key = triton_key()
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def get_benchmark_keys() -> BenchmarkKeys:
    """Collect all benchmark identification keys."""
    return BenchmarkKeys(
        helion_key=get_helion_key(),
        torch_key=get_torch_key(),
        triton_key=get_triton_key(),
        triton_version=triton.__version__,
    )


def clear_inductor_cache() -> None:
    """
    Delete TorchInductor cache directories to ensure clean benchmark runs.

    Removes all directories matching /tmp/torchinductor_*/
    """
    cache_pattern = "/tmp/torchinductor_*"
    cache_dirs = glob.glob(cache_pattern)

    if not cache_dirs:
        logger.debug("No TorchInductor cache directories found")
        return

    removed_count = 0
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            logger.debug(f"Removed cache directory: {cache_dir}")
            removed_count += 1
        except OSError as e:
            logger.warning(f"Failed to remove {cache_dir}: {e}")

    logger.info(
        f"Cleared {removed_count} TorchInductor cache director{'ies' if removed_count != 1 else 'y'}"
    )


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON data atomically using a temporary file."""
    parent_dir = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=parent_dir,
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        try:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except Exception:
            os.unlink(tmp_file.name)
            raise
        shutil.move(tmp_file.name, path)


def _find_output_file(filename: str) -> Optional[Path]:
    """
    Locate the output file in expected locations.

    Searches TRITONBENCH_HELION_PATH first, then current directory.
    """
    search_path = os.environ.get("TRITONBENCH_HELION_PATH", ".")
    candidates = [
        Path(search_path) / filename,
        Path(filename),
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _extract_output_arg(args: List[str]) -> Optional[str]:
    """
    Extract the --output argument value from a list of arguments.

    Handles both '--output value' and '--output=value' formats.
    """
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--output="):
            return arg.split("=", 1)[1]
    return None


def post_process_results(output_filename: str) -> Tuple[bool, Optional[Path]]:
    """
    Process benchmark results: inject keys and move to current directory.

    Args:
        output_filename: Name of the output file to process.

    Returns:
        Tuple of (success: bool, destination_path: Optional[Path])
    """
    source_path = _find_output_file(output_filename)
    if source_path is None:
        logger.warning(f"Output file not found: {output_filename}")
        return False, None

    logger.info(f"Processing results from: {source_path}")

    keys = get_benchmark_keys()

    try:
        with open(source_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {source_path}: {e}")
        return False, None
    except OSError as e:
        logger.error(f"Failed to read {source_path}: {e}")
        return False, None

    # Inject keys into each record's extra_info
    if isinstance(data, list):
        for record in data:
            benchmark = record.get("benchmark")
            if benchmark is not None:
                extra_info = benchmark.setdefault("extra_info", {})
                extra_info.update(keys.to_dict())

    try:
        _write_json_atomic(source_path, data)
    except OSError as e:
        logger.error(f"Failed to write updated JSON: {e}")
        return False, None

    device_name = get_device_name()
    if device_name not in source_path.stem:
        new_filename = f"{source_path.stem}_{device_name}{source_path.suffix}"
    else:
        new_filename = source_path.name

    dest_path = Path.cwd() / new_filename

    # Move to destination
    try:
        shutil.move(str(source_path), str(dest_path))
        logger.info(f"Successfully moved output to: {dest_path}")
        return True, dest_path
    except OSError as e:
        logger.error(f"Failed to move file to {dest_path}: {e}")
        return False, None


def extract_outputs_from_config(config_path: str) -> Dict[str, str]:
    """
    Parse a YAML config to find all --output arguments.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict mapping benchmark names to their output file paths.
    """
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML config {config_path}: {e}")
        return {}

    if not isinstance(config, dict):
        return {}

    outputs = {}
    for benchmark_name, benchmark_config in config.items():
        if not isinstance(benchmark_config, dict):
            continue

        args_str = benchmark_config.get("args", "")
        if not isinstance(args_str, str):
            continue

        output = _extract_output_arg(args_str.split())
        if output:
            outputs[benchmark_name] = output
            logger.debug(f"Found output for benchmark '{benchmark_name}': {output}")

    return outputs


def run(args: Optional[List[str]] = None) -> None:
    """
    Main entry point: run benchmarks and post-process results.

    Args:
        args: Command-line arguments. Uses sys.argv[1:] if None.
    """
    expected_outputs: Dict[str, str] = {}

    # Check env var for config file
    config_path = os.environ.get("TRITONBENCH_RUN_CONFIG")
    if config_path:
        logger.info(f"Using config from TRITONBENCH_RUN_CONFIG: {config_path}")
        config_outputs = extract_outputs_from_config(config_path)
        expected_outputs.update(config_outputs)

    # Check direct CLI arguments
    current_args = args if args is not None else sys.argv[1:]
    cli_output = _extract_output_arg(current_args)
    if cli_output:
        expected_outputs["cli"] = cli_output

    if expected_outputs:
        logger.info(f"Expecting {len(expected_outputs)} output file(s):")
        for benchmark_name, output_file in expected_outputs.items():
            logger.info(f"  - {benchmark_name}: {output_file}")
    else:
        logger.info("No output files expected")

    # Clear TorchInductor cache before running benchmarks
    clear_inductor_cache()

    # Run benchmarks
    logger.info("Starting benchmark run...")
    tritonbench_run(args)
    logger.info("Benchmark run completed")

    # Post-process all expected output files
    if not expected_outputs:
        return

    logger.info(f"Post-processing {len(expected_outputs)} output file(s)...")

    results: Dict[str, Tuple[bool, Optional[Path]]] = {}
    for benchmark_name, output_file in expected_outputs.items():
        logger.info(
            f"Processing output for benchmark '{benchmark_name}': {output_file}"
        )
        success, dest_path = post_process_results(output_file)
        results[benchmark_name] = (success, dest_path)

    # Print summary
    successful = [name for name, (success, _) in results.items() if success]
    failed = [name for name, (success, _) in results.items() if not success]

    logger.info("=" * 50)
    logger.info("POST-PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(
        f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}"
    )

    if successful:
        logger.info("Successful:")
        for name in successful:
            _, dest_path = results[name]
            logger.info(f"  ✓ {name} -> {dest_path}")

    if failed:
        logger.warning("Failed:")
        for name in failed:
            logger.warning(f"  ✗ {name}")


if __name__ == "__main__":
    run()
