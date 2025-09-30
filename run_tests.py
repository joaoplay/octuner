#!/usr/bin/env python3

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return False


def run_unit_tests(coverage=False, verbose=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if coverage:
        cmd.extend(["--cov=octuner", "--cov-report=html", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    # Don't filter by markers for unit tests - run all tests in unit directory
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    # Don't filter by markers for integration tests - run all tests in integration directory
    
    return run_command(cmd, "Integration Tests")


def run_all_tests(coverage=False, verbose=False):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if coverage:
        cmd.extend(["--cov=octuner", "--cov-report=html", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Specific Test: {test_path}")


def run_fast_tests(verbose=False):
    """Run only fast tests (excluding slow and external tests)."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "not slow and not external"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Fast Tests Only")


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError:
        print("✗ pytest not installed")
        return False
    
    # Check if coverage is available (optional)
    try:
        import pytest_cov
        print("✓ pytest-cov available for coverage reports")
    except ImportError:
        print("! pytest-cov not available (coverage reports disabled)")
    
    # Check if required packages are installed
    required_packages = ["octuner", "openai", "google", "optuna", "yaml"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"! {package} not available (some tests may fail)")
    
    # Check for test files
    test_dir = Path("tests")
    if not test_dir.exists():
        print("✗ tests directory not found")
        return False
    
    unit_tests = list(test_dir.glob("unit/**/test_*.py"))
    integration_tests = list(test_dir.glob("integration/**/test_*.py"))
    
    print(f"✓ Found {len(unit_tests)} unit test files")
    print(f"✓ Found {len(integration_tests)} integration test files")
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Octuner Test Runner")
    parser.add_argument(
        "mode", 
        choices=["unit", "integration", "all", "fast", "check", "specific"],
        help="Test execution mode"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report (unit and all modes only)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--test-path", "-t",
        help="Specific test path for 'specific' mode (e.g., tests/unit/test_config/test_loader.py::TestConfigLoader::test_init)"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.mode == "check":
        success = check_test_environment()
        sys.exit(0 if success else 1)
    
    elif args.mode == "unit":
        success = run_unit_tests(coverage=args.coverage, verbose=args.verbose)
    
    elif args.mode == "integration":
        success = run_integration_tests(verbose=args.verbose)
    
    elif args.mode == "all":
        success = run_all_tests(coverage=args.coverage, verbose=args.verbose)
    
    elif args.mode == "fast":
        success = run_fast_tests(verbose=args.verbose)
    
    elif args.mode == "specific":
        if not args.test_path:
            print("Error: --test-path is required for 'specific' mode")
            sys.exit(1)
        success = run_specific_test(args.test_path, verbose=args.verbose)
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✓ All tests completed successfully!")
        if args.coverage and args.mode in ["unit", "all"]:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("✗ Some tests failed!")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()