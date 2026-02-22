"""
Verification script to check if the MLOps pipeline is set up correctly.
"""

import os
import sys
import subprocess
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"{Colors.GREEN}✓{Colors.END} {description}: {filepath}")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.END} {description}: {filepath} not found")
        return False


def check_directory(dirpath, description):
    """Check if a directory exists."""
    if os.path.isdir(dirpath):
        print(f"{Colors.GREEN}✓{Colors.END} {description}: {dirpath}")
        return True
    else:
        print(f"{Colors.YELLOW}!{Colors.END} {description}: {dirpath} not found (may be created later)")
        return False


def check_command(command, description):
    """Check if a command is available."""
    try:
        result = subprocess.run([command, '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"{Colors.GREEN}✓{Colors.END} {description}: {version}")
            return True
    except:
        pass

    print(f"{Colors.RED}✗{Colors.END} {description}: not found")
    return False


def check_python_package(package, description):
    """Check if a Python package is installed."""
    try:
        __import__(package)
        print(f"{Colors.GREEN}✓{Colors.END} {description}")
        return True
    except ImportError:
        print(f"{Colors.RED}✗{Colors.END} {description}: not installed")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print(f"{Colors.BLUE}MLOps Pipeline - Setup Verification{Colors.END}")
    print("=" * 70)

    all_checks = []

    # Check Python version
    print(f"\n{Colors.BLUE}[1] Python Environment{Colors.END}")
    print(f"Python version: {sys.version.split()[0]}")
    python_version = tuple(map(int, sys.version.split()[0].split('.')[:2]))
    if python_version >= (3, 10):
        print(f"{Colors.GREEN}✓{Colors.END} Python 3.10+ requirement met")
        all_checks.append(True)
    else:
        print(f"{Colors.RED}✗{Colors.END} Python 3.10+ required (found {sys.version.split()[0]})")
        all_checks.append(False)

    # Check core files
    print(f"\n{Colors.BLUE}[2] Core Source Files{Colors.END}")
    all_checks.append(check_file('src/data_preprocessing.py', 'Data preprocessing'))
    all_checks.append(check_file('src/model.py', 'Model architecture'))
    all_checks.append(check_file('src/train.py', 'Training script'))
    all_checks.append(check_file('src/inference_service.py', 'Inference API'))

    # Check configuration files
    print(f"\n{Colors.BLUE}[3] Configuration Files{Colors.END}")
    all_checks.append(check_file('requirements.txt', 'Python dependencies'))
    all_checks.append(check_file('Dockerfile', 'Docker configuration'))
    all_checks.append(check_file('docker-compose.yml', 'Docker Compose'))
    all_checks.append(check_file('.gitignore', 'Git ignore'))

    # Check test files
    print(f"\n{Colors.BLUE}[4] Test Files{Colors.END}")
    all_checks.append(check_file('tests/test_preprocessing.py', 'Preprocessing tests'))
    all_checks.append(check_file('tests/test_model.py', 'Model tests'))
    all_checks.append(check_file('tests/test_inference_service.py', 'API tests'))

    # Check CI/CD files
    print(f"\n{Colors.BLUE}[5] CI/CD Configuration{Colors.END}")
    all_checks.append(check_file('.github/workflows/ci.yml', 'CI pipeline'))
    all_checks.append(check_file('.github/workflows/cd.yml', 'CD pipeline'))

    # Check deployment files
    print(f"\n{Colors.BLUE}[6] Deployment Files{Colors.END}")
    all_checks.append(check_file('deployment/k8s/deployment.yaml', 'K8s deployment'))
    all_checks.append(check_file('deployment/k8s/service.yaml', 'K8s service'))
    all_checks.append(check_file('deployment/smoke_tests.py', 'Smoke tests'))

    # Check directories
    print(f"\n{Colors.BLUE}[7] Project Directories{Colors.END}")
    check_directory('data/raw', 'Data directory')
    check_directory('models', 'Models directory')
    check_directory('.dvc', 'DVC directory')

    # Check installed packages
    print(f"\n{Colors.BLUE}[8] Python Packages{Colors.END}")
    all_checks.append(check_python_package('torch', 'PyTorch'))
    all_checks.append(check_python_package('fastapi', 'FastAPI'))
    all_checks.append(check_python_package('mlflow', 'MLflow'))
    all_checks.append(check_python_package('pytest', 'pytest'))

    # Check external tools
    print(f"\n{Colors.BLUE}[9] External Tools{Colors.END}")
    docker_available = check_command('docker', 'Docker')
    kubectl_available = check_command('kubectl', 'kubectl')

    # Summary
    print("\n" + "=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100

    print(f"{Colors.BLUE}Verification Summary{Colors.END}")
    print("=" * 70)
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")

    if percentage == 100:
        print(f"\n{Colors.GREEN}✓ All checks passed! Your setup is complete.{Colors.END}")
        status = 0
    elif percentage >= 80:
        print(f"\n{Colors.YELLOW}! Most checks passed. Review failed items above.{Colors.END}")
        status = 0
    else:
        print(f"\n{Colors.RED}✗ Several checks failed. Please review the setup.{Colors.END}")
        status = 1

    print("\n" + "=" * 70)

    # Recommendations
    if not docker_available:
        print(f"{Colors.YELLOW}Recommendation:{Colors.END} Install Docker for containerization features")

    if not kubectl_available:
        print(f"{Colors.YELLOW}Recommendation:{Colors.END} Install kubectl for Kubernetes deployment")

    if not os.path.exists('data/raw/cat') or not os.path.exists('data/raw/dog'):
        print(f"{Colors.YELLOW}Next Step:{Colors.END} Generate sample data:")
        print(f"  python scripts/generate_sample_data.py --num_samples 100")

    if not os.path.exists('models/best_model.pth'):
        print(f"{Colors.YELLOW}Next Step:{Colors.END} Train the model:")
        print(f"  python src/train.py --data_dir data/raw --num_epochs 5")

    print()
    sys.exit(status)


if __name__ == "__main__":
    main()
