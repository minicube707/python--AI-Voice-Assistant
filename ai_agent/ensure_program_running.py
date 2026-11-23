
import subprocess
import logging
import requests
import time
import platform
import os
import shutil

logger = logging.getLogger("ai_agent")

def ensure_ollama_running(timeout: int = 10):

    url = "http://localhost:11434/api/tags"

    # Check if Ollama is already responding
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return
    except requests.RequestException:
        pass

    # Start Ollama if necessary
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait until the service is available
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        time.sleep(1)

    error_msg = "Ollama did not start within the allotted time."

    logger.exception(error_msg)
    raise Exception(error_msg)


def get_docker_desktop_path():
    system = platform.system()

    if system == "Windows":
        paths = [
            os.path.expandvars(r"%ProgramFiles%\Docker\Docker\Docker Desktop.exe"),
            os.path.expandvars(r"%LocalAppData%\Programs\Docker\Docker Desktop.exe"),
        ]

    elif system == "Darwin":
        paths = ["/Applications/Docker.app/Contents/MacOS/Docker"]

    elif system == "Linux":
        paths = ["docker-desktop"]

    else:
        return None

    for path in paths:
        if os.path.isfile(path) or shutil.which(path):
            return path

    return None


def ensure_docker_running(timeout: int = 60) -> bool:
    """
    Verify that Docker Engine is available.
    If necessary, start Docker Desktop and wait for it to initialize.
    """

    def docker_available():
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    # 1. Check Docker Engine
    if docker_available():
        logger.info("Docker Engine is already up and running.")
        return True

    # 2. Find Docker Desktop
    docker_desktop = get_docker_desktop_path()

    if not docker_desktop:
        raise FileNotFoundError(
            "Docker Desktop executable could not be found."
        )

    # 3. Start Docker Desktop
    logger.info("Starting Docker Desktop...")

    subprocess.Popen(
        [docker_desktop],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # 4. Wait for Docker Engine
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        if docker_available():
            logger.info("Docker Engine is now up and running.")
            return True

        time.sleep(2)

    error_msg = (
        f"Docker Engine is not available after {timeout} seconds. "
        "Check Docker Desktop and its Linux engine."
    )

    logger.error(error_msg)
    raise TimeoutError(error_msg)