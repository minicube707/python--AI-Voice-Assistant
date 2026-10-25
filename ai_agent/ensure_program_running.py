
import subprocess
import logging
import requests
import time

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


def ensure_docker_running(timeout: int = 20) -> bool:
    """
    Verify that Docker Engine is available. 
    If necessary, start Docker Desktop and wait for it to initialize.
    """

    # 1. Check whether Docker Engine is already responding
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Docker Engine is already up and running.")
            return True

    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Docker unavailable: %s", exc)

    # 2. Launch Docker Desktop
    logger.info("Starting Docker Desktop...")

    subprocess.Popen(
        [docker_desktop],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # 3. Wait for Docker Engine to respond
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            if result.returncode == 0:
                logger.info("Docker Engine is now up and running.")
                return True

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        time.sleep(2)

    error_msg = f"Docker Engine is not available after {timeout} seconds. " \
        "Check Docker Desktop and its Linux engine."

    logger.exception(error_msg)
    raise TimeoutError(error_msg)