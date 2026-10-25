from ai_agent.ensure_program_running import ensure_docker_running

import subprocess
import uuid
import logging
import time

from langchain_core.tools import tool

logger = logging.getLogger("sandbox")

IMAGE = "python:3.12-slim"
TIMEOUT = 30
MAX_OUTPUT = 3000


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + "\n...[truncated output]"


def _preview(text: str, limit: int = 300) -> str:
    """Short version of a text for the logs."""
    text = text.strip().replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


@tool
def run_python_in_sandbox(code: str) -> str:
    """Executes Python code within an isolated Docker container and returns the result. 

    Use this for writing, testing, or verifying code. The script must be
    self-contained (including imports) and must include its own assertions for testing purposes. 
    No network access is permitted; only the standard library is available.
    """
    
    code = unidecode.unidecode(code)

    name = f"agent-sandbox-{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker", "run", "--rm", "-i",
        "--name", name,
        "--network", "none",             # no network access
        "--memory", "256m",
        "--cpus", "1",
        "--pids-limit", "128",           # avoid fork bombs
        "--read-only",                   # read-only file system
        "--tmpfs", "/tmp:size=64m",      # only /tmp is writable
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",         # "nobody" user, not root
        IMAGE, "python", "-",            # the code arrives via stdin
    ]

    ensure_docker_running()

    logger.info("Sandbox %s: starting (%d characters of code)", name, len(code))
    logger.debug("Sandbox %s: code received:\n%s", name, code)

    start = time.perf_counter()

    try:
        r = subprocess.run(
            cmd, input=code, capture_output=True, text=True, timeout=TIMEOUT
        )

    except subprocess.TimeoutExpired:
        logger.warning("Sandbox %s: timed out after %ss, stopping container", name, TIMEOUT)
        kill = subprocess.run(["docker", "kill", name], capture_output=True, text=True)

        if kill.returncode != 0:
            logger.error("Sandbox %s: docker kill failed : %s", name, kill.stderr.strip())
        return f"Timeout: the code exceeded {TIMEOUT} seconds."

    except FileNotFoundError:
        logger.error("Docker not found: is it installed and in the PATH ?")
        return "Docker is not installed or is not running."

    except Exception:
        logger.exception("Sandbox %s: unexpected error", name)
        return "Unexpected error during code execution."
    
    duration = time.perf_counter() - start
    level = logging.INFO if r.returncode == 0 else logging.WARNING

    logger.log(
        level,
        "Sandbox %s: completed in %.2fs, exit_code=%s, stdout=%d chars, stderr=%d chars.",
        name, duration, r.returncode, len(r.stdout), len(r.stderr),
    )
    if r.stdout:
        logger.debug("Sandbox %s : stdout : %s", name, _preview(r.stdout))
    if r.stderr:
        logger.debug("Sandbox %s : stderr : %s", name, _preview(r.stderr))

    # Docker itself can fail (missing image, daemon stopped): code 125
    if r.returncode == 125:
        logger.error("Sandbox %s: Docker error: %s", name, r.stderr.strip())

    return _truncate(
        f"exit_code: {r.returncode}\n"
        f"--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr}"
    )