"""Static Service-tier discovery (design 2026-06-25-services-tier-design.md).

Pure (no Executor, no network): reads the repo on disk + the in-progress graph,
and appends confidence-annotated ``SERVICE`` nodes. Sources: CI ``services:`` /
compose ``services:`` (confirmed), ``*_URL`` config schemes + ``package->service``
table (inferred). Structural ``Package->Service`` edges are emitted only for
confirmed services (no false necessary conditions).
"""

from __future__ import annotations

import os
import re as _re
from urllib.parse import urlparse

try:  # PyYAML is available; degrade gracefully if ever absent.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.python_deps.depgraph.service_tables import KNOWN_SERVICE_KINDS

# URL scheme -> canonical service kind.
_SCHEME_TO_KIND: dict[str, str] = {
    "postgres": "postgres", "postgresql": "postgres", "postgresql+psycopg2": "postgres",
    "mysql": "mysql", "mysql+pymysql": "mysql",
    "redis": "redis", "rediss": "redis",
    "mongodb": "mongo", "mongodb+srv": "mongo",
    "amqp": "rabbitmq", "amqps": "rabbitmq",
    "elasticsearch": "elasticsearch",
}


def service_from_url(value: str) -> tuple[str, str | None, int | None] | None:
    """`(kind, host, port)` for a service URL/scheme, or None if unknown."""
    if not value or "://" not in value:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    kind = _SCHEME_TO_KIND.get(parsed.scheme.lower())
    if kind is None:
        return None
    host = parsed.hostname
    port = parsed.port if _has_port(parsed) else None
    return kind, host, port


def _has_port(parsed) -> bool:
    try:
        return parsed.port is not None
    except ValueError:
        return False


def _kind_of(service_name: str, image: str | None) -> str | None:
    """Recognize a service kind from its compose/CI name or image."""
    name = (service_name or "").lower()
    for kind in KNOWN_SERVICE_KINDS:
        if kind in name:
            return kind
    img = (image or "").lower()
    for kind in KNOWN_SERVICE_KINDS:
        if kind in img:
            return kind
    # common image aliases not equal to the kind token
    if "postgres" in img or "postgis" in img:
        return "postgres"
    return None


def _port_of(entry: dict) -> int | None:
    ports = entry.get("ports")
    if isinstance(ports, list) and ports:
        first = str(ports[0])
        tail = first.split(":")[-1].split("/")[0]
        return int(tail) if tail.isdigit() else None
    return None


def _healthcheck_of(entry) -> str:
    if not isinstance(entry, dict):
        return ""
    hc = entry.get("healthcheck")
    if isinstance(hc, dict):
        test = hc.get("test")
        if isinstance(test, (list, tuple)):
            return " ".join(str(t) for t in test)
        if test:
            return str(test)
    return ""


def _services_from_yaml_doc(doc, source: str, out: dict[str, dict]) -> None:
    """Merge a parsed YAML doc's `services:` blocks into `out` (first kind wins)."""
    if not isinstance(doc, dict):
        return
    blocks = []
    if isinstance(doc.get("services"), dict):
        blocks.append(doc["services"])           # compose top-level
    for job in (doc.get("jobs") or {}).values() if isinstance(doc.get("jobs"), dict) else []:
        if isinstance(job, dict) and isinstance(job.get("services"), dict):
            blocks.append(job["services"])       # GitHub Actions job.services
    for block in blocks:
        for svc_name, entry in block.items():
            entry = entry if isinstance(entry, dict) else {}
            image = entry.get("image")
            kind = _kind_of(svc_name, image)
            if kind and kind not in out:
                out[kind] = {"image": image or "", "port": _port_of(entry), "healthcheck": _healthcheck_of(entry), "source": source}


def _load_yaml(path: str):
    if yaml is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None


def scan_compose_services(repo_path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fname in os.listdir(repo_path) if os.path.isdir(repo_path) else []:
        low = fname.lower()
        if (low.startswith("docker-compose") or low.startswith("compose.")) and \
                low.endswith((".yml", ".yaml")):
            doc = _load_yaml(os.path.join(repo_path, fname))
            _services_from_yaml_doc(doc, f"docker-compose: {fname}", out)
    return out


def scan_ci_services(repo_path: str) -> tuple[dict[str, dict], bool]:
    out: dict[str, dict] = {}
    present = False
    wf_dir = os.path.join(repo_path, ".github", "workflows")
    if not os.path.isdir(wf_dir):
        return out, present
    for fname in os.listdir(wf_dir):
        if not fname.lower().endswith((".yml", ".yaml")):
            continue
        doc = _load_yaml(os.path.join(wf_dir, fname))
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), dict):
            for job in doc["jobs"].values():
                if isinstance(job, dict) and isinstance(job.get("services"), dict) and job["services"]:
                    present = True
        _services_from_yaml_doc(doc, f".github/workflows/{fname}", out)
    return out, present


# Ordered (signature regex -> kind). First match wins.
_ERROR_SIGNATURES: tuple[tuple[object, str], ...] = (
    (_re.compile(r"could not connect to server|psycopg2\.OperationalError|connection to server at", _re.I), "postgres"),
    (_re.compile(r"redis(\.exceptions)?\.ConnectionError|Error \d+ connecting to .*redis", _re.I), "redis"),
    (_re.compile(r"pymongo\.errors\.(ServerSelectionTimeoutError|ConnectionFailure)", _re.I), "mongo"),
    (_re.compile(r"amqp|pika\.exceptions|AMQPConnectionError|rabbitmq", _re.I), "rabbitmq"),
    (_re.compile(r"OperationalError.*MySQL|Can't connect to MySQL server", _re.I), "mysql"),
)


def classify_service_error(text: str) -> str | None:
    """Map a connection-failure log to a service kind (failure-driven primitive)."""
    if not text:
        return None
    for pattern, kind in _ERROR_SIGNATURES:
        if pattern.search(text):
            return kind
    return None


def service_db_from_url(value: str) -> str | None:
    """The database name from a service URL path (``postgres://h/appdb`` -> ``appdb``)."""
    if not value or "://" not in value:
        return None
    try:
        path = urlparse(value).path
    except ValueError:
        return None
    name = (path or "").lstrip("/").strip()
    return name or None


def service_bind_url(scheme: str, port: int, db: str) -> str:
    """Uniform in-image Postgres URL (Option B): our creds + loopback host, app's scheme+db.

    The app's original scheme (incl. dialect suffix like ``postgresql+psycopg2``) is
    preserved so SQLAlchemy's driver selection is unchanged; only host/credentials are
    rewritten to the in-image instance configured by the binding obligation.
    """
    return f"{scheme}://postgres:postgres@127.0.0.1:{port}/{db}"
