"""Curated `package -> service` table (tier-5 analogue of
``config_tables.PACKAGE_TO_CONFIG``).  A driver distribution implies a server it
talks to: ``psycopg2`` -> postgres, ``redis`` -> redis, ``pymongo`` -> mongo.

These are INFERRED signals (the suite may mock the DB), so callers must NOT turn
them into structural ``requires`` edges without corroborating evidence (design
§4).  ``celery``/``kombu`` map to a generic ``broker`` kind, not a specific one.
"""

from __future__ import annotations

from src.python_deps.import_mapping import normalize_package_name

# service kind -> (default image, default port)
SERVICE_DEFAULTS: dict[str, tuple[str, int]] = {
    "postgres": ("postgres:16", 5432),
    "mysql": ("mysql:8", 3306),
    "redis": ("redis:7", 6379),
    "mongo": ("mongo:7", 27017),
    "rabbitmq": ("rabbitmq:3", 5672),
    "broker": ("redis:7", 6379),       # generic broker; default to redis
    "elasticsearch": ("elasticsearch:8", 9200),
}

# Services that are safe and useful to provision inside the same evaluation
# container.  This is deliberately small: a recipe is an execution capability,
# not merely a service-name catalogue entry.  Callers still have to opt into
# service provisioning and certify the loopback check before the node can pass.
IN_IMAGE_SERVICE_RECIPES: dict[str, dict[str, str]] = {
    "redis": {
        "system_package": "redis-server",
        "start": "redis-server --daemonize yes",
        "check": (
            "python3 -c \"import socket; "
            "s=socket.create_connection(('127.0.0.1',6379),2); s.close()\""
        ),
    },
}

KNOWN_SERVICE_KINDS: frozenset[str] = frozenset(SERVICE_DEFAULTS)

# Concrete service kinds that can satisfy an abstract `broker` need (celery/kombu).
# Used to suppress the generic `broker` node when a concrete broker is already present.
BROKER_CAPABLE_KINDS: frozenset[str] = frozenset({"redis", "rabbitmq"})

PACKAGE_TO_SERVICE: dict[str, list[str]] = {
    "psycopg2": ["postgres"],
    "psycopg2-binary": ["postgres"],
    "psycopg": ["postgres"],
    "asyncpg": ["postgres"],
    "mysqlclient": ["mysql"],
    "pymysql": ["mysql"],
    "redis": ["redis"],
    "pymongo": ["mongo"],
    "mongoengine": ["mongo"],
    "pika": ["rabbitmq"],
    "kombu": ["broker"],
    "celery": ["broker"],
    "elasticsearch": ["elasticsearch"],
}

_NORMALIZED: dict[str, list[str]] = {
    normalize_package_name(name): kinds for name, kinds in PACKAGE_TO_SERVICE.items()
}


def services_for_package(name: str) -> list[str]:
    """Service kinds a distribution implies, or ``[]`` (fresh list)."""
    return list(_NORMALIZED.get(normalize_package_name(name), ()))


def service_defaults(kind: str) -> tuple[str, int]:
    """`(image, port)` for a known service kind; raises KeyError if unknown."""
    return SERVICE_DEFAULTS[kind]


def in_image_service_recipe(kind: str) -> dict[str, str] | None:
    """Return a fresh, curated in-image recipe for ``kind``, if supported."""
    recipe = IN_IMAGE_SERVICE_RECIPES.get(kind)
    return dict(recipe) if recipe is not None else None
