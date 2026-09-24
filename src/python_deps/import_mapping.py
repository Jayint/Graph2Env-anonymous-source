from __future__ import annotations

import re
from dataclasses import dataclass


CURATED_IMPORT_TO_PACKAGE = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "corsheaders": "django-cors-headers",
    "dateutil": "python-dateutil",
    "django_filters": "django-filter",
    "factory": "factory-boy",
    "fitz": "PyMuPDF",
    "fpdf": "fpdf2",
    "lxml": "lxml",
    "googleapiclient": "google-api-python-client",
    "mysqldb": "mysqlclient",
    "openssl": "pyOpenSSL",
    "nacl": "PyNaCl",
    "pil": "Pillow",
    "pdfminer": "pdfminer.six",
    "psycopg2": "psycopg2",
    "pkg_resources": "setuptools",
    "sklearn": "scikit-learn",
    "socketio": "python-socketio",
    "yaml": "PyYAML",
}

SOURCE_LABELS = {
    "collision_table": "Collision Table",
    "declared_metadata": "Graph2Env declared metadata extension",
    "direct_name": "Lookup Name Variants",
}


@dataclass(frozen=True)
class MappingResult:
    import_name: str
    package_name: str
    source: str
    trust: str


def normalize_package_name(name: str) -> str:
    """Normalize a distribution name according to Python packaging conventions."""
    return re.sub(r"[-_.]+", "-", name).lower()


def top_level_import_name(import_name: str) -> str:
    return import_name.split(".", 1)[0]


def map_import_to_package(
    import_name: str,
    declared_package_names: set[str] | None = None,
) -> MappingResult:
    top_level = top_level_import_name(import_name)
    curated_key = top_level.lower()
    if curated_key in CURATED_IMPORT_TO_PACKAGE:
        return MappingResult(
            import_name=top_level,
            package_name=CURATED_IMPORT_TO_PACKAGE[curated_key],
            source="collision_table",
            trust="high",
        )

    declared_by_normalized_name = {
        normalize_package_name(name): name for name in declared_package_names or set()
    }
    normalized_import = normalize_package_name(top_level)
    if normalized_import in declared_by_normalized_name:
        return MappingResult(
            import_name=top_level,
            package_name=declared_by_normalized_name[normalized_import],
            source="declared_metadata",
            trust="medium",
        )

    return MappingResult(
        import_name=top_level,
        package_name=top_level,
        source="direct_name",
        trust="low",
    )


def mapping_source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())
