#!/usr/bin/env python3
"""Check release integrity and repository manifests without Docker, network, or API calls."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    checksums = ROOT / 'SHA256SUMS'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            expected, name = line.split('  ', 1)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected_size in [('graph2env200.json', 200)]:
        rows = json.loads((ROOT / 'datasets' / name).read_text())
        assert len(rows) == expected_size, name
        assert len({r['full_name'] for r in rows}) == len(rows), name
        for row in rows:
            assert re.fullmatch(r'[^/\s]+/[^/\s]+', row['full_name']), row
            assert re.fullmatch(r'[0-9a-f]{40}', row['commit']), row
        print(f'{name}: {len(rows)} pinned repositories')
    print('Release integrity and manifest checks passed. No benchmark was executed.')


if __name__ == '__main__':
    main()
