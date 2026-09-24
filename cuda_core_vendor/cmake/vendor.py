# Copyright 2026 Open Source Robotics Foundation, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Create a checksum-verified Cargo local registry from the lockfile."""

import argparse
import hashlib
import json
import os
import tarfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lockfile', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    registry = args.destination / 'registry'
    registry.mkdir(parents=True, exist_ok=True)
    packages = {}
    for package in tomllib.loads(args.lockfile.read_text())['package']:
        if 'source' not in package:
            continue
        if package['source'] != 'registry+https://github.com/rust-lang/crates.io-index':
            raise ValueError(f"unsupported crate source: {package['source']}")
        packages.setdefault(package['name'], []).append(package)

    def fetch(relative, url, valid):
        destination = registry / relative
        candidates = [destination]
        if args.cache:
            candidates.append(args.cache / relative)
        for candidate in candidates:
            if candidate.is_file():
                data = candidate.read_bytes()
                if valid(data):
                    break
        else:
            if os.environ.get('CARGO_NET_OFFLINE', '').lower() == 'true':
                raise RuntimeError(f'missing or invalid offline vendor file: {relative}')
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=60) as response:
                        data = response.read()
                    break
                except URLError:
                    if attempt == 2:
                        raise
            if not valid(data):
                raise ValueError(f'checksum or index mismatch: {url}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(destination)
        return data

    def vendor(item):
        name, versions = item
        if len(name) < 3:
            prefix = str(len(name))
        elif len(name) == 3:
            prefix = f'3/{name[0]}'
        else:
            prefix = f'{name[:2]}/{name[2:4]}'
        relative = Path('index') / prefix / name
        expected = {p['version']: p['checksum'] for p in versions}
        files = [relative]

        def index_matches(data):
            entries = {entry['vers']: entry['cksum']
                       for entry in map(json.loads, data.splitlines())}
            return all(entries.get(version) == checksum
                       for version, checksum in expected.items())

        data = fetch(relative, f'https://index.crates.io/{prefix}/{name}', index_matches)
        entries = [line for line in data.splitlines()
                   if json.loads(line)['vers'] in expected]
        temporary = (registry / relative).with_suffix('.tmp')
        temporary.write_bytes(b'\n'.join(entries) + b'\n')
        temporary.replace(registry / relative)
        for package in versions:
            filename = f"{name}-{package['version']}.crate"
            files.append(Path(filename))
            fetch(Path(filename), f'https://static.crates.io/crates/{name}/{filename}',
                  lambda data: hashlib.sha256(data).hexdigest() == package['checksum'])
            if name in ('cuda-core', 'cuda-bindings', 'cuda-core-derive', 'oxide-artifacts'):
                with tarfile.open(registry / filename) as archive:
                    archive.extractall(args.destination / 'sources', filter='data')
        return files

    with ThreadPoolExecutor(max_workers=8) as executor:
        files = {path for paths in executor.map(vendor, packages.items()) for path in paths}
    for path in registry.rglob('*'):
        if path.is_file() and path.relative_to(registry) not in files:
            path.unlink()


if __name__ == '__main__':
    main()
