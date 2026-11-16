"""Download and verify the fixed simulation assets for this task."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/XuejiFang/benchmark-tasks/c3cdb951bb00dd5f4c8338988a57e24da78c47fb/tasks/pack-gym-bag/'
ASSET_ROOT = ROOT / 'assets'


def fetch(entry):
    asset, expected = entry
    destination = ASSET_ROOT / asset
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    source_path = destination.relative_to(ROOT).as_posix()
    with urlopen(SOURCE + quote(source_path, safe='/'), timeout=120) as response:
        contents = response.read()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise RuntimeError(f'SHA-256 mismatch for {source_path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(contents)
    Path(temporary.name).replace(destination)


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'asset_manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(fetch, manifest.items()):
            pass
    print(f'{len(manifest)} simulation assets verified.')
