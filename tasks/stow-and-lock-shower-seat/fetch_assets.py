"""Retrieve and verify the pinned ForgeCAD shower-seat asset."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
BASE = ('https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/'
        'b94ea770666b6c0f82e09a78e9fcb375b223e3b1/shower_seat/simready_usd/')

for name, digest in json.loads((ROOT/'asset_manifest.json').read_text()).items():
    path = ROOT/'assets'/name
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        continue
    with urlopen(BASE+quote(name, safe='/'), timeout=180) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError(f'SHA-256 mismatch for {name}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)
print('Shower-seat source assets verified.')
