"""Download one benchmark source into a new $T_DIG_DATA/downloads/<kind>; no resume.

ZIP members are CRC-checked on extraction and must be relative paths without `..`.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "coco-annotations": (
        "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_trainval2017.zip",
    ),
    "coco-images": (
        "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
        "val2017.zip",
    ),
    "fid-inception": (
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt",
        "inception-2015-12-05.pt",
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=sorted(SOURCES))
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    url, name = SOURCES[args.kind]
    folder = Path(os.environ.get("T_DIG_DATA", ROOT / "data")) / "downloads" / args.kind
    if not args.execute:
        print(
            json.dumps(
                {"url": url, "new_output_directory": str(folder), "execute": False},
                indent=2,
            )
        )
        return
    folder.mkdir(parents=True, exist_ok=False)
    part = folder / (name + ".part")
    sha = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=90) as response, part.open("wb") as out:
        while True:
            data = response.read(8 * 1024 * 1024)
            if not data:
                break
            out.write(data)
            sha.update(data)
    file = folder / name
    part.rename(file)
    extracted = []
    if file.suffix == ".zip":
        with zipfile.ZipFile(file) as archive:
            if args.kind == "coco-annotations":
                members = ["annotations/captions_val2017.json"]
            else:
                members = [n for n in archive.namelist() if n.endswith(".jpg")]
            if args.kind == "coco-images" and len(members) != 5000:
                raise ValueError("Expected 5,000 validation JPEGs")
            for member in members:
                relative = Path(member)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive member")
                destination = folder / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(member)
    (folder / "receipt.json").write_text(
        json.dumps(
            {
                "url": url,
                "file": name,
                "sha256": sha.hexdigest(),
                "bytes": file.stat().st_size,
                "extracted": extracted,
            },
            indent=2,
        )
        + "\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
