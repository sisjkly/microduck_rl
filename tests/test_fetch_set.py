"""fetch_set: pinned extraction of kissqy's networks from his robotd package."""

import io
import json
import os
import tarfile

import pytest

from duck_runtime.fetch_set import KISSQY_SET, extract_kissqy, kissqy_manifest

REAL_TAR = os.environ.get("DUCK_KISSQY_TAR", "/home/claude/ext/Microduck-kissqy/microduck-hd1910/firmware/console/updates/"
                          "Microduck-Robotd-FT6-R17.tar.gz")   # a local copy of his package, if any


def _tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_manifest_lists_every_pinned_network_once():
    m = kissqy_manifest()
    assert m["schema_version"] == 2 and m["robot"]["servos"] == "hd1910"
    files = [e["file"] for e in m["policies"]]
    assert files == [f"kissqy_{n}.onnx" for n in KISSQY_SET] and len(set(files)) == 6


def test_tampered_or_missing_networks_are_refused(tmp_path):
    sha = KISSQY_SET["velstand"][0]
    bad = _tar({f"Microduck-Robotd-FT6-R17/models/{sha[:32]}/policy.onnx": b"not the model"})
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        extract_kissqy(bad, str(tmp_path))
    with pytest.raises(SystemExit, match="not in the package"):
        extract_kissqy(_tar({"x/readme.txt": b"hi"}), str(tmp_path))
    assert not os.path.exists(tmp_path / "manifest.json")


@pytest.mark.skipif(not os.path.exists(REAL_TAR), reason="kissqy package not downloaded")
def test_real_package_extracts_and_loads(tmp_path):
    from duck_runtime.controller import Policy
    from duck_runtime.policy_set import load_set
    with open(REAL_TAR, "rb") as f:
        extract_kissqy(f.read(), str(tmp_path))
    s = load_set(str(tmp_path), loader=Policy)
    assert len(s.skills) == 4 and s.sitstand is not None
    assert json.load(open(tmp_path / "manifest.json"))["policies"][0]["lowpass"] == {"head": 0.5, "legs": 0.7}
