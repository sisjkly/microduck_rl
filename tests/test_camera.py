"""Camera: MJPEG reader, photos, MOV recordings, dashboard endpoints (with a fake rpicam)."""

import json
import os
import struct
import sys
import time
from pathlib import Path

import pytest

from duck_runtime.camera import Camera, MovWriter, _jpeg_size

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fake_camera import JPEG  # noqa: E402

FAKE = f"{sys.executable} {HERE / 'fake_camera.py'}"


def make_cam(tmp_path, **kw):
    return Camera(tmp_path / "media", fps=kw.pop("fps", 30), tuning=None, nice=0,
                  vid_cmd=f"{FAKE} vid", still_cmd=f"{FAKE} still", **kw)


def eventually(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def atoms(buf, start=0, end=None):
    """Top-level atoms of a QuickTime byte string: {kind: (offset, payload)}."""
    out, i, end = {}, start, len(buf) if end is None else end
    while i < end:
        size, kind = struct.unpack(">I4s", buf[i:i + 8])
        out.setdefault(kind.decode(), []).append(buf[i + 8:i + size])
        i += size
    return out


def find(buf, path):
    node = buf
    for kind in path:
        node = atoms(node)[kind][0]
    return node


def test_jpeg_size_and_mov_tables(tmp_path):
    assert _jpeg_size(JPEG) == (32, 24)
    w = MovWriter(tmp_path / "a.mov")
    stamps = [0.0, 0.066, 0.133, 0.333, 0.4]                    # one 200 ms gap (a dropped frame)
    for t in stamps:
        w.add(JPEG, t)
    w.close()
    data = (tmp_path / "a.mov").read_bytes()
    top = atoms(data)
    assert list(top) == ["ftyp", "mdat", "moov"]
    assert len(top["mdat"][0]) == 5 * len(JPEG)
    stbl = find(top["moov"][0], ["trak", "mdia", "minf", "stbl"])
    stsz = atoms(stbl)["stsz"][0]
    assert struct.unpack(">III", stsz[:12])[2] == 5
    stts = atoms(stbl)["stts"][0]
    n = struct.unpack(">I", stts[4:8])[0]
    runs = [struct.unpack(">II", stts[8 + 8 * k:16 + 8 * k]) for k in range(n)]
    durs = [d for c, d in runs for _ in range(c)]
    assert durs[:4] == [66, 67, 200, 67] and len(durs) == 5     # real timestamps, not a fixed rate
    stco = atoms(stbl)["stco"][0]
    first = struct.unpack(">I", stco[8:12])[0]
    assert data[first:first + 2] == b"\xff\xd8"                 # offsets point at the frames
    tkhd = find(top["moov"][0], ["trak", "tkhd"])
    assert struct.unpack(">II", tkhd[-8:]) == (32 << 16, 24 << 16)


def test_mov_decodes_with_ffmpeg(tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    import subprocess
    w = MovWriter(tmp_path / "b.mov")
    for k in range(20):
        w.add(JPEG, k / 15)
    w.close()
    r = subprocess.run([ffmpeg, "-v", "error", "-i", str(tmp_path / "b.mov"), "-f", "null", "-"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and not r.stderr.strip()


def test_live_frames_then_off(tmp_path):
    cam = make_cam(tmp_path)
    try:
        st = cam.set_live(True)
        assert st["live"] and st["running"]
        frame = cam.latest(timeout=5)
        assert frame == JPEG                                     # reassembled across split writes
        seq, f2 = cam.wait_frame(cam.frame_seq, timeout=3)
        assert f2 == JPEG
        assert eventually(lambda: (cam.status()["fps"] or 0) > 10)
        cam.set_live(False)
        assert not cam.status()["running"] and cam.latest(timeout=0.1) is None
        assert cam.status()["fps"] is None
    finally:
        cam.close()


def test_photo_full_res_pauses_and_resumes_the_stream(tmp_path):
    cam = make_cam(tmp_path, still_size=(1640, 1232))
    try:
        r = cam.photo()                                          # camera off: a still
        assert r["ok"] and r["from"] == "still" and r["size"] == [1640, 1232]
        assert (cam.media_dir / r["file"]).read_bytes() == JPEG
        cam.set_live(True)
        assert cam.latest(timeout=5)
        time.sleep(1.05)                                         # file names have 1 s resolution
        r2 = cam.photo()
        assert r2["ok"] and r2["from"] == "still"
        assert cam.status()["running"] and cam.latest(timeout=5)  # live view back
    finally:
        cam.close()


def test_recording_runs_the_camera_and_writes_a_mov(tmp_path):
    cam = make_cam(tmp_path)
    try:
        r = cam.start_recording()
        assert r["ok"] and r["file"].endswith(".mov") and cam.status()["running"]
        assert not cam.status()["live"]
        assert eventually(lambda: (cam.status()["recording"] or {}).get("frames", 0) >= 10)
        assert cam.list_media() == []                            # not listed while being written
        p = cam.photo()                                          # during a recording: a stream frame
        assert p["ok"] and p["from"].startswith("stream")
        done = cam.stop_recording()
        assert done["frames"] >= 10 and done["seconds"] > 0
        assert not cam.status()["running"]                       # it ran only for the recording
        names = {m["name"]: m["kind"] for m in cam.list_media()}
        assert names[done["file"]] == "video" and names[p["file"]] == "photo"
        data = (cam.media_dir / done["file"]).read_bytes()
        assert list(atoms(data)) == ["ftyp", "mdat", "moov"]
    finally:
        cam.close()


def test_camera_dying_is_reported_and_the_recording_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA_DIE_AFTER", "8")
    cam = make_cam(tmp_path)
    try:
        cam.set_live(True)
        cam.start_recording()
        assert eventually(lambda: not cam.status()["running"])
        st = cam.status()
        assert "exited" in st["error"] and "fake camera died" in st["error"] and not st["live"]
        assert st["recording"] is None
        vids = [m for m in cam.list_media() if m["kind"] == "video"]
        assert len(vids) == 1 and vids[0]["bytes"] > 8 * len(JPEG)
    finally:
        cam.close()


def test_photo_failure_is_an_error_not_a_crash(tmp_path):
    cam = Camera(tmp_path / "m", tuning=None, nice=0, vid_cmd=f"{FAKE} vid", still_cmd="false")
    r = cam.photo()
    assert not r["ok"] and "photo failed" in r["error"]
    cam.close()


# -- dashboard ---------------------------------------------------------------------------------

@pytest.fixture()
def camdash(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    import numpy as np
    from duck_runtime.controller import Controller, Tuning
    from duck_runtime.dashboard import Dashboard, make_handler
    from duck_runtime.policy_set import bind_keys, load_set
    from duck_runtime.run import LoadedSet, SetBank
    from duck_runtime.runner import Runner
    from test_policy_set import UprightIO, make_set
    (tmp_path / "set").mkdir()
    loader, _ = make_set(tmp_path / "set")
    pset = load_set(str(tmp_path / "set"), loader=loader)
    io = UprightIO()
    io.voltages = np.full(15, 7.4)
    r = Runner(io, Controller(pset.walk, Tuning.official()), ramp_s=0.02, policies=pset)
    bank = SetBank([LoadedSet("set (xl330)", "set", pset, bind_keys(pset.skills), Tuning.official())])
    cam = make_cam(tmp_path)
    d = Dashboard(r, io, bank, (0.0, 0.0, 0.0, 0.0), backend="test", log_dir=str(tmp_path / "logs"),
                  token="t0k", realtime=True, camera=cam)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(d))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    loop = threading.Thread(target=d.run, daemon=True)
    loop.start()
    d.port = server.server_address[1]
    yield d
    d.shutdown()
    cam.close()
    loop.join(2)
    server.shutdown()


def _req(d, method, path, body=None, token="t0k"):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", d.port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if token and method == "POST":
        headers["X-Duck-Token"] = token
    c.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    return c, c.getresponse()


def test_dashboard_camera_flow(camdash):
    d = camdash
    c, r = _req(d, "GET", "/api/info")
    info = json.loads(r.read())
    assert info["camera"]["size"] == [640, 480]
    c, r = _req(d, "GET", "/camera/stream")                      # no token: refused (it shows the room)
    assert r.status == 403
    c, r = _req(d, "GET", "/camera/stream?token=t0k")            # camera off
    assert r.status == 503
    c, r = _req(d, "POST", "/api/camera", {"action": "live_on"})
    assert json.loads(r.read())["live"]
    c, r = _req(d, "GET", "/camera/stream?token=t0k")
    assert r.status == 200 and r.getheader("Content-Type").startswith("multipart/x-mixed-replace")
    head = r.fp.readline() + r.fp.readline() + r.fp.readline() + r.fp.readline()
    assert b"--duckframe" in head and b"image/jpeg" in head
    n = int([ln for ln in head.split(b"\r\n") if ln.startswith(b"Content-Length")][0].split(b":")[1])
    assert r.fp.read(n) == JPEG
    c.close()
    c, r = _req(d, "GET", "/camera/frame.jpg?token=t0k")
    assert r.status == 200 and r.read() == JPEG
    c, r = _req(d, "POST", "/api/camera", {"action": "record_start"})
    assert json.loads(r.read())["ok"]
    time.sleep(0.6)
    c, r = _req(d, "POST", "/api/camera", {"action": "record_stop"})
    rec = json.loads(r.read())
    assert rec["frames"] > 3
    c, r = _req(d, "POST", "/api/camera", {"action": "live_off"})
    c, r = _req(d, "GET", "/api/media?token=t0k")
    media = json.loads(r.read())
    assert media[0]["name"] == rec["file"] and media[0]["kind"] == "video"
    c, r = _req(d, "GET", f"/media/{rec['file']}?token=t0k")
    body = r.read()
    assert r.status == 200 and r.getheader("Content-Type") == "video/quicktime" and body[4:8] == b"ftyp"
    assert "attachment" in r.getheader("Content-Disposition")
    for bad in ("/media/../secret.mov?token=t0k", "/media/duck_x.mov?token=t0k", f"/media/{rec['file']}"):
        c, r = _req(d, "GET", bad)
        assert r.status in (403, 404)
    c, r = _req(d, "POST", "/api/camera", {"action": "zoom"})
    assert json.loads(r.read())["ok"] is False
    c, r = _req(d, "GET", "/api/events?since=0")
    text = " ".join(e["message"] for e in json.loads(r.read()))
    assert "video saved" in text and "recording started" in text


# -- rotation ----------------------------------------------------------------------------------
def _ffmpeg():
    return pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()


def _rgb(ffmpeg, args, w, h):
    import subprocess
    import numpy as np
    out = subprocess.run([ffmpeg, "-v", "error", *args, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, 3).astype(int)


def test_exif_orientation_added_patched_and_replaced():
    from duck_runtime.camera import jpeg_orientation, jpeg_set_orientation, _exif_orientation_app1
    assert jpeg_orientation(JPEG) is None
    a = jpeg_set_orientation(JPEG, 8)
    assert jpeg_orientation(a) == 8 and _jpeg_size(a) == _jpeg_size(JPEG)
    b = jpeg_set_orientation(a, 6)                      # patched in place: same length
    assert jpeg_orientation(b) == 6 and len(b) == len(a)
    # an Exif block without the tag (as a camera might write) is replaced
    tiff = b"II\x2a\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x010F, 2, 4, 0x00756f79) + b"\0\0\0\0"
    body = b"Exif\x00\x00" + tiff
    c = JPEG[:2] + b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body + JPEG[2:]
    assert jpeg_orientation(c) is None
    d = jpeg_set_orientation(c, 8)
    assert jpeg_orientation(d) == 8 and d.count(b"Exif\x00\x00") == 1
    assert _exif_orientation_app1(8)[:2] == b"\xff\xe1"


def test_exif_orientation_is_what_pillow_shows():
    pil = pytest.importorskip("PIL.Image")
    from PIL import ImageOps
    import io as _io
    from duck_runtime.camera import jpeg_set_orientation
    im = ImageOps.exif_transpose(pil.open(_io.BytesIO(jpeg_set_orientation(JPEG, 8))))
    assert im.size == (24, 32)


@pytest.mark.parametrize("rotate", [90, 270])
def test_mov_rotation_matrix_turns_the_picture_ccw(tmp_path, rotate):
    """A real asymmetric frame through MovWriter(rotate) and ffmpeg's autorotate == numpy rot90."""
    import subprocess
    import numpy as np
    ffmpeg = _ffmpeg()
    src = tmp_path / "f.jpg"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=1", "-frames:v", "1",
                    "-q:v", "2", str(src)], check=True)
    frame = src.read_bytes()
    w = MovWriter(tmp_path / "r.mov", rotate=rotate)
    for k in range(3):
        w.add(frame, k / 15)
    w.close()
    orig = _rgb(ffmpeg, ["-i", str(src)], 64, 48)
    shown = _rgb(ffmpeg, ["-i", str(tmp_path / "r.mov")], 48, 64)     # autorotate is on by default
    want = np.rot90(orig, k=1 if rotate == 90 else -1)                 # np.rot90 k=1: counterclockwise
    assert shown.shape == want.shape and np.abs(shown - want).mean() < 12
    tk = find((tmp_path / "r.mov").read_bytes(), ["moov", "trak", "tkhd"])
    assert struct.unpack(">II", tk[-8:]) == (48 << 16, 64 << 16)


def test_camera_rotate_90_photo_and_status(tmp_path):
    from duck_runtime.camera import jpeg_orientation
    cam = make_cam(tmp_path, rotate=90)
    st = cam.status()
    assert st["rotate"] == 90 and st["view_rotate"] == 90
    r = cam.photo()
    assert r["ok"] and r["size"] == [1232, 1640]
    assert jpeg_orientation((tmp_path / "media" / r["file"]).read_bytes()) == 8
    cam180 = make_cam(tmp_path, rotate=180)
    assert "--rotation" in cam180._tuning_args() and cam180.status()["view_rotate"] == 0
    with pytest.raises(ValueError):
        make_cam(tmp_path, rotate=45)
    cam.close(); cam180.close()
