"""Dashboard: HTTP API + SSE around the real Runner, on a fake body and fake networks."""

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from duck_runtime.controller import Controller, Tuning
from duck_runtime.dashboard import Dashboard, make_handler
from duck_runtime.policy_set import bind_keys, load_set
from duck_runtime.runner import Runner, State

from test_policy_set import UprightIO, make_set

TOKEN = "t0k"


@pytest.fixture()
def dash(tmp_path):
    (tmp_path / "set").mkdir()
    loader, nets = make_set(tmp_path / "set")
    pset = load_set(str(tmp_path / "set"), loader=loader)
    io = UprightIO()
    io.voltages = np.full(15, 7.4)
    io.currents = np.full(15, 100.0)
    io.temperatures = np.full(15, 40.0)
    r = Runner(io, Controller(pset.walk, Tuning.official()), ramp_s=0.02, policies=pset)
    d = Dashboard(r, io, pset, bind_keys(pset.skills), (0.0, 0.0, 0.0, 0.0), backend="test",
                  log_dir=str(tmp_path / "logs"), token=TOKEN, realtime=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(d))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    loop = threading.Thread(target=d.run, daemon=True)
    loop.start()
    d.port = server.server_address[1]
    d.nets = nets
    yield d
    d.shutdown()
    loop.join(2)
    server.shutdown()


def call(d, method, path, body=None, token=TOKEN, conn=None):
    c = conn or http.client.HTTPConnection("127.0.0.1", d.port, timeout=5)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Duck-Token"] = token
    c.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    r = c.getresponse()
    data = r.read()
    return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)


def frame(d, pred=lambda f: True):
    """The first streamed frame that satisfies ``pred`` (frames come at 10 Hz)."""
    c = http.client.HTTPConnection("127.0.0.1", d.port, timeout=5)
    c.request("GET", "/api/stream")
    r = c.getresponse()
    while True:
        line = r.fp.readline()
        if line.startswith(b"data: "):
            f = json.loads(line[6:])
            if pred(f):
                c.close()
                return f


def wait_for(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_info_lists_joints_policy_and_skills(dash):
    st, info = call(dash, "GET", "/api/info")
    assert st == 200 and len(info["joints"]) == 15 and info["joints"][9]["mouth"]
    assert [s["name"] for s in info["policy"]["skills"]] == ["ground_pick", "roulade", "kick_left", "kick_right"]
    assert info["policy"]["sitstand"] and info["token_required"]


def test_controls_need_the_token_and_the_connection_survives_a_refusal(dash):
    c = http.client.HTTPConnection("127.0.0.1", dash.port, timeout=5)
    st, body = call(dash, "POST", "/api/action", {"action": "enable"}, token=None, conn=c)
    assert st == 403 and not body["ok"]
    st, body = call(dash, "POST", "/api/action", {"action": "enable"}, conn=c)   # same keep-alive connection
    assert st == 200 and body["ok"]


def test_enable_start_walk_and_stream(dash):
    assert call(dash, "POST", "/api/action", {"action": "enable"})[1]["ok"]
    assert wait_for(lambda: dash.runner.state is State.HOLD)
    assert call(dash, "POST", "/api/action", {"action": "start"})[1]["ok"]
    call(dash, "POST", "/api/command", {"vx": 0.25, "wz": -0.4, "head": [0, 10, 0, 0]})
    assert wait_for(lambda: dash.runner.controller.last_obs is not None and dash.runner.controller.last_obs[48] == pytest.approx(0.25))
    f = frame(dash, lambda f: f["state"] == "run")
    assert f["mode"] == "walk"
    assert f["cmd"]["vx"] == 0.25 and f["supply"]["min"] == 7.4 and f["supply"]["current_a"] == pytest.approx(1.4)
    assert len(f["joints"]["pos"]) == 15 and f["joints"]["temp"][0] == 40
    assert f["imu"]["pitch"] == 0.0 and f["loop"]["hz"] > 20
    assert dash.runner.controller.last_obs[52] == pytest.approx(np.radians(10), abs=1e-6)   # head pitch command


def test_command_times_out_without_the_heartbeat(dash):
    call(dash, "POST", "/api/action", {"action": "enable"})
    wait_for(lambda: dash.runner.state is State.HOLD)
    call(dash, "POST", "/api/action", {"action": "start"})
    call(dash, "POST", "/api/command", {"vx": 0.3})
    assert wait_for(lambda: dash.runner.controller.last_obs[48] == pytest.approx(0.3))
    assert wait_for(lambda: dash.runner.controller.last_obs[48] == 0.0, timeout=2.0)   # page closed -> stop


def test_limits_and_bad_requests(dash):
    st, body = call(dash, "POST", "/api/command", {"vx": 9, "wz": -9})
    assert body["cmd"]["vx"] == 0.3 and body["cmd"]["wz"] == -1.0
    assert call(dash, "POST", "/api/command", {"head": [1, 2]})[0] == 400
    assert call(dash, "POST", "/api/action", {"action": "selfdestruct"})[1]["ok"] is False
    assert call(dash, "GET", "/logs/..%2Fsecret.npz")[0] == 404


def test_skill_sit_settings_and_events(dash):
    call(dash, "POST", "/api/action", {"action": "enable"})
    wait_for(lambda: dash.runner.state is State.HOLD)
    call(dash, "POST", "/api/action", {"action": "start"})
    st, body = call(dash, "POST", "/api/action", {"action": "skill", "name": "kick_left"})
    assert body["ok"] and wait_for(lambda: len(dash.nets["ball_kick_left"].seen) > 0)
    st, body = call(dash, "POST", "/api/action", {"action": "sit"})
    assert not body["ok"] and "still running" in body["error"]
    assert call(dash, "POST", "/api/settings", {"on_fall": "keep", "getup_timeout_s": 3})[1]["ok"]
    assert dash.runner.on_fall == "keep" and dash.runner.getup_timeout_s == 3.0
    st, ev = call(dash, "GET", "/api/events?since=0")
    text = " ".join(e["message"] for e in ev)
    assert "kick_left" in text and "sit/stand refused" in text and "state hold -> run" in text


def test_record_save_list_download(dash):
    call(dash, "POST", "/api/action", {"action": "enable"})
    assert call(dash, "POST", "/api/record", {"on": True})[1]["ok"]
    time.sleep(0.3)
    st, body = call(dash, "POST", "/api/record", {"on": False})
    name = body["file"]
    assert name and name.endswith(".npz")
    st, logs = call(dash, "GET", "/api/logs")
    assert logs[0]["name"] == name
    st, data = call(dash, "GET", f"/logs/{name}")
    assert st == 200 and data[:2] == b"PK"                     # an npz is a zip
    import io
    z = np.load(io.BytesIO(data), allow_pickle=True)
    assert {"t", "state", "mode", "positions", "voltages", "currents"} <= set(z.files) and len(z["t"]) > 5


def test_page_is_served(dash):
    st, html = call(dash, "GET", "/")
    assert st == 200 and b"Microduck" in html and b"/api/stream" in html
