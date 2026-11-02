"""Stand-in for rpicam-vid / rpicam-still in tests (no camera here).

  python fake_camera.py vid [--framerate N] ... -o -     JPEG frames on stdout until killed
  python fake_camera.py still ... -o out.jpg             one JPEG written to the -o path
  FAKE_CAMERA_DIE_AFTER=N: the vid mode exits by itself after N frames (a dying camera)
"""
import base64
import os
import sys
import time

JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywtQFdBRkxOUlNSMj5a"
    "YVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0//wAAR"
    "CAAYACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEG"
    "E1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWG"
    "h4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEB"
    "AQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYk"
    "NOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0"
    "tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCnRRRXhH0oUUUUAFFFFABRRRQB/9k=")


def main(argv):
    mode, args = argv[0], argv[1:]
    out = args[args.index("-o") + 1]
    if mode == "still":
        with open(out, "wb") as f:
            f.write(JPEG)
        return 0
    fps = float(args[args.index("--framerate") + 1]) if "--framerate" in args else 15.0
    die = int(os.environ.get("FAKE_CAMERA_DIE_AFTER", "0"))
    n = 0
    w = sys.stdout.buffer
    while True:
        # split writes across frames, the way a pipe delivers them
        w.write(JPEG[:100]); w.flush()
        w.write(JPEG[100:]); w.flush()
        n += 1
        if die and n >= die:
            sys.stderr.write("ERROR: *** fake camera died ***\n")
            return 1
        time.sleep(1.0 / fps)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
