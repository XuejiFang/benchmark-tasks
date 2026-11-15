"""Write observer frames as a real-time H.264 trial recording."""
from contextlib import contextmanager
import subprocess

import numpy as np


@contextmanager
def record_trial(output, enabled, *, width=640, height=360, fps=20):
    if not enabled:
        yield None
        return

    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-nostdin',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast',
               '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output/'trial.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    had_error = False

    def write(image):
        frame = np.ascontiguousarray(image, dtype=np.uint8)
        if frame.shape != (height, width, 3):
            raise ValueError(f'Unexpected video frame dimensions: {frame.shape}')
        process.stdin.write(frame.tobytes())

    try:
        yield write
    except BaseException:
        had_error = True
        raise
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = -1
        if code and not had_error:
            raise RuntimeError(f'Video encoder exited with status {code}')
