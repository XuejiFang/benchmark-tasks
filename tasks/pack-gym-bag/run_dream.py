"""T01 model rollout using the unmodified official inspect_robots_dream policy.

Only RGB, measured joint state and the explicit instruction reach Dream.
Privileged physics measurements are recorded separately for scoring.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

INSTRUCTION = (
    'Pack the teal capped water bottle on the coffee-table tray, both blue-and-white '
    'running shoes on the floor, the black wallet on the sofa, and the over-ear '
    'headphones on the sofa into the open olive-green gym bag. Leave all five objects '
    'fully inside the bag, let go of them, and move both grippers clear.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 33 <= args.max_steps <= 3000:
        parser.error('max-steps must be 33..3000')
    if not args.smoke and args.max_steps != 3000:
        parser.error('Official baseline uses the fixed 3000-step run limit; shortened tests need --smoke')
    args.output.mkdir(parents=True, exist_ok=False)
    from inspect_robots_dream.policy import DreamPolicy
    from inspect_robots_dream.session import DreamSession, Episode
    import importlib.metadata
    session = DreamSession()
    session.check(Episode())
    session.close()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    import carb.settings
    settings = carb.settings.get_settings()
    settings.set('/rtx/post/aa/op', 3)
    settings.set('/rtx/post/dlss/execMode', 2)
    from runtime import install
    install(960, 528)
    from yam_arena import inspect_embodiment
    inspect_embodiment._APP = app
    from yam_arena.inspect_embodiment import YamArenaEmbodiment
    from inspect_robots import Scene
    from packing_observer import PackingObserver, array
    from layout import ASSETS, ITEMS
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    embodiment = YamArenaEmbodiment(headless=True, cam_height=176, cam_width=320, control_hz=20,
                                    task=str(Path(__file__).with_name('task.py')), object_jitter=0)
    policy = DreamPolicy(timeout_s=120, record=False)
    policy.bind(embodiment.info)
    scene = Scene(id='gym-packing-official-baseline', instruction=INSTRUCTION)
    metadata = {
        'kind': 'connection_smoke_not_official_score' if args.smoke else 'official_dream_baseline',
        'policy': 'inspect_robots_dream.policy.DreamPolicy', 'model': policy.info.checkpoint,
        'instruction': INSTRUCTION, 'seed': 0, 'max_steps': args.max_steps,
        'control_hz': 20, 'physics_hz': 240, 'observation_size': [320, 176],
        'camera_mapping': {'top_cam': 'overhead overview camera, rotated 90 degrees', 'left_cam': 'left wrist', 'right_cam': 'right wrist'},
        'memory_retrieval': False, 'privileged_state_to_policy': False,
        'prop_state_writes_after_reset': False,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('inspect-robots', 'inspect-robots-dream', 'dreamlabs-client')},
    }
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ('runtime.py', 'task.py', 'layout.py', 'geometry.py', 'packing_observer.py')]]
    import shutil
    for file in files:
        shutil.copy2(file, args.output / file.name)
    metadata['script_sha256'] = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    asset_files = [ASSETS / 'room/physics_room_v1.usda', ASSETS / 'room/duffel_ready.usda',
                   ASSETS / 'bag/duffel_open_curved.usda', *[ASSETS / 'items' / f'{name}.usd' for name in ITEMS]]
    metadata['asset_sha256'] = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in asset_files}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    videos = {}
    def video(name, width, height):
        proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '20', '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(args.output / f'{name}.mp4')], stdin=subprocess.PIPE)
        videos[name] = proc
        return proc
    started = time.monotonic()
    count = 0
    last = None
    failed = False
    log = (args.output / 'trace.jsonl').open('w')
    def policy_view(observation):
        import cv2
        overhead = array(embodiment._base.scene.sensors['overview_cam'].data.output['rgb'])[0, :, :, :3]
        observation.images['top_cam'] = np.ascontiguousarray(cv2.resize(overhead, (320, 176)), dtype=np.uint8)
        return observation
    try:
        policy.reset(scene)
        observation = policy_view(embodiment.reset(scene, seed=0))
        # The independent observer below supplies the physical score. The task-file
        # callback cannot express contact forces and is not used for this rollout.
        embodiment._task_step = lambda: SimpleNamespace(observation=embodiment._observation())
        observer = PackingObserver(embodiment._base)
        np.savez_compressed(args.output / 'scoring_vertices.npz', **observer.points)
        for name in ('top_cam', 'left_cam', 'right_cam'):
            Image.fromarray(observation.images[name]).save(args.output / f'initial_{name}.png')
            video(name, 320, 176)
        video('dream_run', 960, 528)
        for step in range(args.max_steps):
            before = observation
            chunk = policy.act(before)
            if len(chunk.actions) != 1:
                raise RuntimeError('Official adapter changed its one-action-per-step contract')
            action = chunk.actions[0]
            values = np.asarray(action.data, dtype=float)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise RuntimeError('Nonfinite or invalid action')
            result = embodiment.step(action)
            observation = policy_view(result.observation)
            last = observer.capture(values, step)
            last['policy_input_joint_pos'] = np.asarray(before.state['joint_pos']).tolist()
            last['policy_input_image_sha256'] = {name: hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
                                                 for name, img in before.images.items()}
            last['inference_latency_s'] = chunk.inference_latency_s
            log.write(json.dumps(last) + '\n')
            log.flush()
            for name in ('top_cam', 'left_cam', 'right_cam'):
                videos[name].stdin.write(before.images[name].tobytes())
            rgb = array(embodiment._base.scene['top_cam'].data.output['rgb'])[0, :, :, :3].astype(np.uint8)
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 960, 42), fill=(10, 10, 10))
            draw.text((9, 5), 'DREAM-1 API | PHYSICS SIMULATION', fill='white')
            draw.text((9, 24), f'{(step+1)/20:.2f}s | packing score {last["score"]:.3f}', fill='white')
            videos['dream_run'].stdin.write(np.asarray(img).tobytes())
            count += 1
            if step % 100 == 0:
                img.save(args.output / 'latest.png')
                print('STEP', count, 'SCORE', last['score'], 'API_REQUESTS', len(policy.transcript()['requests']), flush=True)
            if last['score'] == 1.:
                break
        report = {'status': 'completed', 'score': last['score'], 'success': last['score'] == 1.,
                  'steps': count, 'duration_s': count / 20, 'wall_time_s': time.monotonic() - started,
                  'smoke_only': args.smoke, 'model': policy.info.checkpoint}
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print('RESULT', json.dumps(report), flush=True)
    except Exception as error:
        failed = True
        # Errors are infrastructure failures, not a valid zero-score model run.
        report = {'status': 'error', 'error_type': type(error).__name__, 'steps': count,
                  'last_observed_score': None if last is None else last['score']}
        (args.output / 'failure.json').write_text(json.dumps(report, indent=2) + '\n')
        print('FAILURE', json.dumps(report), flush=True)
    finally:
        (args.output / 'dream_transcript.json').write_text(json.dumps(policy.transcript(), indent=2) + '\n')
        policy.close()
        log.close()
        for proc in videos.values():
            proc.stdin.close()
            if proc.wait(timeout=30) != 0:
                failed = True
                (args.output / 'video_failure.json').write_text('{"status":"encoder_failed"}\n')
        embodiment.close()
    # Isaac's exit handler can force exit zero; callers must also inspect artifacts.
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
