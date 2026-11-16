"""Run the official Dream policy on an already constructed physical task."""

import hashlib
import importlib.metadata
import json
import time

import numpy as np


def run_dream(output, env, scene, observe, tick, audit, trace, action, max_steps, assets, *, early_stop_interval=1):
    from PIL import Image
    from inspect_robots_dream.policy import DreamPolicy

    policy = DreamPolicy(timeout_s=120,record=False)
    policy.bind(env.info)
    policy.reset(scene)
    try:
        metadata = {'policy':'inspect_robots_dream.policy.DreamPolicy','model':policy.info.checkpoint,
                    'instruction':scene.instruction,'seed':0,'control_hz':20,'max_steps':max_steps,
                    'asset_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in assets},
                    'privileged_state_to_policy':False,
                    'versions':{name:importlib.metadata.version(name) for name in
                                ('inspect-robots','inspect-robots-dream','dreamlabs-client')}}
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        observation = observe()
        for name,frame in observation.images.items():
            Image.fromarray(frame).save(output/f'initial_{name}.png')
        started = time.monotonic()
        for step in range(max_steps):
            chunk = policy.act(observation)
            if len(chunk.actions)!=1:
                raise RuntimeError('Expected one Dream action per control step')
            values = np.asarray(chunk.actions[0].data,dtype=float)
            if values.shape!=(14,) or not np.isfinite(values).all():
                raise RuntimeError('Nonfinite or invalid Dream action')
            action[:] = values
            observation = tick('dream control',chunk.inference_latency_s)
            if step>=19 and (step+1)%early_stop_interval==0 and audit(trace[-20:])['score']==1.:
                break
        result = audit(trace)
        return result | {'status':'completed','kind':'dream','wall_time_s':time.monotonic()-started,
                         'api_requests':len(policy.transcript()['requests'])}
    finally:
        (output/'dream_transcript.json').write_text(json.dumps(policy.transcript(),indent=2)+'\n')
        policy.close()
