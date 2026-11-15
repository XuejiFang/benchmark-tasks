"""Audit physical separation and a stable upper-bowl placement from telemetry."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def audit(folder):
    rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    result = json.loads((folder/'result.json').read_text())
    start = rows[69]['bowls']
    stall = [r['bowls'] for r in rows if r['phase'] == 'lift upper bowl'][-1]
    final = rows[-20:]
    above = np.array([r['bowls']['teal_bowl']['pose_xyzw'] for r in final])
    typical = np.array([r['bowls']['cream_bowl']['pose_xyzw'] for r in rows])
    up = Rotation.from_quat(above[:, 3:]).apply([[0., 0., 1.]])
    mat_error = float(np.max(np.linalg.norm(above[:, :2]-result['mat_xy_m'], axis=1)))
    span = float(np.max(np.linalg.norm(above[:, :3]-above[-1, :3], axis=1)))
    return {
        'steps': len(rows),
        'lift_m': stall['teal_bowl']['pose_xyzw'][2]-start['teal_bowl']['pose_xyzw'][2],
        'lower_bowl_max_rise_m': float(np.max(typical[:,2])-start['cream_bowl']['pose_xyzw'][2]),
        'bilateral_contact_frames': sum(
            min(r['bowls']['teal_bowl']['finger_force_n']) > .5 for r in rows
            if r['phase'] == 'lift upper bowl'),
        'mat_error_max_m': mat_error,
        'final_drift_max_m': span,
        'normal_up_min_dot': float(up[:,2].min()),
        'final_gripper_force_max_n': max(sum(r['bowls']['teal_bowl']['finger_force_n']) for r in final),
        'stable_on_mat': bool(mat_error <= .026 and span <= .003 and np.min(up[:,2]) > .996
                             and np.max(np.abs(above[:,2]-.754)) <= .004),
    }


def score_two_settings(rows):
    if len(rows) < 20:
        return {'score': 0., 'steps': len(rows), 'components': {}}

    final = rows[-20:]
    filters = [
        ('bowls', 'teal_bowl', (.46, .37, .754)),
        ('bowls', 'cream_bowl', (.46, -.37, .754)),
        ('spoons', 'dark_spoon', (.46, .235, .75)),
        ('spoons', 'light_spoon', (.46, -.235, .75)),
    ]
    components = {}
    for kind, name, center in filters:
        poses = np.array([r[kind][name]['pose_xyzw'] for r in final], dtype=float)
        pos = poses[:, :3]
        rotation = Rotation.from_quat(poses[:, 3:])
        up = rotation.apply([0., 0., 1.])[:, 2]
        forward = rotation.apply([0., 1., 0.])[:, 0]
        planar_error = float(np.max(np.linalg.norm(pos[:, :2]-center[:2], axis=1)))
        height_error = float(np.max(np.abs(pos[:, 2]-center[2])))
        drift = float(np.max(np.linalg.norm(pos-pos[-1], axis=1)))
        angular_drift = float(np.max((rotation[-1].inv()*rotation).magnitude()))
        forces = [r[kind][name].get('gripper_force_n') for r in final]
        release_observed = all(f is not None and len(f) == 6 for f in forces)
        peak_contact = float(np.max(np.sum(forces,axis=1))) if release_observed else None
        # Bowl centre on its mat; spoon head toward the robot bases.
        passed = (planar_error <= .026 and height_error <= .008 and drift <= .003
                  and angular_drift <= np.deg2rad(2.) and up.min() >= .996
                  and (kind == 'bowls' or forward.max() <= -.965)
                  and (peak_contact is None or peak_contact <= .1))
        components[name] = {'passed': bool(passed), 'max_planar_error_m': planar_error,
                            'max_height_error_m': height_error, 'max_drift_m': drift,
                            'max_angular_drift_rad': angular_drift,
                            'max_gripper_contact_n': peak_contact}
    return {'score': sum(info['passed'] for info in components.values()) / 4.,
            'steps': len(rows), 'components': components}


def audit_two_settings(folder):
    rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    report = score_two_settings(rows)
    if rows and rows[-1]['phase'] == 'dream control':
        result_path = folder/'result.json'
        transcript_path = folder/'dream_transcript.json'
        meta_path = folder/'metadata.json'
        report['complete'] = False
        if result_path.exists() and transcript_path.exists() and meta_path.exists():
            result = json.loads(result_path.read_text())
            transcript = json.loads(transcript_path.read_text())
            meta = json.loads(meta_path.read_text())
            report['api_requests'] = len(transcript['requests'])
            report['complete'] = bool(result.get('status') == 'completed'
                                      and result['steps'] == report['steps']
                                      and result['score'] == report['score']
                                      and report['api_requests'] > 0
                                      and meta['max_steps'] == 2400
                                      and all(c['max_gripper_contact_n'] is not None
                                              for c in report['components'].values())
                                      and (report['steps'] == meta['max_steps'] or report['score'] == 1.))
    else:
        report['complete'] = bool(rows and rows[-1]['phase'] == 'both settings settle')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial', type=Path)
    parser.add_argument('--two-settings', action='store_true')
    args = parser.parse_args()
    report = audit_two_settings(args.trial) if args.two_settings else audit(args.trial)
    (args.trial/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
