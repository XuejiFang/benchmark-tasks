"""Score the alarm time using recorded physical button travel and release."""

import argparse
import json
from pathlib import Path


START_TIME = 6*60+25
TARGET_TIME = 7*60+15
KEY_VALUES = {'Hour': 60, 'Minute': -5, 'Alarm': 0}
PRESS_TRAVEL = .00215
RELEASE_TRAVEL = .00050
INSTRUCTION = 'Set the alarm to 7:15 AM and arm it. Leave the other clock settings unchanged.'


class ClockInput:
    """The key scan uses a real press and release and emits one event per stroke."""

    def __init__(self):
        self.minute = START_TIME
        self.armed = False
        self.pressed = set()
        self.events = []

    def observe(self, travel):
        change = False
        for key, increment in KEY_VALUES.items():
            if travel[key]>=PRESS_TRAVEL:
                self.pressed.add(key)
            if travel[key]<=RELEASE_TRAVEL and key in self.pressed:
                self.pressed.remove(key)
                if key=='Alarm':
                    self.armed = not self.armed
                else:
                    self.minute = (self.minute+increment) % (24*60)
                self.events.append(key)
                change = True
        return change


def audit(rows):
    if len(rows)<20:
        raise ValueError('At least 20 controls are required to evaluate the released clock')
    clock = ClockInput()
    stable = []
    for row in rows:
        clock.observe(row['travel_m'])
        if row['alarm_minute']!=clock.minute or row['armed']!=clock.armed:
            raise ValueError('Recorded clock state differs from the physical button trace')
        stable.append(clock.minute==TARGET_TIME and not clock.pressed
                      and max(row['travel_m'].values())<=RELEASE_TRAVEL
                      and max(row['contact_n'].values())<=.1)
    correct = all(stable[-20:])
    complete = correct and clock.armed
    return {'score': .5*correct+.5*complete, 'steps': len(rows), 'correct_time_and_released': correct,
            'final_alarm_minute': clock.minute, 'armed': clock.armed, 'complete': complete, 'events': clock.events}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.trial/'trace.jsonl').open()]
    report = audit(rows)
    (args.trial/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
