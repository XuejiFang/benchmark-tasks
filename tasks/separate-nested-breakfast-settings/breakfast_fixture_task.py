"""Two bowls start nested on a dining surface."""
import os

from yam_arena.task_file import Asset

SCENE = 'station'
OBJECTS = [Asset(name, 'custom', (0., 0.), extent=.158, mass=.31)
           for name in ('cream_bowl', 'teal_bowl')]
if os.environ.get('ROBOCURVE_BREAKFAST_FULL') == '1':
    OBJECTS.extend(Asset(name, 'custom', (0., 0.), extent=.173, mass=.041)
                   for name in ('dark_spoon', 'light_spoon'))
