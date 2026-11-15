"""Two free key sets placed on the entryway tray."""
from yam_arena.task_file import Asset

SCENE = 'station'
OBJECTS = [Asset(name, 'custom', (0., 0.), extent=.13, mass=.068)
           for name in ('blue_key', 'orange_key')]
