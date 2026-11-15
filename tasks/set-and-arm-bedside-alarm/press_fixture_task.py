"""Adapter bookkeeping object; the articulated test prop is added by the runner."""
from yam_arena.task_file import Box

SCENE = 'station'
OBJECTS = [Box(name='fixture_marker', pos=(-.5, 0., -.05), size=(.01, .01, .01))]
