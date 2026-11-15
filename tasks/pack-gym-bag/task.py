"""Privileged expert fixture: scoring is recorded by run_expert.py."""
from yam_arena.task_file import Asset
from layout import ITEMS
SCENE='station'
OBJECTS=[Asset('gym_bag','custom',(.0,.0),extent=.8)]+[Asset(n,'custom',(0.,0.),extent=.3,mass=c['mass']) for n,c in ITEMS.items()]
SUCCESS_HOLD_S=2.
SCORE_OVER='end'
