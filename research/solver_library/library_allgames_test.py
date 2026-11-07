"""All games of the shipped library (inference/data/solver_library.json, default path), end to end on the local engine; derived from library_e2e_test.py.
Original: End to end on the local cn04 game: the solver's real _library_due / _run_library / _execute_action / _deliver_host_note on a
stand-in object whose game is the local engine. Library = the b2 harvest (cn04 L1-6 played by the model in ecaoff b2).
Cases: (1) the whole game replays, one host note; (2) the stored hash of L3's 5th action is corrupted: the replay stops there
and the note says so; (3) the L1 start hash is wrong: nothing is replayed; (4) the flag off: nothing is replayed."""
import sys, os, types, json, copy
ROOT = sys.argv[1]; sys.path.insert(0, ROOT)
for k in list(os.environ):
    if k.startswith("ECA_FRZ_"): del os.environ[k]
os.environ.update({"ECA_FRZ_HOST_NOTE": "1", "ECA_FRZ_HOSTNOTE_LEVEL_FIX": "1", "ARC3_ANIMATION": "1",
                   "ECA_FRZ_SOLVER_ENDFACTS": "1"})
import numpy as np
import arc_agi, arcengine
_taaf = types.ModuleType("taaf"); _tg = types.ModuleType("taaf.game"); _ts = types.ModuleType("taaf.solver")
class _X: pass
_tg.GameState, _tg.Game, _ts.Solver = _X, _X, _X; _taaf.game, _taaf.solver = _tg, _ts
sys.modules.update({"taaf": _taaf, "taaf.game": _tg, "taaf.solver": _ts})
for _m in ("requests", "openai", "httpx"):
    try: __import__(_m)
    except ModuleNotFoundError: sys.modules[_m] = types.ModuleType(_m)
from inference.framework import solver as S
from inference.framework import solver_library as L
CLS = next(c for c in vars(S).values() if isinstance(c, type) and hasattr(c, "_deliver_host_note") and hasattr(c, "_execute_action"))
ARC = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)

class St:
    def __init__(self, obs, just_won):
        self.raw = obs; self.levels_completed = int(obs.levels_completed); self.just_won_level = just_won
        self.frame = types.SimpleNamespace(data=np.array(obs.frame[-1], dtype=int))
        self.won = obs.state == arcengine.GameState.WIN; self.available_actions = list(obs.available_actions)
GAME = {}
class Game:
    def __init__(self):
        self.env = ARC.make(GAME["id"].split("-")[0]); obs = self.env.reset()
        self.current_state = St(obs, False); self.number_of_levels = GAME["levels"]
        self.game_run = types.SimpleNamespace(game_id=GAME["id"], history=[])
    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        lv0 = self.current_state.levels_completed
        obs = self.env.step(action.id, data=dict(action.data))
        self.game_run.history.append(action)
        self.current_state = St(obs, obs.levels_completed > lv0)
        return self.current_state

class Fake:
    action_count = property(lambda self: len(self.game.game_run.history))   # the solver's: len(game_run.history)

def fake():
    f = Fake()
    f.game = Game(); f.token_baseline = 0; f.concurrency = 1
    f._mechanic_action_source_family = ""; f._toolbox_last_certified_families = []; f._toolbox_last_certified_level = 0
    f._toolbox_plan_snapshot = []; f._toolbox_last_transition = {}; f._episode_mechanic_memory_state = None
    f.animation_record = None; f.last_engine_action = None; f.history_entries = []
    f._host_note_buffer = []; f._solver_failnote_pending = {}; f._solver_endfacts = {}; f.state_path = None
    f.write_runtime_state = lambda: None; f._append_action_viewer_event = lambda *a, **k: None
    f.write_viewer_payload = lambda: None; f.timing_payload = lambda: {}; f.should_stop = lambda: False
    f.captured = {}
    f.analyzer = types.SimpleNamespace(note_host_actions=lambda lines, **kw: f.captured.update(lines=list(lines), kw=kw))
    for name in ("_library_due", "_run_library", "_execute_action", "_deliver_host_note"):
        setattr(f, name, types.MethodType(getattr(CLS, name), f))
    return f


os.environ.pop("ECA_FRZ_LIBRARY_PATH", None); os.environ["ECA_FRZ_LIBRARY"] = "1"
lib = L.load()
NL = {"ar25": 8, "bp35": 9, "cn04": 6, "dc22": 6, "ls20": 7, "re86": 8, "s5i5": 8, "tn36": 7}
bad = 0
for gid, entries in sorted(lib["games"].items()):
    GAME.update(id=gid, levels=NL[gid[:4]])
    f = fake()
    if f._library_due():
        f._run_library()
    done = f.game.current_state.levels_completed
    exp = len(entries)
    ok = done == exp and f.action_count == sum(len(e["actions"]) for e in entries)
    bad += not ok
    print(("ok   " if ok else "FAIL ") + f"{gid}: {done}/{exp} library levels completed, {f.action_count} actions")
print("LIBRARY_ALLGAMES_OK" if not bad else "LIBRARY_ALLGAMES_FAIL %d" % bad)
