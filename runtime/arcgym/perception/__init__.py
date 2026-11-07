"""Frame-based perception bridge (Phase 10 / Phase 11).

Replaces the per-game adapter's ground-truth backdoor
(``env._env._game.current_level.get_sprites_by_tag(...)``) with
inference from the rendered 64×64 frame. Required for remote
ARC-AGI-3 API deployment where the game internals are not Python-
accessible.

Each game-specific perception module exports a ``perceive_<game>``
function: ``frame -> state`` where ``state`` is a dict matching the
adapter's expected interface (player_pos, target_positions,
fill_zones, walls, ...).
"""
