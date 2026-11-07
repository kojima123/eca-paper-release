"""bp35 perception-driven adapter — L1+L2+L3+L4 unlock.

bp35 is a click-destroy/walk platformer:
  - ACTION3 = walk left, ACTION4 = walk right (gravity falls in -y).
  - ACTION6 = click; sprite at clicked cell determines effect:
      - qclfkhjnaac (destructible plat): destroyed.
      - yuuqpmlxorv (solid toggle): morphs in-place to oonshderxef
        (passable).
      - oonshderxef (passable toggle): morphs back to yuuqpmlxorv.
      - lrpkmzabbfa: flips gravity (vivnprldht).
      - etlsaqqtjvn: spawns adjacent siblings into empty cells.
    AND, if the click cell is directly "below" the player
    (player.x, player.y - 1 with vivnprldht=True) AND the cell
    becomes passable after morph/destroy, a fall is triggered
    toward the gem.
  - Win when player walks into the gem cell (fjlzdjxhant) OR when a
    fall lands on the gem cell (only the cell directly below player
    is checked for fall trigger; remote clicks destroy without fall).

The world is 11x36+ cells, each cell 6 frame-pixels. Camera y-offset
follows the player (cam_y = player.y * 6 - 36) so screen_y for a
click at world (gx, gy) is `gy*6 + 3 - cam_y`.

L1 plan (17 actions, hand-derived from bp35.py source):
  Player at (3, 23), gem at (3, 7). 17-action click+walk plan.

L2 plan (44 actions, hand-derived):
  Player at (3, 37), gem at (5, 7). 50x11 grid with spike rows
  (color 15 = ubhhgljbnpu) creating death barriers. Path:
    - Walk right to x=8, drop col 8 to y=29 (2 plat clicks).
    - Traverse left along y=29 destroying col 5/4/3/2 plats so the
      player can walk to (2, 29) and click (2, 28) to drop to (2, 25)
      (col 2 has wall at y=24 catching the fall before any spike).
    - Walk right to (5, 25); click (5, 24) and (5, 23) to drop into
      the (5, 21) pocket (wall at (5, 20) catches the fall).
    - Walk left to (3, 21); click (3, 20) drops to (3, 18) (the lone
      empty cell in the y=18 spike row).
    - Click (3, 16) remotely (destroy without fall), then click (3, 17)
      drops to (3, 16) through the now-empty (3, 16).
    - Walk along y=16 destroying (4..8, 16) plats; reach (8, 16).
    - Click (8, 15) and (8, 14) to drop down col 8 to (8, 11)
      (plat at (8, 10) catches).
    - Walk left along y=11; (6, 11) walk falls onto (6, 10) plat.
    - Walk left to (5, 10); click (5, 9) drops onto gem (5, 7) -> WIN.

  Wall-threat budget: the (aknlbboysnc, jcyhkseuorf) moving wall
  starts at y=44 and advances 1 grid cell per non-fall horizontal walk
  whose wjidupyeoa parity is even. The plan keeps the wall well above
  the player throughout (final wall y ~= 34 vs player y ~= 7).

L3 plan (39 actions, hand-derived):
  Player at (3, 28), gem at (7, 7). 11x41 grid with yuuqpmlxorv /
  oonshderxef toggle pairs creating reversible solid/passable cells.
  The fall-path detection (fsvnqdbzrp) treats oonshderxef as empty
  (falls through) and yuuqpmlxorv as solid (stops fall). Spike rows
  at y=21 (cols 3-6), y=16 (cols 5-6), y=10 (cols 4-6) and y=4 are
  death barriers. Path:
    - Click (5, 28) toggle 1->2 to unblock walk-right corridor.
    - Walk right (3,28)->(7,28); click (7,27) destructible -> fall
      to (7, 24).
    - Click (5, 23), (4, 23), (3, 23) to convert oonshderxef ->
      yuuqpmlxorv (solid), creating a platform above the y=21 spike
      row so we can walk left along y=24 without dying.
    - Walk left (7,24)->(2,24)->fall through col 2 (always-empty
      corridor down to wall at y=18) to (2, 19).
    - Walk right (2,19)->(3,19)->fall via empty (3, 18) to land on
      wall at y=17.
    - Click (5, 18), (5, 17), (6, 18), (6, 17): morph (5/6, 18) to
      passable AND morph (5/6, 17) to solid (so walking onto the
      passable layer doesn't free-fall onto the y=16 spikes below).
    - Walk right (3,18)->(7,18)->fall through col 7 (cols 7-8 are
      empty all the way down to col-7 wall at y=12) to (7, 13).
    - Click (4, 12), (5, 12), (6, 12) to convert oonshderxef ->
      yuuqpmlxorv (solid), creating a platform above the y=10
      spike row so we can walk along y=13.
    - Walk left (7,13)->(2,13). On the first step from (7,14) the
      player falls 1 cell to (6, 13) because (7, 12) is wall but
      (6, 12) is now solid toggle.
    - Click (2, 12) directly below player (player.y-1=12) to
      trigger fall: (2, 12) morphs to oonshderxef and the fall
      cascades down col 2 through empty cells y=11..7 stopping on
      the (2, 6) yuuqpmlxorv solid -> player at (2, 7).
    - Walk right (2,7)->(4,7); click (5, 7) toggle to passable;
      walk to (5, 7); click (6, 7) toggle to passable; walk to
      (6, 7); walk to (7, 7) GEM -> WIN.

  Wall-threat budget: 39 actions / 64 budget; player ends at y=7
  while the wall starts at y=35 and advances at most ~19 cells, so
  no overlap.

L4 plan (25 actions, hand-derived):
  Player at (4, 14), gem at (4, 27). 11x39 grid. **New mechanic**:
  `lrpkmzabbfa` (gravity-flip switches, color 8) at (5, 7), (3, 23),
  (5, 23), (4, 31). Clicking a switch toggles `vivnprldht`
  (gravity direction) AND triggers a forced fall in the new
  direction; switch sprite is removed (single-use). Wall threat is
  disabled for L4+ (`mylefxfaev` returns False at qswcochjodb > 3).

  Path: walk right ×4 to ride fall-up ladder to (8, 9), click switch
  (5, 7) -> gravity DOWN, fall to (8, 16); walk left ×5 to (3, 16);
  click (3, 17) plat -> fall to (3, 21); walk right ×2 to (5, 21);
  click switch (3, 23) -> gravity UP, fall up to (5, 20); walk
  right ×3 to (8, 20); remote-destroy plats (8, 23) and (8, 24);
  click switch (5, 23) -> gravity DOWN, fall down col 8 to (8, 29);
  walk left ×4 to (4, 29); click switch (4, 31) -> gravity UP,
  forced fall up onto gem (4, 27) -> WIN.

Public API:
    plan_next_actions_from_state(state) -> list of (GameAction, data)
    perceive_and_plan(frame) -> list of (GameAction, data)
"""
from __future__ import annotations

from typing import Any, List, Tuple

from arcengine import GameAction

from arcgym.perception.bp35_perception import perceive_bp35


# Convert (world_grid_x, world_grid_y, expected_player_y_at_click)
# -> screen click (x, y) for ACTION6.
# cam_y = player_y * 6 - 31 + tprcybckbl, where tprcybckbl = -5 if
# gravity is UP (vivnprldht=True) else +5. So:
#   gravity UP   -> cam_y = player_y * 6 - 36
#   gravity DOWN -> cam_y = player_y * 6 - 26
# screen_x = gx * 6 + 3
# screen_y = gy * 6 + 3 - cam_y
# L1-L3 keep gravity UP throughout, so the original `_click` ignores
# direction and always uses the UP formula. L4 introduces
# `lrpkmzabbfa` (gravity-flip switches) and toggles direction
# multiple times within a level, so a direction-aware variant is
# needed.
def _click(gx: int, gy: int, player_y_at_click: int) -> dict:
    cam_y = player_y_at_click * 6 - 36
    sx = gx * 6 + 3
    sy = gy * 6 + 3 - cam_y
    return {"x": sx, "y": sy}


def _click_g(gx: int, gy: int, player_y_at_click: int, gravity_up: bool) -> dict:
    """Direction-aware click for L4+. ``gravity_up`` is the value of
    ``vivnprldht`` at the moment of the click."""
    cam_y = player_y_at_click * 6 - (36 if gravity_up else 26)
    sx = gx * 6 + 3
    sy = gy * 6 + 3 - cam_y
    return {"x": sx, "y": sy}


# L1 plan: list of (GameAction, data dict).
_L1_PLAN: List[Tuple[GameAction, dict]] = [
    # 1-5: walk right; on step 4 the player enters empty col x=7 and
    #      falls to (7, 20); step 5 walks to (8, 20).
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 6: click (8, 19) with player at y=20 -> destroy + fall to (8, 16).
    (GameAction.ACTION6, _click(8, 19, player_y_at_click=20)),
    # 7-9: walk left to (5, 16).
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 10: click (4, 16); player at y=16 (no fall trigger, just destroys).
    (GameAction.ACTION6, _click(4, 16, player_y_at_click=16)),
    # 11: click (4, 15); player at y=16.
    (GameAction.ACTION6, _click(4, 15, player_y_at_click=16)),
    # 12: walk left -> player falls to (4, 13).
    (GameAction.ACTION3, {}),
    # 13: click (4, 12); player at y=13 -> destroy + fall to (4, 10).
    (GameAction.ACTION6, _click(4, 12, player_y_at_click=13)),
    # 14: walk right to (5, 10).
    (GameAction.ACTION4, {}),
    # 15: click (5, 9); player at y=10 -> destroy + fall to (5, 7).
    (GameAction.ACTION6, _click(5, 9, player_y_at_click=10)),
    # 16-17: walk left twice -> reach gem at (3, 7) -> WIN.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
]


# L2 plan: 44 actions. Player at (3, 37), gem at (5, 7).
_L2_PLAN: List[Tuple[GameAction, dict]] = [
    # 1-5: walk right (3,37)->(8,37). Player rests on plats at y=36.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 6: click (8, 36) at player (8,37) -> destroy + 1-cell fall to (8,36).
    (GameAction.ACTION6, _click(8, 36, player_y_at_click=37)),
    # 7: click (8, 35) at player (8,36) -> destroy + fall through col 8
    #    cols y=34..29 (all empty) until plat at (8, 28) -> stops at (8, 29).
    (GameAction.ACTION6, _click(8, 35, player_y_at_click=36)),
    # 8-9: walk left (8,29)->(6,29) along the col-6/7/8 empty band.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 10-17: zig-zag click-destroy + walk-left along y=29 down to (2, 29).
    #        Each click destroys plat to the left; subsequent walk fills.
    (GameAction.ACTION6, _click(5, 29, player_y_at_click=29)),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION6, _click(4, 29, player_y_at_click=29)),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION6, _click(3, 29, player_y_at_click=29)),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION6, _click(2, 29, player_y_at_click=29)),
    (GameAction.ACTION3, {}),
    # 18: at (2, 29). click (2, 28) -> destroy + fall through col 2 y=27..25
    #     until wall at (2, 24) -> stops at (2, 25).
    (GameAction.ACTION6, _click(2, 28, player_y_at_click=29)),
    # 19-21: walk right (2,25)->(5,25). y=24 is wall (cols 2,3,4) then plat at
    #         (5, 24). No falls.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 22: click (5, 24) at player (5,25) -> destroy + 1-cell fall to (5, 24).
    (GameAction.ACTION6, _click(5, 24, player_y_at_click=25)),
    # 23: click (5, 23) at player (5,24) -> destroy + fall through (5, 22), (5, 21)
    #     until wall at (5, 20) -> stops at (5, 21).
    (GameAction.ACTION6, _click(5, 23, player_y_at_click=24)),
    # 24-25: walk left (5,21)->(3,21). y=20 has plats at (3,4,5) so no fall.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 26: click (3, 20) at player (3,21) -> destroy + fall through (3, 19) empty,
    #     (3, 18) empty, until plat at (3, 17) -> stops at (3, 18).
    (GameAction.ACTION6, _click(3, 20, player_y_at_click=21)),
    # 27: click (3, 16) REMOTE (player.y-1=17 != 16) -> destroys plat without fall.
    #     Player still at (3, 18).
    (GameAction.ACTION6, _click(3, 16, player_y_at_click=18)),
    # 28: click (3, 17) at player (3,18) -> destroy + fall through now-empty
    #     (3, 16), until plat at (3, 15) -> stops at (3, 16).
    (GameAction.ACTION6, _click(3, 17, player_y_at_click=18)),
    # 29-38: zig-zag click-destroy + walk-right along y=16 to (8, 16).
    #         Each click destroys plat in next col; walk steps onto it.
    (GameAction.ACTION6, _click(4, 16, player_y_at_click=16)),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION6, _click(5, 16, player_y_at_click=16)),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION6, _click(6, 16, player_y_at_click=16)),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION6, _click(7, 16, player_y_at_click=16)),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION6, _click(8, 16, player_y_at_click=16)),
    (GameAction.ACTION4, {}),
    # 39: click (8, 15) at player (8,16) -> destroy + 1-cell fall to (8, 15).
    (GameAction.ACTION6, _click(8, 15, player_y_at_click=16)),
    # 40: click (8, 14) at player (8,15) -> destroy + fall through (8, 13..11)
    #     empty, plat at (8, 10) -> stops at (8, 11).
    (GameAction.ACTION6, _click(8, 14, player_y_at_click=15)),
    # 41-43: walk left along y=11. (7,11) no fall (plat at 7,10); (6,11) walks
    #         and FALLS onto (6, 10) (plat at (6, 9)); (5,10) no fall (plat at 5,9).
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 44: click (5, 9) at player (5,10) -> destroy + fall onto gem (5, 7) -> WIN.
    (GameAction.ACTION6, _click(5, 9, player_y_at_click=10)),
]


# L3 plan: 39 actions. Player at (3, 28), gem at (7, 7).
_L3_PLAN: List[Tuple[GameAction, dict]] = [
    # 1: click (5, 28) yuuqpmlxorv -> oonshderxef (unblock walk-right)
    (GameAction.ACTION6, _click(5, 28, player_y_at_click=28)),
    # 2-5: walk right (3,28)->(7,28). (5, 28) is now passable.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 6: click (7, 27) destructible -> destroy + fall to (7, 24)
    #    (col 7 empty y=26..24, wall at (7, 23) catches).
    (GameAction.ACTION6, _click(7, 27, player_y_at_click=28)),
    # 7-9: convert (3, 23), (4, 23), (5, 23) oonshderxef -> yuuqpmlxorv
    #      so they become solid platforms above the y=21 spike row.
    (GameAction.ACTION6, _click(3, 23, player_y_at_click=24)),
    (GameAction.ACTION6, _click(4, 23, player_y_at_click=24)),
    (GameAction.ACTION6, _click(5, 23, player_y_at_click=24)),
    # 10-13: walk left (7,24)->(3,24). With (5/4/3, 23) now solid,
    #        fsvnqdbzrp returns wkpljpfhtg=0 so player walks normally.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 14: walk left (3,24)->(2,24)->fall through col 2 (always empty
    #     y=23..19) to wall at y=18 -> player at (2, 19).
    (GameAction.ACTION3, {}),
    # 15: walk right (2,19)->(3,19)->fall to (3, 18) (wall at y=17).
    (GameAction.ACTION4, {}),
    # 16-19: morph (5, 18), (5, 17), (6, 18), (6, 17) so cols 5/6 at
    #        row 18 are passable AND row 17 is solid (block fall onto
    #        the y=16 spikes). Order matters but all 4 commute as far
    #        as final state is concerned.
    (GameAction.ACTION6, _click(5, 18, player_y_at_click=18)),
    (GameAction.ACTION6, _click(5, 17, player_y_at_click=18)),
    (GameAction.ACTION6, _click(6, 18, player_y_at_click=18)),
    (GameAction.ACTION6, _click(6, 17, player_y_at_click=18)),
    # 20-22: walk right (3,18)->(6,18). (5/6, 18) are now oonshderxef
    #        (passable) but (5/6, 17) are now yuuqpmlxorv (solid),
    #        so fsvnqdbzrp wkpljpfhtg=0 and player walks normally.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 23: walk right (6,18)->(7,18)->fall through col 7 (empty
    #     y=17..13) until wall at (7, 12) -> player at (7, 13).
    (GameAction.ACTION4, {}),
    # 24-26: convert (4, 12), (5, 12), (6, 12) oonshderxef ->
    #        yuuqpmlxorv so they become solid platforms above the
    #        y=10 spike row.
    (GameAction.ACTION6, _click(4, 12, player_y_at_click=13)),
    (GameAction.ACTION6, _click(5, 12, player_y_at_click=13)),
    (GameAction.ACTION6, _click(6, 12, player_y_at_click=13)),
    # 27: walk left (7,13)->(6,14) which falls to (6, 13). Initial
    #     pywlvyklps walks at y=13 but wait — fsvnqdbzrp((6, 13)):
    #     (6, 12) is now yuuqpmlxorv solid -> wkpljpfhtg=0 walks.
    #     Actually walking from (7, 13) west: kylnomcxty at (6, 13)
    #     = []; fsvnqdbzrp((6, 13)) finds (6, 12) solid -> walks.
    (GameAction.ACTION3, {}),
    # 28-31: walk left (6,13)->(2,13). All cells y=13 cols 2-5 empty;
    #        cells y=12 cols 2/3 are solid yuuqpmlxorv (always),
    #        cols 4/5/6 are solid yuuqpmlxorv (we morphed). So
    #        wkpljpfhtg=0 across the row.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 32: click (2, 12) yuuqpmlxorv -> oonshderxef. Click matches
    #     (player.x, player.y - 1) = (2, 12) so pbsitubcfd triggers
    #     a fall from (2, 12); fall cascades through empty y=11..7
    #     and stops on (2, 6) yuuqpmlxorv solid -> player at (2, 7).
    (GameAction.ACTION6, _click(2, 12, player_y_at_click=13)),
    # 33-34: walk right (2,7)->(4,7). (2/3/4, 6) are yuuqpmlxorv
    #        solid (always), so player walks at y=7 normally.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 35: click (5, 7) yuuqpmlxorv -> oonshderxef.
    (GameAction.ACTION6, _click(5, 7, player_y_at_click=7)),
    # 36: walk right (4,7)->(5,7). Now passable.
    (GameAction.ACTION4, {}),
    # 37: click (6, 7) yuuqpmlxorv -> oonshderxef.
    (GameAction.ACTION6, _click(6, 7, player_y_at_click=7)),
    # 38: walk right (5,7)->(6,7). Now passable.
    (GameAction.ACTION4, {}),
    # 39: walk right (6,7)->(7,7) GEM -> line 4096 win() -> WIN.
    (GameAction.ACTION4, {}),
]


# L4 plan: 25 actions. Player at (4, 14), gem at (4, 27).
# 11x39 grid (camera tracks player; visible window 64 px tall = ~10
# cells). New mechanic: `lrpkmzabbfa` (gravity-flip switches, color
# 8) at (5, 7), (3, 23), (5, 23), (4, 31). Clicking a switch toggles
# `vivnprldht` (gravity direction) AND triggers a forced fall in the
# new direction (`pbsitubcfd(..., qssroarxob=True)`); the switch
# sprite is then removed (single-use).
#
# Wall threat is disabled at L4+ (`mylefxfaev` returns False for
# `qswcochjodb > 3` at bp35.py:4053), so the moving wall budget is
# not a concern.
#
# Geometry of the gem chamber: gem at (4, 27) is enclosed below
# y=24 walls and behind y=22 wall. Cols 3 and 5 at y=23 are switch
# cells, col 4 at y=23 and y=24 are walls, cols 7-8 at y=23-24 are
# destructible plats. The only fall-route to the gem is via cols
# 7-8 dropping past destroyed plats to y=29, then walking to col 4
# along y=29, and finally clicking switch (4, 31) to flip gravity
# UP and ride the resulting forced fall onto the gem at (4, 27).
#
# Path:
#   (4, 14) gravity UP -> walk right ×4 -> falls up via the y=13/12
#     ladder to (8, 9).
#   click switch (5, 7) -> flip to gravity DOWN -> forced fall down
#     col 8 to (8, 16) (wall at y=17 blocks col 8).
#   walk left ×5 to (3, 16) where (3, 17) is a destructible plat.
#   click (3, 17) plat -> destroy + fall to (3, 21) (col 3 empty
#     y=18..21, wall at y=22).
#   walk right ×2 to (5, 21).
#   click switch (3, 23) -> flip to gravity UP -> forced fall up
#     to (5, 20) (plat at (5, 19) blocks).
#   walk right ×3 to (8, 20). y=19 has plats at cols 5-8 catching
#     each step (gravity UP so cell ABOVE = plat, no fall).
#   click (8, 23) and (8, 24) plats remotely (no fall trigger;
#     remote-click rule from L1 still applies).
#   click switch (5, 23) -> flip to gravity DOWN -> forced fall
#     down col 8 from (8, 21) past the destroyed (8, 23/24) plats
#     and the gap at y=25 to (8, 29) (wall at y=30).
#   walk left ×4 along y=29 to (4, 29). y=30 is wall throughout
#     so no falls.
#   click switch (4, 31) -> flip to gravity UP -> forced fall up
#     from (4, 28) to (4, 27) gem -> WIN via `landed_on_gem` at
#     bp35.py:4256-4257.
_L4_PLAN: List[Tuple[GameAction, dict]] = [
    # 1-4: walk right (4,14)->(5,14)->(6,13)->(7,12)->(8,9) via the
    #      ladder of fall-up steps under the y=11/12/13 ceiling.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 5: click switch (5, 7) at player (8, 9) gravity UP.
    #    Flip gravity to DOWN, forced fall (8, 9)->(8, 16).
    (GameAction.ACTION6, _click_g(5, 7, player_y_at_click=9, gravity_up=True)),
    # 6-10: walk left (8,16)->(7,16)->(6,16)->(5,16)->(4,16)->(3,16).
    #       y=17 cols 4-10 are walls so no fall during walks.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 11: click (3, 17) plat at player (3, 16) gravity DOWN.
    #     kojxiszwpx == eylagpkfjn so destroy + fall to (3, 21).
    (GameAction.ACTION6, _click_g(3, 17, player_y_at_click=16, gravity_up=False)),
    # 12-13: walk right (3,21)->(4,21)->(5,21). y=22 is wall (cols
    #        0-6) so no fall.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 14: click switch (3, 23) at player (5, 21) gravity DOWN.
    #     Flip to UP, forced fall up (5, 21)->(5, 20) (plat at
    #     (5, 19) blocks).
    (GameAction.ACTION6, _click_g(3, 23, player_y_at_click=21, gravity_up=False)),
    # 15-17: walk right (5,20)->(6,20)->(7,20)->(8,20). y=19 plats
    #        at cols 5-8 block the fall-up so each walk is no-fall.
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    (GameAction.ACTION4, {}),
    # 18-19: remote-destroy plats (8, 23) and (8, 24) at player
    #        (8, 20) gravity UP. eylagpkfjn = (8, 19) (plat) so the
    #        destroy click does NOT match eylagpkfjn -> no fall, just
    #        plat destruction (the L1 remote-click pattern).
    (GameAction.ACTION6, _click_g(8, 23, player_y_at_click=20, gravity_up=True)),
    (GameAction.ACTION6, _click_g(8, 24, player_y_at_click=20, gravity_up=True)),
    # 20: click switch (5, 23) at player (8, 20) gravity UP.
    #     Flip to DOWN, forced fall down col 8 (8, 21)..(8, 29) (wall
    #     at y=30; (8, 22) empty, (8, 23/24) destroyed, (8, 25..29)
    #     empty).
    (GameAction.ACTION6, _click_g(5, 23, player_y_at_click=20, gravity_up=True)),
    # 21-24: walk left (8,29)->(7,29)->(6,29)->(5,29)->(4,29). y=30
    #        wall everywhere so no fall.
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    (GameAction.ACTION3, {}),
    # 25: click switch (4, 31) at player (4, 29) gravity DOWN.
    #     Flip to UP, forced fall up (4, 29)->(4, 28)->(4, 27)=gem
    #     -> `landed_on_gem` triggers WIN.
    (GameAction.ACTION6, _click_g(4, 31, player_y_at_click=29, gravity_up=False)),
]


_PLANS: dict[str, List[Tuple[GameAction, dict]]] = {
    "bp35_l1": _L1_PLAN,
    "bp35_l2": _L2_PLAN,
    "bp35_l3": _L3_PLAN,
    "bp35_l4": _L4_PLAN,
}


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    if sig and sig in _PLANS:
        return list(_PLANS[sig])
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_bp35(frame)
    return plan_next_actions_from_state(state)
