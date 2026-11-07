"""tr87 rule-table parameter-rotation adapter (Phase 13.B + Phase 15 L2-L6).

Handles tr87 levels 1-6.

Mechanism (per `step()`, `bsqsshqpox`, `wpbnovjwkv`):
  - The level has chains in `cifzvbcuwqe`: each chain is a (left,
    right) pair of `nxkictbbvzt` sprite lists. Top-row sprites
    (`zvojhrjxxm`) form a "premise" sequence; bottom-row sprites
    (`ztgmtnnufb`) form the editable "consequence" sequence.
  - Win predicate `bsqsshqpox` walks the top row and greedily picks
    the FIRST chain whose `left` is a prefix of `top[i:]`; the
    matching `right` must equal `bottom[j:j+len(right)]`. The walk
    consumes `len(left)` from top and `len(right)` from bottom.
  - ACTION3/4: cycle the cursor `qvtymdcqear_index`.
    - Default mode: cursor cycles over `ztgmtnnufb` (bottom row).
    - alter_rules: cursor cycles over flat list of all chain entries
      (`[set for rule in chains for set in rule]`), 2*N_chains slots.
  - ACTION1/2: rotate the currently-selected element by -1/+1
    (modulo `kjgicbtgrt = 7`). Only the suffix digit rotates.
    - Default mode: rotates one bottom sprite.
    - alter_rules: rotates ALL sprites in the selected chain entry
      simultaneously by the same delta.

Generalisation map:
  - L1: alter=None, single-element chains. Direct 1:1 rotation.
  - L2/L3: alter=None, multi-element chains. Greedy chain match
    over top → desired bottom; same per-slot rotation as L1.
  - L4: double_translation=True (alter=None). The win predicate
    composes two chains: chain1.left matches top, chain1.right
    matches chain2.left (prefix via zip), chain2.right is the final
    required bottom. tjaqvwdgkxe markers are absent in all levels
    (verified via probe), so `lonhgifaes` is always the identity.
  - L5: alter_rules=True (no double_translation). Top and bottom
    are FROZEN at their initial random rotations because the
    rotation actions write to chain entries and not to the
    top/bot lists' Sprite refs (which become stale post-rotation
    in the level dict). We brute-search rotation deltas for each
    chain entry such that the greedy match through frozen top
    succeeds AND the resulting required bottom equals frozen bot.
  - L6: double_translation + alter_rules + tree_translation. The
    win predicate substitutes each sprite in the matched chain.right
    with another chain whose left[0].name matches that sprite's
    name (recursive 1-level expansion). For the L6 layout, three
    A-prefix chains [k=0/2/4] are used for the top match, and
    three B-prefix chains [k=1/3/5] are used for tree substitution.
    Solver enumerates top-match permutations × outer right rotations
    × sub-left rotations and derives the remaining sub-right
    rotations from the bot constraints.

Public API:
    plan_next_actions(game) -> list of (GameAction, data) tuples.
"""
from __future__ import annotations

from itertools import product
from typing import Any, List, Optional, Tuple

from arcengine import GameAction


_KJGICBTGRT = 7  # tr87.kjgicbtgrt — rotation cycle modulus


def _short_rotation(current_name: str, target_name: str) -> Tuple[int, int]:
    """Min rotation steps from current to target. Returns (count, +1/-1).

    Names end in a digit 1..7. Forward (+1) cycles 1->2->...->7->1.
    """
    cur_n = int(current_name[-1])
    tgt_n = int(target_name[-1])
    fwd = (tgt_n - cur_n) % _KJGICBTGRT
    bwd = (cur_n - tgt_n) % _KJGICBTGRT
    if fwd <= bwd:
        return (fwd, 1)
    return (bwd, -1)


def _short_rotation_steps(delta_digits: int) -> Tuple[int, int]:
    """Min rotation steps for a given digit delta. delta in {0..6}."""
    fwd = delta_digits % _KJGICBTGRT
    bwd = (-delta_digits) % _KJGICBTGRT
    if fwd <= bwd:
        return (fwd, 1)
    return (bwd, -1)


def _rotate_digit(digit: int, delta: int) -> int:
    """Apply delta rotation to a digit in 1..7. Mirrors wpbnovjwkv."""
    # wpbnovjwkv: new = (old + delta - 1) % 7 + 1, applied per step.
    # Composing k applications of +1 advances digit by k mod 7.
    return ((digit - 1 + delta) % _KJGICBTGRT) + 1


def _find_double_translation_chain(
    intermediate_names: List[str],
    chains: List[Tuple[List[Any], List[Any]]],
) -> Optional[Tuple[List[Any], List[Any]]]:
    """Find first chain whose left, when zipped with intermediate_names, matches.

    Mirrors the inner loop of bsqsshqpox under double_translation.
    Note: zip stops at min length (matches the game's behavior).
    """
    for left, right in chains:
        left_names = [s.name for s in left]
        if all(a == b for a, b in zip(intermediate_names, left_names)):
            return (left, right)
    return None


def _compute_desired_bottom_no_alter(
    top_names: List[str],
    chains: List[Tuple[List[Any], List[Any]]],
    double_translation: bool,
) -> Optional[List[str]]:
    """Walk top, greedy-match each chain prefix; return the required bottom.

    Mirrors `bsqsshqpox` in non-alter_rules mode, optionally with
    double_translation chain composition.
    """
    desired: List[str] = []
    i = 0
    n = len(top_names)
    while i < n:
        found = False
        for left, right in chains:
            k = len(left)
            if i + k > n:
                continue
            if all(top_names[i + j] == left[j].name for j in range(k)):
                if double_translation:
                    intermediate = [s.name for s in right]
                    pair = _find_double_translation_chain(intermediate, chains)
                    if pair is None:
                        return None
                    final_right = pair[1]
                else:
                    final_right = right
                desired.extend(s.name for s in final_right)
                i += k
                found = True
                break
        if not found:
            return None
    return desired


def _greedy_match_top_with_deltas(
    top_names: List[str],
    chains: List[Tuple[List[Any], List[Any]]],
    left_deltas: List[int],
) -> Optional[List[Tuple[int, int, int]]]:
    """Walk top with rotated chain.lefts; return list of (chain_idx, top_i, top_j_end).

    Each entry says: at top index top_i..top_j_end, chain chain_idx was
    matched. None if the walk doesn't cover all of top.
    """
    matches: List[Tuple[int, int, int]] = []
    i = 0
    n = len(top_names)
    while i < n:
        found = False
        for k_idx, (left, _) in enumerate(chains):
            kl = len(left)
            if i + kl > n:
                continue
            ok = True
            for j in range(kl):
                # chain[k_idx].left[j] rotated by left_deltas[k_idx].
                rotated_digit = _rotate_digit(int(left[j].name[-1]), left_deltas[k_idx])
                expected_name = left[j].name[:-1] + str(rotated_digit)
                if top_names[i + j] != expected_name:
                    ok = False
                    break
            if ok:
                matches.append((k_idx, i, i + kl))
                i += kl
                found = True
                break
        if not found:
            return None
    return matches


def _solve_alter_rules(
    top_names: List[str],
    bot_names: List[str],
    chains: List[Tuple[List[Any], List[Any]]],
) -> Optional[Tuple[List[int], List[int]]]:
    """Brute-search rotation deltas for alter_rules levels.

    Returns (left_deltas, right_deltas) per chain such that the greedy
    walk through frozen top matches and the corresponding bottom
    sequence equals frozen bot. Each delta is in {0..6}.

    Heuristic: minimize total rotation cost.
    """
    n_chains = len(chains)
    best: Optional[Tuple[int, List[int], List[int]]] = None

    # Cost of a single delta: short_rotation_steps from 0 (it's just
    # the value applied to the current configuration).
    def cost_of(delta: int) -> int:
        return _short_rotation_steps(delta)[0]

    # Iterate left deltas. 7^n_chains. For tr87 L5 with 4 chains, 7^4 = 2401.
    if n_chains > 5:
        # Search space too large.
        return None
    for left_deltas in product(range(_KJGICBTGRT), repeat=n_chains):
        matches = _greedy_match_top_with_deltas(top_names, chains, list(left_deltas))
        if matches is None:
            continue
        # Compute required bottom for these matches.
        # bot consumed: sum of len(chain[k_idx].right) for each match.
        # right_deltas[k_idx] must rotate chain[k_idx].right to match
        # bot at the corresponding j positions.
        right_deltas: List[Optional[int]] = [None] * n_chains
        bot_idx = 0
        ok = True
        for k_idx, _, _ in matches:
            right_sprites = chains[k_idx][1]
            kr = len(right_sprites)
            if bot_idx + kr > len(bot_names):
                ok = False
                break
            # Required digit shift for first sprite in this chain.right.
            base_digit = int(right_sprites[0].name[-1])
            target_digit = int(bot_names[bot_idx][-1])
            # Letters must match.
            if right_sprites[0].name[:-1] != bot_names[bot_idx][:-1]:
                ok = False
                break
            req_delta = (target_digit - base_digit) % _KJGICBTGRT
            # All sprites in this chain.right must conform to the same delta.
            for j in range(kr):
                rotated = _rotate_digit(int(right_sprites[j].name[-1]), req_delta)
                expected = right_sprites[j].name[:-1] + str(rotated)
                if expected != bot_names[bot_idx + j]:
                    ok = False
                    break
            if not ok:
                break
            if right_deltas[k_idx] is not None and right_deltas[k_idx] != req_delta:
                # This chain.right consumed twice with conflicting deltas.
                ok = False
                break
            right_deltas[k_idx] = req_delta
            bot_idx += kr
        if not ok or bot_idx != len(bot_names):
            continue
        # Fill any unused right_deltas with 0 (they don't affect win).
        rd_filled = [d if d is not None else 0 for d in right_deltas]
        cost = sum(cost_of(d) for d in left_deltas) + sum(cost_of(d) for d in rd_filled)
        if best is None or cost < best[0]:
            best = (cost, list(left_deltas), rd_filled)
    if best is None:
        return None
    return best[1], best[2]


def _plan_alter_rules(
    game: Any,
    chains: List[Tuple[List[Any], List[Any]]],
    top_names: List[str],
    bot_names: List[str],
    cursor: int,
) -> List[Tuple[GameAction, dict]]:
    """Plan rotations for alter_rules level. Returns [] if unsolvable."""
    sol = _solve_alter_rules(top_names, bot_names, chains)
    if sol is None:
        return []
    left_deltas, right_deltas = sol

    # Cursor cycles over flat list: chain[0].left, chain[0].right,
    # chain[1].left, chain[1].right, ..., 2*n_chains slots.
    n_slots = 2 * len(chains)
    # Build list of (slot_index, delta).
    slots: List[Tuple[int, int]] = []
    for k_idx in range(len(chains)):
        if left_deltas[k_idx] != 0:
            slots.append((2 * k_idx, left_deltas[k_idx]))
        if right_deltas[k_idx] != 0:
            slots.append((2 * k_idx + 1, right_deltas[k_idx]))

    if not slots:
        # Already winning? Shouldn't happen but bail safely.
        return []

    # Visit slots in ascending order (cursor moves +1 per ACTION4).
    slots.sort(key=lambda x: x[0])

    plan: List[Tuple[GameAction, dict]] = []
    cur = cursor
    for slot_idx, delta in slots:
        # Cursor navigation (cyclic, n_slots).
        diff = (slot_idx - cur) % n_slots
        # Going backward might be shorter, but ACTION4 forward only requires diff steps.
        # Let's pick the shorter direction.
        fwd = diff
        bwd = (cur - slot_idx) % n_slots
        if fwd <= bwd:
            for _ in range(fwd):
                plan.append((GameAction.ACTION4, {}))
        else:
            for _ in range(bwd):
                plan.append((GameAction.ACTION3, {}))
        cur = slot_idx
        # Apply rotation delta. ACTION2=+1, ACTION1=-1.
        steps, dirn = _short_rotation_steps(delta)
        rot_action = GameAction.ACTION2 if dirn == 1 else GameAction.ACTION1
        for _ in range(steps):
            plan.append((rot_action, {}))

    return plan


def _solve_l6(
    top_names: List[str],
    bot_names: List[str],
    chains: List[Tuple[List[Any], List[Any]]],
) -> Optional[Tuple[List[int], List[int]]]:
    """L6-layout solver: alter_rules + double_translation + tree_translation.

    L6 has 3 A-prefix chains (used for top-row match) and 3 B-prefix
    chains (used for tree substitution). All length-1 lefts. Sub-chain
    rights have length 1; outer-chain rights have length 2.
    """
    n_chains = len(chains)
    if n_chains != 6:
        return None
    a_chain_idx: List[int] = []
    b_chain_idx: List[int] = []
    for k, (left, right) in enumerate(chains):
        if len(left) != 1:
            return None
        prefix = left[0].name[:-1]
        if prefix.endswith("A"):
            if len(right) != 2:
                return None
            a_chain_idx.append(k)
        elif prefix.endswith("B"):
            if len(right) != 1:
                return None
            b_chain_idx.append(k)
        else:
            return None
    if len(a_chain_idx) != 3 or len(b_chain_idx) != 3:
        return None
    if len(top_names) != 3 or len(bot_names) != 6:
        return None

    # Top-row digits.
    top_digits = [int(name[-1]) for name in top_names]
    bot_digits = [int(name[-1]) for name in bot_names]
    # Outer chain.left digits.
    a_left_digits = {k: int(chains[k][0][0].name[-1]) for k in a_chain_idx}
    # Outer chain.right base digits (length 2).
    a_right_digits = {k: [int(s.name[-1]) for s in chains[k][1]] for k in a_chain_idx}
    # Sub chain.left digits.
    b_left_digits = {k: int(chains[k][0][0].name[-1]) for k in b_chain_idx}
    # Sub chain.right base digit.
    b_right_digits = {k: int(chains[k][1][0].name[-1]) for k in b_chain_idx}

    from itertools import permutations

    best_cost = None
    best_plan: Optional[Tuple[List[int], List[int]]] = None

    for perm in permutations(a_chain_idx, 3):
        # perm[i] = chain matched at top position i.
        d_left = [0] * n_chains
        for i in range(3):
            k_outer = perm[i]
            d_left[k_outer] = (top_digits[i] - a_left_digits[k_outer]) % _KJGICBTGRT
        # Greedy ordering check at each top position.
        valid = True
        for i in range(3):
            for k_outer in a_chain_idx:
                if k_outer >= perm[i]:
                    break
                rot_d = ((a_left_digits[k_outer] - 1 + d_left[k_outer]) % _KJGICBTGRT) + 1
                if rot_d == top_digits[i]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Iterate outer right rotations (d_right[a_chain_idx]).
        for d_r_a0 in range(_KJGICBTGRT):
            for d_r_a1 in range(_KJGICBTGRT):
                for d_r_a2 in range(_KJGICBTGRT):
                    d_right = [0] * n_chains
                    d_right_outer_vals = [d_r_a0, d_r_a1, d_r_a2]
                    for j, k in enumerate(a_chain_idx):
                        d_right[k] = d_right_outer_vals[j]

                    # Compute B-letter sequence (in tree-expansion order).
                    # For each top position i, outer chain perm[i].right has 2 sprites.
                    # Each sprite (after d_right rotation) is the B-letter to substitute.
                    b_seq: List[int] = []  # B-digits in order
                    for i in range(3):
                        k_outer = perm[i]
                        for j in range(2):
                            base = a_right_digits[k_outer][j]
                            rot_d = ((base - 1 + d_right[k_outer]) % _KJGICBTGRT) + 1
                            b_seq.append(rot_d)

                    # Group by unique B-digit; check (B → C) consistency.
                    b_to_c: dict[int, int] = {}
                    consistent = True
                    for idx, b_dig in enumerate(b_seq):
                        c_req = bot_digits[idx]
                        if b_dig in b_to_c:
                            if b_to_c[b_dig] != c_req:
                                consistent = False
                                break
                        else:
                            b_to_c[b_dig] = c_req
                    if not consistent or len(b_to_c) > 3:
                        continue

                    # Iterate sub-left rotations (d_left[b_chain_idx]).
                    for d_l_b0 in range(_KJGICBTGRT):
                        for d_l_b1 in range(_KJGICBTGRT):
                            for d_l_b2 in range(_KJGICBTGRT):
                                d_left_sub_vals = {
                                    b_chain_idx[0]: d_l_b0,
                                    b_chain_idx[1]: d_l_b1,
                                    b_chain_idx[2]: d_l_b2,
                                }
                                # Greedy substitute: for each unique B, find first sub-chain
                                # (in chain index order) whose rotated left[0] matches.
                                b_to_subchain: dict[int, int] = {}
                                ok = True
                                for b_dig in b_to_c:
                                    picked = None
                                    for k_sub in b_chain_idx:
                                        rot = ((b_left_digits[k_sub] - 1 + d_left_sub_vals[k_sub]) % _KJGICBTGRT) + 1
                                        if rot == b_dig:
                                            picked = k_sub
                                            break
                                    if picked is None:
                                        ok = False
                                        break
                                    b_to_subchain[b_dig] = picked
                                if not ok:
                                    continue
                                # Each sub-chain handles at most one B (since one rotation = one match).
                                if len(set(b_to_subchain.values())) != len(b_to_subchain):
                                    continue
                                # Derive d_right for each used sub-chain.
                                d_right_sub: dict[int, int] = {}
                                ok = True
                                for b_dig, k_sub in b_to_subchain.items():
                                    c_req = b_to_c[b_dig]
                                    d_right_sub[k_sub] = (c_req - b_right_digits[k_sub]) % _KJGICBTGRT

                                # Assemble full d_left, d_right.
                                d_left_full = list(d_left)
                                d_right_full = list(d_right)
                                for k_sub, dl in d_left_sub_vals.items():
                                    d_left_full[k_sub] = dl
                                for k_sub, dr in d_right_sub.items():
                                    d_right_full[k_sub] = dr

                                # Cost: rotation steps over all 12 deltas.
                                cost = 0
                                for k in range(n_chains):
                                    cost += _short_rotation_steps(d_left_full[k])[0]
                                    cost += _short_rotation_steps(d_right_full[k])[0]

                                if best_cost is None or cost < best_cost:
                                    best_cost = cost
                                    best_plan = (list(d_left_full), list(d_right_full))

    if best_plan is None:
        return None
    return best_plan


def _plan_l6(
    game: Any,
    chains: List[Tuple[List[Any], List[Any]]],
    top_names: List[str],
    bot_names: List[str],
    cursor: int,
) -> List[Tuple[GameAction, dict]]:
    """Plan rotations for L6 (alter+double+tree). Returns [] if unsolvable."""
    sol = _solve_l6(top_names, bot_names, chains)
    if sol is None:
        return []
    d_left_full, d_right_full = sol
    # Each chain entry's slot index in alter_rules cursor cycle: 2*k_chain + (0 for left, 1 for right).
    n_slots = 2 * len(chains)
    slots: List[Tuple[int, int]] = []
    for k in range(len(chains)):
        if d_left_full[k] != 0:
            slots.append((2 * k, d_left_full[k]))
        if d_right_full[k] != 0:
            slots.append((2 * k + 1, d_right_full[k]))
    if not slots:
        return []
    slots.sort(key=lambda x: x[0])

    plan: List[Tuple[GameAction, dict]] = []
    cur = cursor
    for slot_idx, delta in slots:
        fwd = (slot_idx - cur) % n_slots
        bwd = (cur - slot_idx) % n_slots
        if fwd <= bwd:
            for _ in range(fwd):
                plan.append((GameAction.ACTION4, {}))
        else:
            for _ in range(bwd):
                plan.append((GameAction.ACTION3, {}))
        cur = slot_idx
        steps, dirn = _short_rotation_steps(delta)
        rot_action = GameAction.ACTION2 if dirn == 1 else GameAction.ACTION1
        for _ in range(steps):
            plan.append((rot_action, {}))
    return plan


def plan_next_actions(game: Any) -> List[Tuple[GameAction, dict]]:
    """Return a win sequence for the current level as (action, data) tuples."""
    chains = getattr(game, "cifzvbcuwqe", None)
    top_row = getattr(game, "zvojhrjxxm", None)
    bot_row = getattr(game, "ztgmtnnufb", None)
    cursor = getattr(game, "qvtymdcqear_index", None)
    if not chains or not top_row or not bot_row or cursor is None:
        return []

    alter_rules = bool(game.current_level.get_data("alter_rules"))
    double_translation = bool(game.current_level.get_data("double_translation"))
    tree_translation = bool(game.current_level.get_data("tree_translation"))

    top_names = [s.name for s in top_row]
    bot_names = [s.name for s in bot_row]

    if tree_translation:
        # L6: alter + double + tree. Only this combo is supported.
        if alter_rules and double_translation:
            return _plan_l6(game, chains, top_names, bot_names, cursor)
        return []

    if alter_rules:
        # L5: alter_rules without tree/double.
        if double_translation:
            return []
        return _plan_alter_rules(game, chains, top_names, bot_names, cursor)

    # Default mode (alter=False): mutable bottom, frozen top.
    desired = _compute_desired_bottom_no_alter(top_names, chains, double_translation)
    if desired is None or len(desired) != len(bot_row):
        return []

    plan: List[Tuple[GameAction, dict]] = []
    cur_pos = cursor
    for i, tgt in enumerate(desired):
        cur_name = bot_names[i]
        steps, dirn = _short_rotation(cur_name, tgt)
        # Navigate cursor from cur_pos to i. Cursor cycles over bottom (length=len(bot)).
        n_slots = len(bot_row)
        fwd = (i - cur_pos) % n_slots
        bwd = (cur_pos - i) % n_slots
        if fwd <= bwd:
            for _ in range(fwd):
                plan.append((GameAction.ACTION4, {}))
        else:
            for _ in range(bwd):
                plan.append((GameAction.ACTION3, {}))
        cur_pos = i
        rot_action = GameAction.ACTION2 if dirn == 1 else GameAction.ACTION1
        for _ in range(steps):
            plan.append((rot_action, {}))

    return plan
