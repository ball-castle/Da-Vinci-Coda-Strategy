import time
from typing import Dict, List, Tuple, Set, Optional

# Constants for bitmask
FULL_MASK = 0b111111111111

def create_mask(available_numbers: List[int]) -> int:
    mask = 0
    for num in available_numbers:
        if 0 <= num <= 11:
            mask |= (1 << num)
    return mask

def get_numbers_from_mask(mask: int) -> List[int]:
    return [i for i in range(12) if (mask & (1 << i))]

def fast_dfs_bitmask(
    search_positions,
    available_cards,
    preassigned_hidden,
    feasible_domains_func,
    behavior_model,
    guess_signals_by_player,
    game_state,
    hard_position_weights,
    soft_position_weights,
    accumulate_func
):
    print('[Bitmask Engine] Hook invoked!')
    
    legal_world_count = 0
    total_soft_weight = 0.0
    current_assignment = {
        player_id: dict(cards_by_slot)
        for player_id, cards_by_slot in preassigned_hidden.items()
    }

    # Extremely fast cache using integers
    dead_state_cache: Set[int] = set()

    def dfs(remaining_positions, remaining_cards) -> bool:
        nonlocal legal_world_count, total_soft_weight

        # Build a rapid integer signature mapping players and cards
        # Note: In production this is packed into Int64
        # hash() gives a quick Int for the current snapshot
        sig_tuple = (len(remaining_positions), tuple(current_assignment.keys()), tuple(len(v) for v in current_assignment.values()), remaining_cards)
        state_signature = hash(sig_tuple)
        
        if state_signature in dead_state_cache:
            return False

        if not remaining_positions:
            legal_world_count += 1
            hypothesis_by_player = {
                pid: dict(cbs) for pid, cbs in current_assignment.items() if cbs
            }
            # Calculate score
            soft_weight = behavior_model.score_hypothesis(
                hypothesis_by_player, guess_signals_by_player, game_state
            )
            total_soft_weight += soft_weight
            accumulate_func(
                hard_position_weights, soft_position_weights, current_assignment, soft_weight
            )
            return True

        domains = feasible_domains_func(remaining_positions, remaining_cards, current_assignment)
        if any(len(d) == 0 for _, d in domains):
            dead_state_cache.add(state_signature)
            return False

        # Sort by most constrained
        domains.sort(key=lambda x: (len(x[1]), x[0].player_id, x[0].slot_index))
        next_position, next_domain = domains[0]
        next_rem_pos = tuple(p for p in remaining_positions if p != next_position)
        
        found_any = False
        for candidate in next_domain:
            current_assignment[next_position.player_id][next_position.slot_index] = candidate
            next_rem_cards = tuple(c for c in remaining_cards if c != candidate)

            if dfs(next_rem_pos, next_rem_cards):
                found_any = True

            del current_assignment[next_position.player_id][next_position.slot_index]

        if not found_any:
            dead_state_cache.add(state_signature)
        return found_any

    start_t = time.time()
    dfs(tuple(search_positions), tuple(available_cards))
    print(f'[Bitmask Engine] fast_dfs_bitmask completed in {time.time() - start_t:.4f} secs')
    
    return legal_world_count, total_soft_weight

