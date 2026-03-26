from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2, sqrt
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    Card,
    CardSlot,
    GameState,
    GuessAction,
    PlayerState,
)

ProbabilityMatrix = Dict[int, Dict[Card, float]]
FullProbabilityMatrix = Dict[str, ProbabilityMatrix]
SlotKey = Tuple[str, int]

CARD_COLOR_RANK = {"B": 0, "W": 1}
SOFT_BEHAVIOR_BLEND = 0.28


@dataclass(frozen=True)
class HiddenPosition:
    player_id: str
    slot_index: int
    order_index: int
    color: Optional[str]


@dataclass
class HardConstraintSet:
    fixed_by_slot: Dict[SlotKey, Card]
    forbidden_by_slot: Dict[SlotKey, Set[Card]]


@dataclass(frozen=True)
class GuessSignal:
    action_index: int
    guesser_id: str
    target_player_id: str
    target_slot_index: int
    guessed_card: Card
    result: bool
    continued_turn: Optional[bool]


def numeric_card_value(card: Optional[Card]) -> Optional[int]:
    if card is None:
        return None
    value = card[1]
    if value == JOKER or not isinstance(value, int):
        return None
    return int(value)


def card_sort_key(card: Card) -> Tuple[int, int]:
    color, value = card
    if value == JOKER:
        return (MAX_CARD_VALUE + 1, CARD_COLOR_RANK[color])
    return (int(value), CARD_COLOR_RANK[color])


def serialize_card(card: Card) -> List[Any]:
    return [card[0], card[1]]


def normalize_card_distribution(weights: Dict[Card, float]) -> Dict[Card, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {card: weight / total for card, weight in weights.items()}


def slot_key(player_id: str, slot_index: int) -> SlotKey:
    return (player_id, slot_index)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HardConstraintCompiler:
    """Compile hard public evidence into fixed/forbidden slot constraints."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def compile(self) -> HardConstraintSet:
        fixed_by_slot: Dict[SlotKey, Card] = {}
        forbidden_by_slot: Dict[SlotKey, Set[Card]] = defaultdict(set)

        for player_id in self.game_state.inference_player_ids():
            for slot in self.game_state.resolved_ordered_slots(player_id):
                known = slot.known_card()
                if known is not None:
                    self._fix_card(
                        fixed_by_slot,
                        forbidden_by_slot,
                        slot_key(player_id, slot.slot_index),
                        known,
                    )

        for action in getattr(self.game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue

            guessed_card = action.guessed_card()
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if guessed_card is not None and target_player_id is not None and target_slot_index is not None:
                key = slot_key(target_player_id, target_slot_index)
                if getattr(action, "result", False):
                    self._fix_card(fixed_by_slot, forbidden_by_slot, key, guessed_card)
                else:
                    forbidden_by_slot[key].add(guessed_card)

            revealed_card = self._revealed_card_from_action(action)
            revealed_player_id = getattr(action, "revealed_player_id", None)
            revealed_slot_index = getattr(action, "revealed_slot_index", None)
            if revealed_card is not None and revealed_player_id is not None and revealed_slot_index is not None:
                self._fix_card(
                    fixed_by_slot,
                    forbidden_by_slot,
                    slot_key(revealed_player_id, revealed_slot_index),
                    revealed_card,
                )

        return HardConstraintSet(
            fixed_by_slot=fixed_by_slot,
            forbidden_by_slot={key: set(value) for key, value in forbidden_by_slot.items()},
        )

    def _fix_card(
        self,
        fixed_by_slot: Dict[SlotKey, Card],
        forbidden_by_slot: Dict[SlotKey, Set[Card]],
        key: SlotKey,
        card: Card,
    ) -> None:
        current = fixed_by_slot.get(key)
        if current is not None and current != card:
            raise ValueError(
                f"Conflicting hard evidence for slot {key}: {current} vs {card}"
            )
        if card in forbidden_by_slot.get(key, set()):
            raise ValueError(
                f"Slot {key} cannot be both fixed to and forbidden from {card}"
            )
        fixed_by_slot[key] = card

    def _revealed_card_from_action(self, action: Any) -> Optional[Card]:
        if hasattr(action, "revealed_card"):
            revealed = action.revealed_card()
            if revealed is not None:
                return revealed

        color = getattr(action, "revealed_color", None)
        value = getattr(action, "revealed_value", None)
        if color is None or value is None:
            return None
        return (color, value)


class BehavioralLikelihoodModel:
    """Structured soft action likelihood model, separate from hard constraints."""

    SELF_EXACT_GUESS_PENALTY = 0.14
    SELF_ADJACENT_ANCHOR_BONUS = 1.08

    TARGET_PLAYER_BEST_MATCH_BONUS = 1.07
    TARGET_PLAYER_CLOSE_MATCH_BONUS = 1.03
    TARGET_PLAYER_WEAK_CHOICE_PENALTY = 0.94
    TARGET_PLAYER_REPEAT_FOCUS_BONUS = 1.04
    TARGET_PLAYER_CONFIDENT_CHAIN_BONUS = 1.05
    TARGET_PLAYER_BREAK_CONFIDENT_CHAIN_PENALTY = 0.95
    TARGET_PLAYER_SWITCH_AFTER_FAILURE_BONUS = 1.04
    TARGET_PLAYER_SWITCH_FAILURE_CONTINUITY_BONUS = 1.03

    TARGET_SLOT_BEST_MATCH_BONUS = 1.08
    TARGET_SLOT_CLOSE_MATCH_BONUS = 1.03
    TARGET_SLOT_WEAK_CHOICE_PENALTY = 0.91
    TARGET_SLOT_RETRY_AFTER_FAILURE_BONUS = 1.06
    TARGET_SLOT_CONFIDENT_ADJACENT_FOLLOW_BONUS = 1.04
    TARGET_SLOT_FAILURE_ADJACENT_PROBE_BONUS = 1.03

    TARGET_IN_INTERVAL_BONUS = 1.15
    TARGET_NARROW_INTERVAL_BONUS = 1.10
    TARGET_OUTSIDE_INTERVAL_PENALTY = 0.80
    TARGET_NEIGHBOR_BONUS = 1.10
    TARGET_CLOSE_BONUS = 1.04
    TARGET_FAR_PENALTY = 0.97
    TARGET_VALUE_PROGRESSIVE_STEP_BONUS = 1.08
    TARGET_VALUE_DIRECTIONAL_BONUS = 1.03
    TARGET_VALUE_STALLED_PENALTY = 0.90
    TARGET_VALUE_WRONG_DIRECTION_PENALTY = 0.92
    TARGET_VALUE_CONFIDENT_CHAIN_LOCAL_STEP_BONUS = 1.04
    TARGET_VALUE_CONFIDENT_CHAIN_JUMP_PENALTY = 0.95
    TARGET_VALUE_SANDWICH_EXACT_BONUS = 1.12
    TARGET_VALUE_SANDWICH_FILL_BONUS = 1.06
    TARGET_VALUE_WIDE_GAP_CENTER_BONUS = 1.05
    TARGET_VALUE_WIDE_GAP_EDGE_PENALTY = 0.96
    TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS = 1.03
    WRONG_COLOR_SLOT_PENALTY = 0.88

    CONTINUE_HIGH_ATTACKABILITY_BONUS = 1.10
    CONTINUE_LOW_ATTACKABILITY_PENALTY = 0.93
    STOP_LOW_ATTACKABILITY_BONUS = 1.04
    STOP_HIGH_ATTACKABILITY_PENALTY = 0.97

    ATTACKABILITY_TIGHT_THRESHOLD = 0.34
    CONTINUATION_PRIOR_BASE = 0.52
    CONTINUATION_ATTACKABILITY_GAIN = 1.35
    CONTINUATION_MIN = 0.08
    CONTINUATION_MAX = 0.95
    CONTINUATION_RECENT_CONTINUE_BONUS = 0.04
    CONTINUATION_RECENT_STOP_PENALTY = 0.04
    CONTINUATION_CONFIDENT_STREAK_BONUS = 0.03
    EPSILON = 1e-9

    def build_guess_signals(
        self,
        game_state: GameState,
    ) -> Dict[str, List[GuessSignal]]:
        signals_by_player: Dict[str, List[GuessSignal]] = defaultdict(list)

        guess_action_index = 0
        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue

            guessed_card = action.guessed_card()
            guesser_id = getattr(action, "guesser_id", None)
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if (
                guessed_card is None
                or guesser_id is None
                or target_player_id is None
                or target_slot_index is None
            ):
                continue

            signals_by_player[guesser_id].append(
                GuessSignal(
                    action_index=guess_action_index,
                    guesser_id=guesser_id,
                    target_player_id=target_player_id,
                    target_slot_index=target_slot_index,
                    guessed_card=guessed_card,
                    result=bool(getattr(action, "result", False)),
                    continued_turn=getattr(action, "continued_turn", None),
                )
            )
            guess_action_index += 1

        return signals_by_player

    def continuation_profile(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
    ) -> Dict[str, float]:
        if acting_player_id is None:
            return {
                "continue_rate": self.CONTINUATION_PRIOR_BASE,
                "observations": 0.0,
                "history_blend": 0.0,
            }

        signals = guess_signals_by_player.get(acting_player_id, ())
        observed = [
            signal
            for signal in signals
            if signal.result and signal.continued_turn is not None
        ]
        observations = len(observed)
        if observations == 0:
            return {
                "continue_rate": self.CONTINUATION_PRIOR_BASE,
                "observations": 0.0,
                "history_blend": 0.0,
            }

        continue_count = sum(1 for signal in observed if signal.continued_turn)
        continue_rate = (continue_count + 1.0) / (observations + 2.0)
        recent_signal = observed[-1]
        if recent_signal.continued_turn:
            continue_rate += self.CONTINUATION_RECENT_CONTINUE_BONUS
        else:
            continue_rate -= self.CONTINUATION_RECENT_STOP_PENALTY
        if observations >= 2 and observed[-1].continued_turn and observed[-2].continued_turn:
            continue_rate += self.CONTINUATION_CONFIDENT_STREAK_BONUS
        continue_rate = clamp(continue_rate, self.CONTINUATION_MIN, self.CONTINUATION_MAX)
        history_blend = min(0.5, 0.18 * observations)
        return {
            "continue_rate": continue_rate,
            "observations": float(observations),
            "history_blend": history_blend,
        }

    def estimate_continue_likelihood(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> Dict[str, float]:
        attackability = self.estimate_matrix_attackability(
            full_probability_matrix,
            acting_player_id=acting_player_id,
            exclude_slot=exclude_slot,
        )
        attackability_prior = clamp(
            0.5 + self.CONTINUATION_ATTACKABILITY_GAIN * (attackability - self.ATTACKABILITY_TIGHT_THRESHOLD),
            self.CONTINUATION_MIN,
            self.CONTINUATION_MAX,
        )

        profile = self.continuation_profile(guess_signals_by_player, acting_player_id)
        history_blend = profile["history_blend"]
        continue_likelihood = ((1.0 - history_blend) * attackability_prior) + (history_blend * profile["continue_rate"])

        if attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD and profile["continue_rate"] >= 0.60:
            continue_likelihood *= 1.03
        elif attackability < self.ATTACKABILITY_TIGHT_THRESHOLD and profile["continue_rate"] <= 0.45:
            continue_likelihood *= 0.98

        continue_likelihood = clamp(continue_likelihood, self.CONTINUATION_MIN, self.CONTINUATION_MAX)
        return {
            "continue_likelihood": continue_likelihood,
            "attackability": attackability,
            "history_continue_rate": profile["continue_rate"],
            "history_observations": profile["observations"],
        }

    def score_hypothesis(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        game_state: GameState,
    ) -> float:
        weight = 1.0

        for signals in guess_signals_by_player.values():
            for signal in signals:
                weight *= self._score_signal(game_state, hypothesis_by_player, signal)
                if weight <= 0.0:
                    return 0.0

        return weight

    def estimate_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        *,
        acting_player_id: Optional[str] = None,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best = 0.0
        for player_id in game_state.inference_player_ids():
            if acting_player_id is not None and player_id == acting_player_id:
                continue
            best = max(
                best,
                self._player_attackability(
                    game_state,
                    hypothesis_by_player,
                    player_id,
                    exclude_slot=exclude_slot,
                ),
            )
        return best

    def estimate_matrix_attackability(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        *,
        acting_player_id: Optional[str] = None,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best = 0.0
        for player_id, probability_matrix in full_probability_matrix.items():
            if acting_player_id is not None and player_id == acting_player_id:
                continue
            for slot_index, slot_distribution in probability_matrix.items():
                key = slot_key(player_id, slot_index)
                if exclude_slot is not None and key == exclude_slot:
                    continue
                if not slot_distribution:
                    continue
                max_probability = max(slot_distribution.values())
                concentration = sum(probability * probability for probability in slot_distribution.values())
                effective_support = 1.0 / max(self.EPSILON, concentration)
                certainty = max_probability / max(1.0, effective_support ** 0.5)
                if len(slot_distribution) <= 2:
                    certainty *= 1.05
                best = max(best, certainty)
        return clamp(best, 0.0, 1.0)

    def _score_signal(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        target_card = self._resolve_slot_card(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        if target_card is None:
            return 1.0

        weight = 1.0
        guesser_cards = self._player_cards(
            game_state,
            signal.guesser_id,
            hypothesis_by_player.get(signal.guesser_id, {}),
        )
        weight *= self._score_self_hand(guesser_cards, signal.guessed_card)
        weight *= self._score_target_player_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        weight *= self._score_target_slot_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        weight *= self._score_target_value_selection(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )
        weight *= self._score_target_slot(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
        )
        weight *= self._score_continue_decision(
            game_state,
            hypothesis_by_player,
            signal,
        )
        return max(self.EPSILON, weight)

    def _player_cards(
        self,
        game_state: GameState,
        player_id: str,
        hidden_cards_by_slot: Dict[int, Card],
    ) -> Sequence[Card]:
        cards: List[Card] = []
        for slot in game_state.resolved_ordered_slots(player_id):
            known = slot.known_card()
            if known is not None:
                cards.append(known)
                continue
            inferred = hidden_cards_by_slot.get(slot.slot_index)
            if inferred is not None:
                cards.append(inferred)
        return cards

    def _score_self_hand(
        self,
        player_cards: Sequence[Card],
        guessed_card: Card,
    ) -> float:
        weight = 1.0
        if guessed_card in player_cards:
            weight *= self.SELF_EXACT_GUESS_PENALTY

        guessed_numeric = numeric_card_value(guessed_card)
        if guessed_numeric is None:
            return weight

        same_color_values = {
            numeric_card_value(card)
            for card in player_cards
            if card[0] == guessed_card[0] and numeric_card_value(card) is not None
        }
        if (guessed_numeric - 1) in same_color_values or (guessed_numeric + 1) in same_color_values:
            weight *= self.SELF_ADJACENT_ANCHOR_BONUS
        return weight

    def _score_target_player_selection(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        candidate_scores: Dict[str, float] = {}
        for player_id in game_state.players:
            if player_id == signal.guesser_id:
                continue
            player_score = self._player_attackability(
                game_state,
                hypothesis_by_player,
                player_id,
            )
            if player_score > 0.0:
                candidate_scores[player_id] = player_score

        if len(candidate_scores) <= 1:
            return 1.0

        chosen_score = candidate_scores.get(signal.target_player_id)
        if chosen_score is None:
            return 1.0

        best_score = max(candidate_scores.values())
        weight = 1.0
        if chosen_score >= best_score - self.EPSILON:
            weight *= self.TARGET_PLAYER_BEST_MATCH_BONUS
        elif chosen_score >= best_score * 0.85:
            weight *= self.TARGET_PLAYER_CLOSE_MATCH_BONUS
        else:
            weight *= self.TARGET_PLAYER_WEAK_CHOICE_PENALTY

        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is not None and previous_signal.target_player_id == signal.target_player_id:
            if previous_signal.result and previous_signal.continued_turn:
                weight *= self.TARGET_PLAYER_REPEAT_FOCUS_BONUS
                weight *= self.TARGET_PLAYER_CONFIDENT_CHAIN_BONUS
            elif previous_signal.result:
                weight *= self.TARGET_PLAYER_REPEAT_FOCUS_BONUS
        elif (
            previous_signal is not None
            and not previous_signal.result
        ):
            previous_target_score = candidate_scores.get(previous_signal.target_player_id, 0.0)
            if chosen_score > (previous_target_score + 0.08):
                weight *= self.TARGET_PLAYER_SWITCH_AFTER_FAILURE_BONUS
            if self._has_failure_switch_guess_continuity(previous_signal, signal):
                weight *= self.TARGET_PLAYER_SWITCH_FAILURE_CONTINUITY_BONUS
        elif (
            previous_signal is not None
            and previous_signal.result
            and previous_signal.continued_turn
        ):
            weight *= self.TARGET_PLAYER_BREAK_CONFIDENT_CHAIN_PENALTY
        return weight

    def _has_failure_switch_guess_continuity(
        self,
        previous_signal: GuessSignal,
        signal: GuessSignal,
    ) -> bool:
        if previous_signal.result or previous_signal.target_player_id == signal.target_player_id:
            return False

        previous_card = previous_signal.guessed_card
        current_card = signal.guessed_card
        if previous_card[0] != current_card[0]:
            return False

        previous_value = numeric_card_value(previous_card)
        current_value = numeric_card_value(current_card)
        if previous_value is None or current_value is None:
            return False
        return abs(previous_value - current_value) <= 2

    def _score_target_slot_selection(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        candidate_scores: Dict[int, float] = {}
        for slot in game_state.resolved_ordered_slots(signal.target_player_id):
            if slot.known_card() is not None:
                continue
            score = self._slot_attackability(
                game_state,
                hypothesis_by_player,
                signal.target_player_id,
                slot.slot_index,
            )
            if score > 0.0:
                candidate_scores[slot.slot_index] = score

        if len(candidate_scores) <= 1:
            return 1.0

        chosen_score = candidate_scores.get(signal.target_slot_index)
        if chosen_score is None:
            return 1.0

        best_score = max(candidate_scores.values())
        weight = 1.0
        if chosen_score >= best_score - self.EPSILON:
            weight *= self.TARGET_SLOT_BEST_MATCH_BONUS
        elif chosen_score >= best_score * 0.85:
            weight *= self.TARGET_SLOT_CLOSE_MATCH_BONUS
        else:
            weight *= self.TARGET_SLOT_WEAK_CHOICE_PENALTY

        if self._has_prior_failed_guess_on_slot(game_state, signal):
            weight *= self.TARGET_SLOT_RETRY_AFTER_FAILURE_BONUS
        if self._is_adjacent_slot_follow_after_confident_hit(game_state, signal):
            weight *= self.TARGET_SLOT_CONFIDENT_ADJACENT_FOLLOW_BONUS
        if self._is_adjacent_slot_probe_after_failure(game_state, signal):
            weight *= self.TARGET_SLOT_FAILURE_ADJACENT_PROBE_BONUS
        return weight

    def _is_adjacent_slot_follow_after_confident_hit(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> bool:
        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is None:
            return False
        if not previous_signal.result or not previous_signal.continued_turn:
            return False
        if previous_signal.target_player_id != signal.target_player_id:
            return False
        return abs(previous_signal.target_slot_index - signal.target_slot_index) == 1

    def _is_adjacent_slot_probe_after_failure(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> bool:
        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is None:
            return False
        if previous_signal.result:
            return False
        if previous_signal.target_player_id != signal.target_player_id:
            return False
        return abs(previous_signal.target_slot_index - signal.target_slot_index) == 1

    def _score_target_slot(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
    ) -> float:
        guessed_card = signal.guessed_card
        weight = 1.0

        try:
            slot = game_state.get_slot(signal.target_player_id, signal.target_slot_index)
        except ValueError:
            return 1.0

        known_slot_color = getattr(slot, "color", None)
        if known_slot_color is not None and guessed_card[0] != known_slot_color:
            weight *= self.WRONG_COLOR_SLOT_PENALTY

        guessed_numeric = numeric_card_value(guessed_card)
        target_numeric = numeric_card_value(target_card)
        if guessed_numeric is None or target_numeric is None:
            return weight

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )

        if low is not None and high is not None:
            if low <= guessed_numeric <= high:
                weight *= self.TARGET_IN_INTERVAL_BONUS
                if width <= 2:
                    weight *= self.TARGET_NARROW_INTERVAL_BONUS
            else:
                weight *= self.TARGET_OUTSIDE_INTERVAL_PENALTY

        distance = abs(guessed_numeric - target_numeric)
        if distance == 0:
            return weight
        if distance == 1:
            return weight * self.TARGET_NEIGHBOR_BONUS
        if distance <= 2:
            return weight * self.TARGET_CLOSE_BONUS
        return weight * (self.TARGET_FAR_PENALTY ** max(1, distance - 2))

    def _score_target_value_selection(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
        guesser_cards: Sequence[Card],
    ) -> float:
        return self._value_selection_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )["total_weight"]

    def _value_selection_breakdown(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
        guesser_cards: Sequence[Card],
    ) -> Dict[str, Any]:
        guessed_numeric = numeric_card_value(signal.guessed_card)
        target_numeric = numeric_card_value(target_card)
        if guessed_numeric is None or target_numeric is None:
            neutral = {"weight": 1.0, "reason": "neutral"}
            return {
                "total_weight": 1.0,
                "signal_tags": [],
                "dominant_signal": {
                    "source": "neutral",
                    "reason": "neutral",
                    "weight": 1.0,
                },
                "progressive": neutral,
                "anchor": neutral,
                "boundary": neutral,
            }

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        progressive = self._progressive_value_component(
            guessed_numeric,
            target_numeric,
            self._prior_failed_numeric_guesses_on_slot(game_state, signal),
            self._previous_confident_success_numeric_guess(game_state, signal),
        )
        anchor = self._same_color_anchor_value_component(
            guessed_numeric,
            self._same_color_numeric_values(guesser_cards, signal.guessed_card[0]),
        )
        boundary = self._local_boundary_value_component(
            guessed_numeric,
            low,
            high,
            width,
        )
        signal_tags = [
            component["reason"]
            for component in (progressive, anchor, boundary)
            if component["reason"] != "neutral"
        ]
        return {
            "total_weight": progressive["weight"] * anchor["weight"] * boundary["weight"],
            "signal_tags": signal_tags,
            "dominant_signal": self._dominant_signal(
                {
                    "progressive": progressive,
                    "same_color_anchor": anchor,
                    "local_boundary": boundary,
                }
            ),
            "progressive": progressive,
            "anchor": anchor,
            "boundary": boundary,
        }

    def _progressive_value_component(
        self,
        guessed_numeric: int,
        target_numeric: int,
        prior_failed_values: Sequence[int],
        previous_confident_value: Optional[int],
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "latest_failed_value": None,
            "expected_progress": None,
            "direction": 0,
            "previous_confident_value": previous_confident_value,
        }
        if not prior_failed_values:
            if previous_confident_value is None:
                return component
            delta_from_confident = abs(guessed_numeric - previous_confident_value)
            if delta_from_confident == 1:
                component["weight"] = self.TARGET_VALUE_CONFIDENT_CHAIN_LOCAL_STEP_BONUS
                component["reason"] = "confident_chain_local_step"
            elif delta_from_confident >= 4:
                component["weight"] = self.TARGET_VALUE_CONFIDENT_CHAIN_JUMP_PENALTY
                component["reason"] = "confident_chain_jump"
            return component

        latest_failed = prior_failed_values[-1]
        direction = 0
        if target_numeric > latest_failed:
            direction = 1
        elif target_numeric < latest_failed:
            direction = -1

        component["latest_failed_value"] = latest_failed
        component["direction"] = direction
        if direction == 0:
            return component

        expected_progress = latest_failed + direction
        delta = guessed_numeric - latest_failed
        component["expected_progress"] = expected_progress
        if guessed_numeric == expected_progress:
            component["weight"] = self.TARGET_VALUE_PROGRESSIVE_STEP_BONUS
            component["reason"] = "progressive_step"
        elif delta == 0:
            component["weight"] = self.TARGET_VALUE_STALLED_PENALTY
            component["reason"] = "stalled_after_failure"
        elif delta * direction > 0:
            component["weight"] = self.TARGET_VALUE_DIRECTIONAL_BONUS
            component["reason"] = "directional_progress"
        else:
            component["weight"] = self.TARGET_VALUE_WRONG_DIRECTION_PENALTY
            component["reason"] = "wrong_direction"
        return component

    def _previous_confident_success_numeric_guess(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> Optional[int]:
        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is None:
            return None
        if not previous_signal.result or not previous_signal.continued_turn:
            return None
        if previous_signal.target_player_id != signal.target_player_id:
            return None
        if previous_signal.guessed_card[0] != signal.guessed_card[0]:
            return None
        return numeric_card_value(previous_signal.guessed_card)

    def explain_guess_signals(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        game_state: GameState,
    ) -> List[Dict[str, Any]]:
        explanations: List[Dict[str, Any]] = []
        for player_id in sorted(guess_signals_by_player):
            for signal in guess_signals_by_player[player_id]:
                explanations.append(
                    self.explain_signal(
                        hypothesis_by_player,
                        game_state,
                        signal,
                    )
                )
        return explanations

    def explain_signal(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        game_state: GameState,
        signal: GuessSignal,
    ) -> Dict[str, Any]:
        target_card = self._resolve_slot_card(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        base_explanation = {
            "action_index": signal.action_index,
            "guesser_id": signal.guesser_id,
            "target_player_id": signal.target_player_id,
            "target_slot_index": signal.target_slot_index,
            "guessed_card": serialize_card(signal.guessed_card),
            "result": signal.result,
            "continued_turn": signal.continued_turn,
        }
        if target_card is None:
            return {
                **base_explanation,
                "hypothesis_target_card": None,
                "total_weight": 1.0,
                "component_weights": {},
                "value_selection": {
                    "total_weight": 1.0,
                    "signal_tags": [],
                    "dominant_signal": {
                        "source": "neutral",
                        "reason": "neutral",
                        "weight": 1.0,
                    },
                },
            }

        guesser_cards = self._player_cards(
            game_state,
            signal.guesser_id,
            hypothesis_by_player.get(signal.guesser_id, {}),
        )
        self_hand_weight = self._score_self_hand(guesser_cards, signal.guessed_card)
        target_player_weight = self._score_target_player_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        target_slot_selection_weight = self._score_target_slot_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        value_selection = self._value_selection_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )
        target_slot_weight = self._score_target_slot(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
        )
        continue_weight = self._score_continue_decision(
            game_state,
            hypothesis_by_player,
            signal,
        )
        component_weights = {
            "self_hand": self_hand_weight,
            "target_player_selection": target_player_weight,
            "target_slot_selection": target_slot_selection_weight,
            "target_value_selection": value_selection["total_weight"],
            "target_slot_fit": target_slot_weight,
            "continue_decision_fit": continue_weight,
        }
        total_weight = 1.0
        for weight in component_weights.values():
            total_weight *= weight

        return {
            **base_explanation,
            "hypothesis_target_card": serialize_card(target_card),
            "total_weight": max(self.EPSILON, total_weight),
            "component_weights": component_weights,
            "value_selection": value_selection,
        }

    def describe_candidate_value_signal(
        self,
        *,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, Any]:
        guessed_numeric = numeric_card_value(guessed_card)
        if guessed_numeric is None:
            neutral = {"weight": 1.0, "reason": "neutral"}
            return {
                "signal_tags": [],
                "dominant_signal": {
                    "source": "neutral",
                    "reason": "neutral",
                    "weight": 1.0,
                },
                "progressive": neutral,
                "anchor": neutral,
                "boundary": neutral,
            }

        guesser_cards = self._player_cards(
            game_state,
            guesser_id,
            hypothesis_by_player.get(guesser_id, {}),
        )
        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            target_player_id,
            target_slot_index,
        )
        progressive = self._candidate_progressive_probe_component(
            game_state,
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
            guessed_numeric=guessed_numeric,
        )
        anchor = self._same_color_anchor_value_component(
            guessed_numeric,
            self._same_color_numeric_values(guesser_cards, guessed_card[0]),
        )
        boundary = self._local_boundary_value_component(
            guessed_numeric,
            low,
            high,
            width,
        )
        signal_tags = [
            component["reason"]
            for component in (progressive, anchor, boundary)
            if component["reason"] != "neutral"
        ]
        return {
            "signal_tags": signal_tags,
            "dominant_signal": self._dominant_signal(
                {
                    "progressive": progressive,
                    "same_color_anchor": anchor,
                    "local_boundary": boundary,
                }
            ),
            "progressive": progressive,
            "anchor": anchor,
            "boundary": boundary,
        }

    def _same_color_anchor_value_component(
        self,
        guessed_numeric: int,
        same_color_values: Sequence[int],
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "same_color_values": list(same_color_values),
            "left_anchor": None,
            "right_anchor": None,
            "anchor_gap": None,
            "nearest_anchor_distance": None,
        }
        if not same_color_values:
            return component

        nearest_anchor_distance = min(abs(guessed_numeric - value) for value in same_color_values)
        component["nearest_anchor_distance"] = nearest_anchor_distance
        if nearest_anchor_distance >= 4:
            component["weight"] *= 0.97
            component["reason"] = "far_from_same_color_anchor"

        left_anchor = max((value for value in same_color_values if value < guessed_numeric), default=None)
        right_anchor = min((value for value in same_color_values if value > guessed_numeric), default=None)
        component["left_anchor"] = left_anchor
        component["right_anchor"] = right_anchor

        if left_anchor is None or right_anchor is None:
            return component

        anchor_gap = right_anchor - left_anchor
        component["anchor_gap"] = anchor_gap
        midpoint = (left_anchor + right_anchor) / 2.0
        if anchor_gap == 2 and guessed_numeric == left_anchor + 1:
            component["weight"] *= self.TARGET_VALUE_SANDWICH_EXACT_BONUS
            component["reason"] = "same_color_sandwich_exact"
        elif 2 < anchor_gap <= 4 and left_anchor < guessed_numeric < right_anchor:
            component["weight"] *= self.TARGET_VALUE_SANDWICH_FILL_BONUS
            component["reason"] = "same_color_gap_fill"
        elif anchor_gap >= 6 and abs(guessed_numeric - midpoint) <= 1.0:
            component["weight"] *= self.TARGET_VALUE_WIDE_GAP_CENTER_BONUS
            component["reason"] = "same_color_wide_gap_center"
        return component

    def _local_boundary_value_component(
        self,
        guessed_numeric: int,
        low: Optional[int],
        high: Optional[int],
        width: int,
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "low": low,
            "high": high,
            "width": width,
        }
        if low is None and high is None:
            return component

        if low is not None and high is not None:
            if width <= 3 and guessed_numeric in {low + 1, high - 1}:
                component["weight"] *= self.TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS
                component["reason"] = "narrow_boundary_probe"
            elif width >= 6:
                midpoint = (low + high) / 2.0
                component["midpoint"] = midpoint
                if abs(guessed_numeric - midpoint) <= 1.0:
                    component["weight"] *= self.TARGET_VALUE_WIDE_GAP_CENTER_BONUS
                    component["reason"] = "wide_gap_center_probe"
                elif guessed_numeric in {low + 1, high - 1}:
                    component["weight"] *= self.TARGET_VALUE_WIDE_GAP_EDGE_PENALTY
                    component["reason"] = "wide_gap_edge_hug"
            return component

        if low is not None and guessed_numeric == low + 1:
            component["weight"] *= self.TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS
            component["reason"] = "single_sided_low_boundary_probe"
        if high is not None and guessed_numeric == high - 1:
            component["weight"] *= self.TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS
            component["reason"] = "single_sided_high_boundary_probe"
        return component

    def _dominant_signal(
        self,
        components_by_source: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        dominant_source = "neutral"
        dominant_reason = "neutral"
        dominant_weight = 1.0
        dominant_strength = 0.0

        for source, component in components_by_source.items():
            weight = float(component.get("weight", 1.0))
            strength = abs(weight - 1.0)
            if strength <= dominant_strength + self.EPSILON:
                continue
            dominant_source = source
            dominant_reason = str(component.get("reason", "neutral"))
            dominant_weight = weight
            dominant_strength = strength

        return {
            "source": dominant_source,
            "reason": dominant_reason,
            "weight": dominant_weight,
        }

    def _candidate_progressive_probe_component(
        self,
        game_state: GameState,
        *,
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_numeric: int,
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "latest_failed_value": None,
        }
        latest_failed = self._latest_failed_numeric_guess_on_slot(
            game_state,
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
        )
        if latest_failed is None:
            return component

        component["latest_failed_value"] = latest_failed
        delta = guessed_numeric - latest_failed
        if abs(delta) == 1:
            component["weight"] = 1.05
            component["reason"] = "retry_step_probe"
        elif abs(delta) == 2:
            component["weight"] = 1.02
            component["reason"] = "retry_directional_probe"
        elif delta == 0:
            component["weight"] = 0.97
            component["reason"] = "retry_stalled_probe"
        return component

    def _same_color_numeric_values(
        self,
        player_cards: Sequence[Card],
        color: str,
    ) -> List[int]:
        values = sorted(
            numeric_card_value(card)
            for card in player_cards
            if card[0] == color and numeric_card_value(card) is not None
        )
        return [value for value in values if value is not None]

    def _score_continue_decision(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        if not signal.result or signal.continued_turn is None:
            return 1.0

        attackability = self.estimate_attackability(
            game_state,
            hypothesis_by_player,
            acting_player_id=signal.guesser_id,
            exclude_slot=slot_key(signal.target_player_id, signal.target_slot_index),
        )
        if signal.continued_turn:
            if attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD:
                return self.CONTINUE_HIGH_ATTACKABILITY_BONUS
            return self.CONTINUE_LOW_ATTACKABILITY_PENALTY

        if attackability < self.ATTACKABILITY_TIGHT_THRESHOLD:
            return self.STOP_LOW_ATTACKABILITY_BONUS
        return self.STOP_HIGH_ATTACKABILITY_PENALTY

    def _player_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best = 0.0
        for slot in game_state.resolved_ordered_slots(player_id):
            if slot.known_card() is not None:
                continue
            key = slot_key(player_id, slot.slot_index)
            if exclude_slot is not None and key == exclude_slot:
                continue
            best = max(
                best,
                self._slot_attackability(
                    game_state,
                    hypothesis_by_player,
                    player_id,
                    slot.slot_index,
                ),
            )
        return best

    def _slot_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> float:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return 0.0

        if slot.known_card() is not None:
            return 0.0

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            player_id,
            slot_index,
        )
        color_bonus = 1.0
        if getattr(slot, "color", None) is not None:
            color_bonus = 1.12

        confidence = color_bonus / max(1.0, float(width + 1))
        if low is not None and high is not None and width <= 2:
            confidence *= 1.08
        return confidence

    def _previous_guess_signal(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> Optional[GuessSignal]:
        previous: Optional[GuessSignal] = None
        guess_action_index = 0

        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            if guess_action_index >= signal.action_index:
                break

            guessed_card = action.guessed_card()
            guesser_id = getattr(action, "guesser_id", None)
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if (
                guessed_card is not None
                and guesser_id == signal.guesser_id
                and target_player_id is not None
                and target_slot_index is not None
            ):
                previous = GuessSignal(
                    action_index=guess_action_index,
                    guesser_id=guesser_id,
                    target_player_id=target_player_id,
                    target_slot_index=target_slot_index,
                    guessed_card=guessed_card,
                    result=bool(getattr(action, "result", False)),
                    continued_turn=getattr(action, "continued_turn", None),
                )

            guess_action_index += 1

        return previous

    def _has_prior_failed_guess_on_slot(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> bool:
        return bool(self._prior_failed_numeric_guesses_on_slot(game_state, signal))

    def _prior_failed_numeric_guesses_on_slot(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> List[int]:
        failed_values: List[int] = []
        guess_action_index = 0
        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            if guess_action_index >= signal.action_index:
                break

            if (
                getattr(action, "guesser_id", None) == signal.guesser_id
                and getattr(action, "target_player_id", None) == signal.target_player_id
                and getattr(action, "target_slot_index", None) == signal.target_slot_index
                and not bool(getattr(action, "result", False))
            ):
                guessed_card = action.guessed_card()
                guessed_numeric = numeric_card_value(guessed_card)
                if guessed_numeric is not None:
                    failed_values.append(guessed_numeric)

            guess_action_index += 1
        return failed_values

    def _latest_failed_numeric_guess_on_slot(
        self,
        game_state: GameState,
        *,
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
    ) -> Optional[int]:
        latest_failed: Optional[int] = None
        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            if (
                getattr(action, "guesser_id", None) == guesser_id
                and getattr(action, "target_player_id", None) == target_player_id
                and getattr(action, "target_slot_index", None) == target_slot_index
                and not bool(getattr(action, "result", False))
            ):
                guessed_numeric = numeric_card_value(action.guessed_card())
                if guessed_numeric is not None:
                    latest_failed = guessed_numeric
        return latest_failed

    def _slot_numeric_interval(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Tuple[Optional[int], Optional[int], int]:
        slots = game_state.resolved_ordered_slots(player_id)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return None, None, MAX_CARD_VALUE

        lower: Optional[int] = None
        upper: Optional[int] = None

        cursor = order_index - 1
        while cursor >= 0:
            left_card = self._resolve_slot_card(game_state, hypothesis_by_player, player_id, slots[cursor].slot_index)
            left_value = numeric_card_value(left_card)
            if left_value is not None:
                lower = left_value
                break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(slots):
            right_card = self._resolve_slot_card(game_state, hypothesis_by_player, player_id, slots[cursor].slot_index)
            right_value = numeric_card_value(right_card)
            if right_value is not None:
                upper = right_value
                break
            cursor += 1

        effective_low = 0 if lower is None else lower
        effective_high = MAX_CARD_VALUE if upper is None else upper
        if effective_high < effective_low:
            effective_high = effective_low
        width = max(0, effective_high - effective_low)
        return lower, upper, width

    def _resolve_slot_card(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Optional[Card]:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return None

        known_card = slot.known_card()
        if known_card is not None:
            return known_card
        return hypothesis_by_player.get(player_id, {}).get(slot_index)

class DaVinciInferenceEngine:
    """Infer hard posterior, soft posterior, and blended posterior for all hidden slots."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_player_ids = [
            player_id
            for player_id in game_state.inference_player_ids()
            if game_state.resolved_ordered_slots(player_id)
        ]
        self.player_slots = {
            player_id: game_state.resolved_ordered_slots(player_id)
            for player_id in self.inference_player_ids
        }
        self.hard_constraints = HardConstraintCompiler(game_state).compile()
        self.preassigned_hidden: Dict[str, Dict[int, Card]] = {
            player_id: {} for player_id in self.inference_player_ids
        }
        self.search_positions: List[HiddenPosition] = []
        self.position_by_key: Dict[SlotKey, HiddenPosition] = {}
        self.player_slot_order: Dict[str, Dict[int, int]] = {}
        self.publicly_collapsed_slots: Set[SlotKey] = set()

        self._build_slot_views()
        self.available_cards = tuple(self._get_available_cards())
        self.base_domains = self._build_base_domains()
        self._validate_initial_domains()

    def _build_all_possible_cards(self) -> Tuple[Card, ...]:
        ordered_cards = [
            (color, value)
            for value in range(MAX_CARD_VALUE + 1)
            for color in CARD_COLORS
        ]
        ordered_cards.extend((color, JOKER) for color in CARD_COLORS)
        return tuple(ordered_cards)

    def _build_slot_views(self) -> None:
        for player_id, slots in self.player_slots.items():
            order_map: Dict[int, int] = {}
            for order_index, slot in enumerate(slots):
                order_map[slot.slot_index] = order_index
                key = slot_key(player_id, slot.slot_index)
                forced = self.hard_constraints.fixed_by_slot.get(key)
                known = slot.known_card()

                if known is not None:
                    if forced is not None and forced != known:
                        raise ValueError(
                            f"Slot {key} conflicts between state {known} and action evidence {forced}"
                        )
                    continue

                if forced is not None:
                    self.preassigned_hidden[player_id][slot.slot_index] = forced
                    self.publicly_collapsed_slots.add(key)
                    continue

                if slot.value is None:
                    hidden = HiddenPosition(
                        player_id=player_id,
                        slot_index=slot.slot_index,
                        order_index=order_index,
                        color=getattr(slot, "color", None),
                    )
                    self.search_positions.append(hidden)
                    self.position_by_key[key] = hidden
            self.player_slot_order[player_id] = order_map

    def _get_available_cards(self) -> Tuple[Card, ...]:
        known_cards = set(self.game_state.known_cards())
        fixed_hidden_cards = {
            card
            for cards_by_slot in self.preassigned_hidden.values()
            for card in cards_by_slot.values()
        }
        used_cards = known_cards | fixed_hidden_cards
        return tuple(
            card
            for card in self._build_all_possible_cards()
            if card not in used_cards
        )

    def _build_base_domains(self) -> Dict[SlotKey, Tuple[Card, ...]]:
        base_domains: Dict[SlotKey, Tuple[Card, ...]] = {}
        for position in self.search_positions:
            key = slot_key(position.player_id, position.slot_index)
            forbidden = self.hard_constraints.forbidden_by_slot.get(key, set())
            domain = []
            for card in self.available_cards:
                if position.color is not None and card[0] != position.color:
                    continue
                if card in forbidden:
                    continue
                domain.append(card)
            base_domains[key] = tuple(domain)
        return base_domains

    def _validate_initial_domains(self) -> None:
        if len(set(self.available_cards)) != len(self.available_cards):
            raise ValueError("Available card pool contains duplicates.")
        for key, domain in self.base_domains.items():
            if not domain:
                raise ValueError(f"Hidden slot {key} has empty hard domain.")

    def infer_hidden_probabilities(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        behavior_model: BehavioralLikelihoodModel,
    ) -> Tuple[
        FullProbabilityMatrix,
        FullProbabilityMatrix,
        FullProbabilityMatrix,
        int,
        float,
    ]:
        if not self.search_positions and not any(self.preassigned_hidden.values()):
            return {}, {}, {}, 0, 0.0

        hard_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }
        soft_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }

        legal_world_count = 0
        total_soft_weight = 0.0
        current_assignment: Dict[str, Dict[int, Card]] = {
            player_id: dict(cards_by_slot)
            for player_id, cards_by_slot in self.preassigned_hidden.items()
        }
        dead_state_cache: Set[Tuple[Tuple[Card, ...], Tuple[Tuple[str, Tuple[Optional[Card], ...]], ...]]] = set()

        def dfs(remaining_positions: Tuple[HiddenPosition, ...], remaining_cards: Tuple[Card, ...]) -> bool:
            nonlocal legal_world_count, total_soft_weight

            state_signature = self._state_signature(current_assignment, remaining_cards)
            if state_signature in dead_state_cache:
                return False

            if not remaining_positions:
                legal_world_count += 1
                hypothesis_by_player = {
                    player_id: dict(cards_by_slot)
                    for player_id, cards_by_slot in current_assignment.items()
                    if cards_by_slot
                }
                soft_weight = behavior_model.score_hypothesis(
                    hypothesis_by_player,
                    guess_signals_by_player,
                    self.game_state,
                )
                total_soft_weight += soft_weight
                self._accumulate_world(
                    hard_position_weights,
                    soft_position_weights,
                    current_assignment,
                    soft_weight,
                )
                return True

            feasible_domains = self._feasible_domains(remaining_positions, remaining_cards, current_assignment)
            if any(len(domain) == 0 for _, domain in feasible_domains):
                dead_state_cache.add(state_signature)
                return False

            next_position, next_domain = min(feasible_domains, key=lambda item: (len(item[1]), item[0].player_id, item[0].slot_index))
            next_remaining_positions = tuple(
                position for position in remaining_positions if position != next_position
            )
            found_any_solution = False

            for candidate in next_domain:
                current_assignment[next_position.player_id][next_position.slot_index] = candidate
                next_remaining_cards = tuple(card for card in remaining_cards if card != candidate)
                found_any_solution = dfs(next_remaining_positions, next_remaining_cards) or found_any_solution
                del current_assignment[next_position.player_id][next_position.slot_index]

            if not found_any_solution:
                dead_state_cache.add(state_signature)
            return found_any_solution

        dfs(tuple(self.search_positions), self.available_cards)

        if legal_world_count == 0:
            return {}, {}, {}, 0, 0.0

        hard_matrix = self._normalize_matrix(hard_position_weights, legal_world_count)
        soft_matrix = (
            self._normalize_matrix(soft_position_weights, total_soft_weight)
            if total_soft_weight > 0
            else hard_matrix
        )
        blended_matrix = self._blend_probability_matrices(hard_matrix, soft_matrix)
        return hard_matrix, soft_matrix, blended_matrix, legal_world_count, total_soft_weight

    def _state_signature(
        self,
        current_assignment: Dict[str, Dict[int, Card]],
        remaining_cards: Tuple[Card, ...],
    ) -> Tuple[Tuple[Card, ...], Tuple[Tuple[str, Tuple[Optional[Card], ...]], ...]]:
        player_signatures: List[Tuple[str, Tuple[Optional[Card], ...]]] = []
        for player_id in sorted(self.player_slots):
            signature: List[Optional[Card]] = []
            for slot in self.player_slots[player_id]:
                known = slot.known_card()
                if known is not None:
                    signature.append(known)
                    continue
                signature.append(current_assignment[player_id].get(slot.slot_index))
            player_signatures.append((player_id, tuple(signature)))
        return tuple(sorted(remaining_cards, key=card_sort_key)), tuple(player_signatures)

    def _feasible_domains(
        self,
        remaining_positions: Tuple[HiddenPosition, ...],
        remaining_cards: Tuple[Card, ...],
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> List[Tuple[HiddenPosition, Tuple[Card, ...]]]:
        remaining_card_set = set(remaining_cards)
        feasible: List[Tuple[HiddenPosition, Tuple[Card, ...]]] = []
        for position in remaining_positions:
            key = slot_key(position.player_id, position.slot_index)
            candidates = []
            for candidate in self.base_domains[key]:
                if candidate not in remaining_card_set:
                    continue
                if not self._fits_local_order(position, candidate, current_assignment):
                    continue
                candidates.append(candidate)
            candidates.sort(key=card_sort_key)
            feasible.append((position, tuple(candidates)))
        return feasible

    def _fits_local_order(
        self,
        position: HiddenPosition,
        candidate: Card,
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> bool:
        if candidate[1] == JOKER:
            return True

        left_bound = self._nearest_numeric_bound(position.player_id, position.order_index, -1, current_assignment)
        if left_bound is not None and self._card_precedes(candidate, left_bound):
            return False

        right_bound = self._nearest_numeric_bound(position.player_id, position.order_index, 1, current_assignment)
        if right_bound is not None and self._card_precedes(right_bound, candidate):
            return False

        return True

    def _nearest_numeric_bound(
        self,
        player_id: str,
        order_index: int,
        direction: int,
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> Optional[Card]:
        slots = self.player_slots[player_id]
        cursor = order_index + direction
        while 0 <= cursor < len(slots):
            slot = slots[cursor]
            card = slot.known_card()
            if card is None:
                card = current_assignment[player_id].get(slot.slot_index)
            if card is not None and card[1] != JOKER:
                return card
            cursor += direction
        return None

    def _card_precedes(self, left: Card, right: Card) -> bool:
        if left[1] == JOKER or right[1] == JOKER:
            return False
        if left[1] != right[1]:
            return int(left[1]) < int(right[1])
        return CARD_COLOR_RANK[left[0]] < CARD_COLOR_RANK[right[0]]

    def _accumulate_world(
        self,
        hard_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        soft_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        current_assignment: Dict[str, Dict[int, Card]],
        soft_weight: float,
    ) -> None:
        for player_id, cards_by_slot in current_assignment.items():
            for slot_index, card in cards_by_slot.items():
                hard_position_weights[player_id][slot_index][card] += 1.0
                soft_position_weights[player_id][slot_index][card] += soft_weight

    def _normalize_matrix(
        self,
        position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        total_weight: float,
    ) -> FullProbabilityMatrix:
        if total_weight <= 0:
            return {}
        result: FullProbabilityMatrix = {}
        for player_id, weighted_slots in position_weights.items():
            if not weighted_slots:
                continue
            result[player_id] = {
                slot_index: {
                    card: weight / total_weight
                    for card, weight in weighted_cards.items()
                }
                for slot_index, weighted_cards in weighted_slots.items()
            }
        return result

    def _blend_probability_matrices(
        self,
        hard_matrix: FullProbabilityMatrix,
        soft_matrix: FullProbabilityMatrix,
    ) -> FullProbabilityMatrix:
        blended: FullProbabilityMatrix = {}
        for player_id in sorted(set(hard_matrix) | set(soft_matrix)):
            player_positions: ProbabilityMatrix = {}
            all_slot_indices = set(hard_matrix.get(player_id, {})) | set(soft_matrix.get(player_id, {}))
            for slot_index in sorted(all_slot_indices):
                hard_dist = hard_matrix.get(player_id, {}).get(slot_index, {})
                soft_dist = soft_matrix.get(player_id, {}).get(slot_index, hard_dist)
                cards = set(hard_dist) | set(soft_dist)
                blended_weights = {
                    card: ((1.0 - SOFT_BEHAVIOR_BLEND) * hard_dist.get(card, 0.0))
                    + (SOFT_BEHAVIOR_BLEND * soft_dist.get(card, 0.0))
                    for card in cards
                }
                player_positions[slot_index] = normalize_card_distribution(blended_weights)
            if player_positions:
                blended[player_id] = player_positions
        return blended


class DaVinciDecisionEngine:
    """Score moves by immediate EV, continuation likelihood, and stop-aware thresholding."""

    HIT_REWARD = 10.0
    MISS_BASE = 12.0
    MISS_ENDGAME_MULTIPLIER = 12.0
    INFORMATION_GAIN_WEIGHT = 0.45
    CONTINUATION_DISCOUNT = 0.72
    MAX_CANDIDATES_PER_SLOT = 8
    MIN_CANDIDATE_PROBABILITY = 1e-9

    STOP_MARGIN_BASE = 0.18
    STOP_MARGIN_RISK_SCALE = 0.035
    STOP_MARGIN_SHORT_HAND = 0.22
    STOP_MARGIN_LOW_CONFIDENCE = 0.28
    STOP_MARGIN_WEAK_CONTINUATION = 0.16
    STOP_MARGIN_WEAK_EDGE = 0.22
    STOP_MARGIN_WEAK_ROLLOUT = 0.24
    STOP_MARGIN_FRAGILE_POST_HIT = 0.18
    STOP_MARGIN_TOP_K_SUPPORT = 0.18
    STOP_MARGIN_LOW_ATTACKABILITY = 0.18
    STOP_MARGIN_BEHAVIOR_ROLLOUT = 0.16
    STOP_EDGE_REFERENCE = 0.18
    ROLLOUT_MARGIN_REFERENCE = 0.40
    POST_HIT_GAP_REFERENCE = 0.22
    POST_HIT_TOP_K_COUNT = 3
    CONTINUATION_TOP_K_BLEND = 0.38
    LOW_CONFIDENCE_GUARD_MARGIN = 0.22
    WEAK_EDGE_GUARD_MARGIN = 0.18
    ATTACKABILITY_REFERENCE = BehavioralLikelihoodModel.ATTACKABILITY_TIGHT_THRESHOLD
    DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER = 1.0
    DEFAULT_BEHAVIOR_MATCH_MULTIPLIER = 1.0
    BEHAVIOR_MATCH_CONTEXT_TOP_K = 3
    BEHAVIOR_MATCH_DECISION_SCALE = 0.45
    BEHAVIOR_MATCH_SUPPORT_REFERENCE = 0.10
    BEHAVIOR_MATCH_CONTEXT_COUNT_REFERENCE = 4.0
    BEHAVIOR_MATCH_COMPONENT_WEIGHT_REFERENCE = 0.15
    BEHAVIOR_MATCH_COMPONENT_PENALTY_WEIGHT = 0.40
    BEHAVIOR_MATCH_NET_STRUCTURE_SCALE = 0.10
    BEHAVIOR_MATCH_DECISION_WINDOW = 0.12
    BEHAVIOR_MATCH_DECISION_NET_STRUCTURE_SCALE = 0.50
    POST_HIT_BEHAVIOR_SUPPORT_SCALE = 0.08
    POST_HIT_BEHAVIOR_SUPPORT_REFERENCE = 0.20

    def calculate_risk_factor(self, my_hidden_count: int) -> float:
        exposure = 1.0 / max(1, my_hidden_count)
        return self.MISS_BASE + self.MISS_ENDGAME_MULTIPLIER * exposure

    def evaluate_all_moves(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        hidden_index_by_player: Dict[str, Dict[int, int]],
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        behavior_guidance_profile: Optional[Dict[str, float]] = None,
        game_state: Optional[GameState] = None,
        behavior_map_hypothesis: Optional[Dict[str, Dict[int, Card]]] = None,
        blocked_slots: Optional[Set[SlotKey]] = None,
        rollout_depth: int = 1,
    ) -> Tuple[List[Dict[str, Any]], float]:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        blocked_slots = blocked_slots or set()
        moves: List[Dict[str, Any]] = []

        for player_id in sorted(full_probability_matrix):
            probability_matrix = full_probability_matrix[player_id]
            hidden_index_by_slot = hidden_index_by_player.get(player_id, {})
            for target_slot_index, slot_distribution in probability_matrix.items():
                if slot_key(player_id, target_slot_index) in blocked_slots:
                    continue
                ranked_candidates = sorted(
                    slot_distribution.items(),
                    key=lambda item: (-item[1], card_sort_key(item[0])),
                )
                for card, probability in ranked_candidates[: self.MAX_CANDIDATES_PER_SLOT]:
                    if probability <= self.MIN_CANDIDATE_PROBABILITY:
                        continue
                    move = self._score_single_move(
                        full_probability_matrix=full_probability_matrix,
                        my_hidden_count=my_hidden_count,
                        risk_factor=risk_factor,
                        hidden_index_by_slot=hidden_index_by_slot,
                        player_id=player_id,
                        slot_index=target_slot_index,
                        card=card,
                        probability=probability,
                        slot_distribution=slot_distribution,
                        behavior_model=behavior_model,
                        guess_signals_by_player=guess_signals_by_player,
                        acting_player_id=acting_player_id,
                        behavior_guidance_profile=behavior_guidance_profile,
                        game_state=game_state,
                        behavior_map_hypothesis=behavior_map_hypothesis,
                        rollout_depth=rollout_depth,
                    )
                    moves.append(move)

        moves.sort(
            key=lambda move: (
                -move.get("ranking_score", move["expected_value"]),
                -move["win_probability"],
                -move["continuation_value"],
                -move["continuation_likelihood"],
                move["target_player_id"],
                move["target_slot_index"],
                card_sort_key((move["guess_card"][0], move["guess_card"][1])),
            )
        )
        return moves, risk_factor

    def choose_best_move(
        self,
        all_moves: List[Dict[str, Any]],
        *,
        risk_factor: float,
        my_hidden_count: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not all_moves:
            stop_threshold = self._stop_threshold(
                risk_factor=risk_factor,
                my_hidden_count=my_hidden_count,
                best_move=None,
            )
            return None, {
                "evaluated_move_count": 0,
                "best_immediate_value": 0.0,
                "best_expected_value": 0.0,
                "best_win_probability": 0.0,
                "best_continuation_value": 0.0,
                "best_continuation_likelihood": 0.0,
                "best_behavior_guidance_multiplier": self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                "best_behavior_guidance_signal_count": 0.0,
                "best_behavior_guidance_support": 0.0,
                "best_behavior_guidance_stable_ratio": 0.0,
                "best_ranking_score": 0.0,
                "best_behavior_match_multiplier": self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER,
                "best_behavior_match_bonus": 0.0,
                "best_behavior_match_support": 0.0,
                "best_behavior_match_decision_bonus": 0.0,
                "best_behavior_match_decision_structure_adjustment": 0.0,
                "best_behavior_match_ranking_bonus": 0.0,
                "best_behavior_match_net_structure": 0.0,
                "best_behavior_match_structure_adjustment": 0.0,
                "best_behavior_match_candidate_confidence": 0.0,
                "best_behavior_match_component_support": 0.0,
                "best_behavior_match_component_strength": 0.0,
                "best_behavior_match_component_penalty": 0.0,
                "best_behavior_match_context_focus": 0.0,
                "best_behavior_rollout_pressure": 0.0,
                "best_post_hit_behavior_support_adjustment": 0.0,
                "best_post_hit_behavior_support_gain": 0.0,
                "best_post_hit_behavior_fragility_drag": 0.0,
                "best_post_hit_continue_score": 0.0,
                "best_post_hit_stop_score": 0.0,
                "best_post_hit_continue_margin": 0.0,
                "best_post_hit_best_gap": 0.0,
                "best_post_hit_guidance_multiplier": self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                "best_post_hit_guidance_support": 0.0,
                "best_post_hit_guidance_stable_ratio": 0.0,
                "best_post_hit_guidance_signal_count": 0.0,
                "best_post_hit_guidance_rebuild_applied": False,
                "best_post_hit_guidance_rebuild_signal_count": 0.0,
                "best_post_hit_guidance_augmented_slot_count": 0.0,
                "best_post_hit_guidance_multiplier_delta": 0.0,
                "best_post_hit_guidance_source_shift": "neutral",
                "best_post_hit_guidance_source_shift_strength": 0.0,
                "best_post_hit_guidance_rebuild_reason": "post-hit rebuild unavailable",
                "best_post_hit_top_k_expected_continue_margin": 0.0,
                "best_post_hit_top_k_continue_margin": 0.0,
                "best_post_hit_top_k_expected_support_ratio": 0.0,
                "best_post_hit_top_k_support_ratio": 0.0,
                "stop_threshold": stop_threshold,
                "stop_score": stop_threshold,
                "continue_score": 0.0,
                "continue_margin": -stop_threshold,
                "recommend_stop": True,
                "decision_score_breakdown": {
                    "base_stop_threshold": stop_threshold,
                    "edge_pressure": 0.0,
                    "rollout_pressure": 0.0,
                    "fragile_rollout_pressure": 0.0,
                    "top_k_rollout_pressure": 0.0,
                    "attackability_pressure": 0.0,
                    "behavior_rollout_pressure": 0.0,
                    "post_hit_behavior_support_adjustment": 0.0,
                    "post_hit_behavior_support_gain": 0.0,
                    "post_hit_behavior_fragility_drag": 0.0,
                    "post_hit_behavior_support_signal": 0.0,
                    "post_hit_behavior_fragility_signal": 0.0,
                    "post_hit_behavior_support_strength": 0.0,
                    "post_hit_behavior_fragility_strength": 0.0,
                    "behavior_match_decision_bonus": 0.0,
                    "behavior_match_decision_structure_adjustment": 0.0,
                    "behavior_match_net_structure": 0.0,
                    "behavior_match_ranking_bonus": 0.0,
                    "behavior_match_structure_adjustment": 0.0,
                    "behavior_match_candidate_confidence": 0.0,
                    "behavior_match_component_support": 0.0,
                    "behavior_match_component_strength": 0.0,
                    "behavior_match_component_penalty": 0.0,
                    "behavior_match_context_focus": 0.0,
                },
                "stop_reason": "没有可评估的候选动作。",
            }

        best_move = all_moves[0]
        second_move = all_moves[1] if len(all_moves) > 1 else None
        stop_threshold = self._stop_threshold(
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
            best_move=best_move,
        )
        decision_snapshot = self._evaluate_continue_decision(
            best_move=best_move,
            second_move=second_move,
            stop_threshold=stop_threshold,
            my_hidden_count=my_hidden_count,
        )
        stop_reason = self._build_stop_reason(
            should_continue=decision_snapshot["should_continue"],
            best_move=best_move,
            stop_threshold=stop_threshold,
            stop_score=decision_snapshot["stop_score"],
            continue_score=decision_snapshot["continue_score"],
            continue_margin=decision_snapshot["continue_margin"],
            low_confidence_guard=decision_snapshot["low_confidence_guard"],
            weak_edge_guard=decision_snapshot["weak_edge_guard"],
        )

        decision_summary = {
            "evaluated_move_count": len(all_moves),
            "best_immediate_value": best_move.get("immediate_expected_value", best_move["expected_value"]),
            "best_expected_value": best_move["expected_value"],
            "best_win_probability": best_move["win_probability"],
            "best_continuation_value": best_move.get("continuation_value", 0.0),
            "best_continuation_likelihood": best_move.get("continuation_likelihood", 0.0),
            "best_behavior_guidance_multiplier": best_move.get("behavior_guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER),
            "best_behavior_guidance_signal_count": best_move.get("behavior_guidance_signal_count", 0.0),
            "best_behavior_guidance_support": best_move.get("behavior_guidance_support", 0.0),
            "best_behavior_guidance_stable_ratio": best_move.get("behavior_guidance_stable_ratio", 0.0),
            "best_ranking_score": best_move.get("ranking_score", best_move["expected_value"]),
            "best_behavior_match_multiplier": best_move.get("behavior_match_multiplier", self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER),
            "best_behavior_match_bonus": best_move.get("behavior_match_bonus", 0.0),
            "best_behavior_match_support": best_move.get("behavior_match_support", 0.0),
            "best_behavior_match_decision_bonus": decision_snapshot["behavior_match_decision_bonus"],
            "best_behavior_match_decision_structure_adjustment": decision_snapshot["behavior_match_decision_structure_adjustment"],
            "best_behavior_match_ranking_bonus": best_move.get("behavior_match_ranking_bonus", 0.0),
            "best_behavior_match_net_structure": decision_snapshot["behavior_match_net_structure"],
            "best_behavior_match_structure_adjustment": best_move.get("behavior_match_structure_adjustment", 0.0),
            "best_behavior_match_candidate_confidence": decision_snapshot["behavior_match_candidate_confidence"],
            "best_behavior_match_component_support": decision_snapshot["behavior_match_component_support"],
            "best_behavior_match_component_strength": decision_snapshot["behavior_match_component_strength"],
            "best_behavior_match_component_penalty": decision_snapshot["behavior_match_component_penalty"],
            "best_behavior_match_context_focus": decision_snapshot["behavior_match_context_focus"],
            "best_behavior_rollout_pressure": decision_snapshot["decision_score_breakdown"]["behavior_rollout_pressure"],
            "best_post_hit_behavior_support_adjustment": decision_snapshot["decision_score_breakdown"]["post_hit_behavior_support_adjustment"],
            "best_post_hit_behavior_support_gain": decision_snapshot["decision_score_breakdown"]["post_hit_behavior_support_gain"],
            "best_post_hit_behavior_fragility_drag": decision_snapshot["decision_score_breakdown"]["post_hit_behavior_fragility_drag"],
            "best_attackability_after_hit": best_move.get("attackability_after_hit", 0.0),
            "best_post_hit_continue_score": best_move.get("post_hit_continue_score", 0.0),
            "best_post_hit_stop_score": best_move.get("post_hit_stop_score", 0.0),
            "best_post_hit_continue_margin": best_move.get("post_hit_continue_margin", 0.0),
            "best_post_hit_best_gap": best_move.get("post_hit_best_gap", 0.0),
            "best_post_hit_guidance_multiplier": best_move.get("post_hit_guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER),
            "best_post_hit_guidance_support": best_move.get("post_hit_guidance_support", 0.0),
            "best_post_hit_guidance_stable_ratio": best_move.get("post_hit_guidance_stable_ratio", 0.0),
            "best_post_hit_guidance_signal_count": best_move.get("post_hit_guidance_signal_count", 0.0),
            "best_post_hit_guidance_rebuild_applied": bool(
                best_move.get("post_hit_guidance_debug", {}).get("rebuild_applied", False)
            ),
            "best_post_hit_guidance_rebuild_signal_count": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("rebuilt_signal_count", 0.0),
            "best_post_hit_guidance_augmented_slot_count": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("augmented_known_slot_count", 0.0),
            "best_post_hit_guidance_multiplier_delta": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("blended_delta_from_base", {}).get("guidance_multiplier", 0.0),
            "best_post_hit_guidance_source_shift": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("blended_dominant_source_shift", "neutral"),
            "best_post_hit_guidance_source_shift_strength": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("blended_dominant_source_shift_strength", 0.0),
            "best_post_hit_guidance_rebuild_reason": self._build_post_hit_guidance_rebuild_reason(
                best_move.get("post_hit_guidance_debug", {})
            ),
            "best_post_hit_top_k_expected_continue_margin": best_move.get("post_hit_top_k_expected_continue_margin", 0.0),
            "best_post_hit_top_k_continue_margin": best_move.get("post_hit_top_k_continue_margin", 0.0),
            "best_post_hit_top_k_expected_support_ratio": best_move.get("post_hit_top_k_expected_support_ratio", 0.0),
            "best_post_hit_top_k_support_ratio": best_move.get("post_hit_top_k_support_ratio", 0.0),
            "best_gap": decision_snapshot["best_gap"],
            "stop_threshold": stop_threshold,
            "stop_score": decision_snapshot["stop_score"],
            "continue_score": decision_snapshot["continue_score"],
            "continue_margin": decision_snapshot["continue_margin"],
            "recommend_stop": not decision_snapshot["should_continue"],
            "decision_score_breakdown": decision_snapshot["decision_score_breakdown"],
            "stop_reason": stop_reason,
        }
        return (best_move if decision_snapshot["should_continue"] else None), decision_summary

    def _score_single_move(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        risk_factor: float,
        hidden_index_by_slot: Dict[int, int],
        player_id: str,
        slot_index: int,
        card: Card,
        probability: float,
        slot_distribution: Dict[Card, float],
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        behavior_guidance_profile: Optional[Dict[str, float]],
        game_state: Optional[GameState],
        behavior_map_hypothesis: Optional[Dict[str, Dict[int, Card]]],
        rollout_depth: int,
    ) -> Dict[str, Any]:
        info_gain = self._expected_slot_entropy_reduction(slot_distribution, card)
        continuation_value = 0.0
        continuation_likelihood = 0.0
        attackability_after_hit = 0.0
        history_continue_rate = behavior_model.CONTINUATION_PRIOR_BASE
        next_best_immediate_ev = 0.0
        post_hit_continuation_value = 0.0
        post_hit_continue_score = 0.0
        post_hit_stop_score = 0.0
        post_hit_continue_margin = 0.0
        post_hit_should_continue = False
        post_hit_best_gap = 0.0
        post_hit_gap_adjustment = 1.0
        post_hit_guidance_multiplier = self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER
        post_hit_guidance_support = 0.0
        post_hit_guidance_stable_ratio = 0.0
        post_hit_guidance_signal_count = 0.0
        post_hit_guidance_debug: Dict[str, Any] = self._default_post_hit_guidance_rebuild_debug(
            base_profile=behavior_guidance_profile,
            acting_player_id=acting_player_id,
        )
        post_hit_top_k_expected_continue_margin = 0.0
        post_hit_top_k_continue_margin = 0.0
        post_hit_top_k_expected_support_ratio = 0.0
        post_hit_top_k_support_ratio = 0.0
        post_hit_top_k_positive_count = 0.0
        post_hit_behavior_support_adjustment = 0.0
        post_hit_behavior_support_gain = 0.0
        post_hit_behavior_fragility_drag = 0.0
        post_hit_behavior_support_signal = 0.0
        post_hit_behavior_fragility_signal = 0.0
        post_hit_behavior_support_strength = 0.0
        post_hit_behavior_fragility_strength = 0.0
        behavior_guidance_multiplier = self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER
        behavior_guidance_support = 0.0
        behavior_guidance_stable_ratio = 0.0
        behavior_guidance_signal_count = 0.0
        behavior_match_multiplier = self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER
        behavior_match_bonus = 0.0
        behavior_match_ranking_bonus = 0.0
        behavior_match_net_structure = 0.0
        behavior_match_structure_adjustment = 0.0
        behavior_match_support = 0.0
        behavior_match_confidence_breakdown = {
            "candidate_confidence": 1.0,
            "component_support": 1.0,
            "component_strength": 1.0,
            "component_penalty": 0.0,
            "context_focus": 1.0,
        }
        behavior_candidate_signal = {
            "signal_tags": [],
            "dominant_signal": {
                "source": "neutral",
                "reason": "neutral",
                "weight": 1.0,
            },
        }

        success_matrix = self._success_posterior(full_probability_matrix, player_id, slot_index, card)
        if success_matrix:
            next_best_immediate_ev = self._best_immediate_ev(
                success_matrix,
                my_hidden_count,
            )
            continuation_assessment = behavior_model.estimate_continue_likelihood(
                success_matrix,
                guess_signals_by_player,
                acting_player_id,
                exclude_slot=slot_key(player_id, slot_index),
            )
            continuation_likelihood = continuation_assessment["continue_likelihood"]
            attackability_after_hit = continuation_assessment["attackability"]
            history_continue_rate = continuation_assessment["history_continue_rate"]
            if behavior_guidance_profile is not None:
                behavior_guidance_multiplier = max(
                    0.0,
                    float(behavior_guidance_profile.get("guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER)),
                )
                behavior_guidance_support = float(behavior_guidance_profile.get("average_posterior_support", 0.0))
                behavior_guidance_stable_ratio = float(behavior_guidance_profile.get("stable_signal_ratio", 0.0))
                behavior_guidance_signal_count = float(behavior_guidance_profile.get("signal_count", 0.0))
                continuation_likelihood = clamp(
                    continuation_likelihood * behavior_guidance_multiplier,
                    behavior_model.CONTINUATION_MIN,
                    behavior_model.CONTINUATION_MAX,
                )
            if rollout_depth > 0:
                post_hit_rollout = self._evaluate_post_hit_rollout(
                    success_matrix=success_matrix,
                    my_hidden_count=my_hidden_count,
                    behavior_model=behavior_model,
                    guess_signals_by_player=guess_signals_by_player,
                    acting_player_id=acting_player_id,
                    behavior_guidance_profile=behavior_guidance_profile,
                    game_state=game_state,
                    target_player_id=player_id,
                    target_slot_index=slot_index,
                    guessed_card=card,
                    rollout_depth=rollout_depth - 1,
                )
                post_hit_continue_score = post_hit_rollout["continue_score"]
                post_hit_stop_score = post_hit_rollout["stop_score"]
                post_hit_continue_margin = post_hit_rollout["continue_margin"]
                post_hit_should_continue = post_hit_rollout["should_continue"]
                post_hit_best_gap = post_hit_rollout["best_gap"]
                post_hit_guidance_multiplier = post_hit_rollout["behavior_guidance_multiplier"]
                post_hit_guidance_support = post_hit_rollout["behavior_guidance_support"]
                post_hit_guidance_stable_ratio = post_hit_rollout["behavior_guidance_stable_ratio"]
                post_hit_guidance_signal_count = post_hit_rollout["behavior_guidance_signal_count"]
                post_hit_guidance_debug = post_hit_rollout["guidance_debug"]
                post_hit_top_k_expected_continue_margin = post_hit_rollout["top_k_expected_continue_margin"]
                post_hit_top_k_continue_margin = post_hit_rollout["top_k_continue_margin"]
                post_hit_top_k_expected_support_ratio = post_hit_rollout["top_k_expected_support_ratio"]
                post_hit_top_k_support_ratio = post_hit_rollout["top_k_support_ratio"]
                post_hit_top_k_positive_count = post_hit_rollout["top_k_positive_count"]
                if post_hit_continue_margin > 0.0 and post_hit_best_gap < self.POST_HIT_GAP_REFERENCE:
                    post_hit_gap_adjustment = max(
                        0.35,
                        post_hit_best_gap / self.POST_HIT_GAP_REFERENCE,
                    )
                rollout_margin_basis = (
                    ((1.0 - self.CONTINUATION_TOP_K_BLEND) * max(0.0, post_hit_continue_margin))
                    + (self.CONTINUATION_TOP_K_BLEND * post_hit_top_k_continue_margin)
                )
                post_hit_continuation_value = (
                    self.CONTINUATION_DISCOUNT
                    * continuation_likelihood
                    * max(0.0, rollout_margin_basis)
                    * post_hit_gap_adjustment
                )
                continuation_value = probability * post_hit_continuation_value
                post_hit_behavior_support_breakdown = self._post_hit_behavior_support_breakdown(
                    best_move={
                        "post_hit_stop_score": post_hit_stop_score,
                        "post_hit_continue_margin": post_hit_continue_margin,
                        "continuation_value": continuation_value,
                        "post_hit_top_k_expected_continue_margin": post_hit_top_k_expected_continue_margin,
                        "post_hit_top_k_continue_margin": post_hit_top_k_continue_margin,
                        "post_hit_top_k_expected_support_ratio": post_hit_top_k_expected_support_ratio,
                        "post_hit_top_k_support_ratio": post_hit_top_k_support_ratio,
                        "post_hit_guidance_multiplier": post_hit_guidance_multiplier,
                        "post_hit_guidance_support": post_hit_guidance_support,
                        "post_hit_guidance_stable_ratio": post_hit_guidance_stable_ratio,
                    }
                )
                post_hit_behavior_support_adjustment = post_hit_behavior_support_breakdown["adjustment"]
                post_hit_behavior_support_gain = post_hit_behavior_support_breakdown["support_gain"]
                post_hit_behavior_fragility_drag = post_hit_behavior_support_breakdown["fragility_drag"]
                post_hit_behavior_support_signal = post_hit_behavior_support_breakdown["support_signal"]
                post_hit_behavior_fragility_signal = post_hit_behavior_support_breakdown["fragility_signal"]
                post_hit_behavior_support_strength = post_hit_behavior_support_breakdown["support_strength"]
                post_hit_behavior_fragility_strength = post_hit_behavior_support_breakdown["fragility_strength"]

        hit_reward = probability * self.HIT_REWARD
        miss_penalty = (1.0 - probability) * risk_factor
        info_bonus = info_gain * self.INFORMATION_GAIN_WEIGHT
        immediate_expected_value = hit_reward - miss_penalty + info_bonus
        expected_value = immediate_expected_value + continuation_value
        if (
            behavior_guidance_profile is not None
            and game_state is not None
            and behavior_map_hypothesis is not None
            and acting_player_id is not None
        ):
            behavior_candidate_signal = self._posterior_candidate_signal(
                full_probability_matrix=full_probability_matrix,
                game_state=game_state,
                behavior_model=behavior_model,
                behavior_map_hypothesis=behavior_map_hypothesis,
                guesser_id=acting_player_id,
                target_player_id=player_id,
                target_slot_index=slot_index,
                guessed_card=card,
            )
            behavior_match_support = self._behavior_match_support(
                behavior_guidance_profile=behavior_guidance_profile,
                behavior_candidate_signal=behavior_candidate_signal,
            )
            behavior_match_multiplier = self._behavior_match_multiplier(
                behavior_match_support=behavior_match_support,
                stable_signal_ratio=float(behavior_guidance_profile.get("stable_signal_ratio", 0.0)),
            )
            behavior_match_bonus = max(0.0, immediate_expected_value) * (behavior_match_multiplier - 1.0)
            behavior_match_confidence_breakdown = self._behavior_match_candidate_confidence_breakdown(
                best_move={
                    "behavior_candidate_signal": behavior_candidate_signal,
                }
            )
        behavior_match_ranking_breakdown = self._behavior_match_ranking_breakdown(
            best_move={
                "behavior_match_bonus": behavior_match_bonus,
                "behavior_match_candidate_confidence": behavior_match_confidence_breakdown["candidate_confidence"],
                "behavior_match_component_support": behavior_match_confidence_breakdown["component_support"],
                "behavior_match_component_strength": behavior_match_confidence_breakdown["component_strength"],
                "behavior_match_component_penalty": behavior_match_confidence_breakdown["component_penalty"],
                "behavior_match_context_focus": behavior_match_confidence_breakdown["context_focus"],
            }
        )
        behavior_match_ranking_bonus = behavior_match_ranking_breakdown["ranking_bonus"]
        behavior_match_net_structure = behavior_match_ranking_breakdown["net_structure"]
        behavior_match_structure_adjustment = behavior_match_ranking_breakdown["structure_adjustment"]
        ranking_score = expected_value + behavior_match_ranking_bonus

        return {
            "target_player_id": player_id,
            "target_index": hidden_index_by_slot.get(slot_index, slot_index),
            "target_slot_index": slot_index,
            "guess_card": serialize_card(card),
            "win_probability": probability,
            "immediate_expected_value": immediate_expected_value,
            "expected_value": expected_value,
            "information_gain": info_gain,
            "continuation_value": continuation_value,
            "post_hit_continuation_value": post_hit_continuation_value,
            "next_best_immediate_value": next_best_immediate_ev,
            "continuation_likelihood": continuation_likelihood,
            "behavior_guidance_multiplier": behavior_guidance_multiplier,
            "behavior_guidance_support": behavior_guidance_support,
            "behavior_guidance_stable_ratio": behavior_guidance_stable_ratio,
            "behavior_guidance_signal_count": behavior_guidance_signal_count,
            "behavior_match_multiplier": behavior_match_multiplier,
            "behavior_match_bonus": behavior_match_bonus,
            "behavior_match_ranking_bonus": behavior_match_ranking_bonus,
            "behavior_match_net_structure": behavior_match_net_structure,
            "behavior_match_structure_adjustment": behavior_match_structure_adjustment,
            "behavior_match_support": behavior_match_support,
            "behavior_match_candidate_confidence": behavior_match_confidence_breakdown["candidate_confidence"],
            "behavior_match_component_support": behavior_match_confidence_breakdown["component_support"],
            "behavior_match_component_strength": behavior_match_confidence_breakdown["component_strength"],
            "behavior_match_component_penalty": behavior_match_confidence_breakdown["component_penalty"],
            "behavior_match_context_focus": behavior_match_confidence_breakdown["context_focus"],
            "behavior_candidate_signal": behavior_candidate_signal,
            "ranking_score": ranking_score,
            "attackability_after_hit": attackability_after_hit,
            "post_hit_continue_score": post_hit_continue_score,
            "post_hit_stop_score": post_hit_stop_score,
            "post_hit_continue_margin": post_hit_continue_margin,
            "post_hit_should_continue": post_hit_should_continue,
            "post_hit_best_gap": post_hit_best_gap,
            "post_hit_gap_adjustment": post_hit_gap_adjustment,
            "post_hit_guidance_multiplier": post_hit_guidance_multiplier,
            "post_hit_guidance_support": post_hit_guidance_support,
            "post_hit_guidance_stable_ratio": post_hit_guidance_stable_ratio,
            "post_hit_guidance_signal_count": post_hit_guidance_signal_count,
            "post_hit_guidance_debug": post_hit_guidance_debug,
            "post_hit_behavior_support_adjustment": post_hit_behavior_support_adjustment,
            "post_hit_behavior_support_gain": post_hit_behavior_support_gain,
            "post_hit_behavior_fragility_drag": post_hit_behavior_fragility_drag,
            "post_hit_behavior_support_signal": post_hit_behavior_support_signal,
            "post_hit_behavior_fragility_signal": post_hit_behavior_fragility_signal,
            "post_hit_behavior_support_strength": post_hit_behavior_support_strength,
            "post_hit_behavior_fragility_strength": post_hit_behavior_fragility_strength,
            "post_hit_top_k_expected_continue_margin": post_hit_top_k_expected_continue_margin,
            "post_hit_top_k_continue_margin": post_hit_top_k_continue_margin,
            "post_hit_top_k_expected_support_ratio": post_hit_top_k_expected_support_ratio,
            "post_hit_top_k_support_ratio": post_hit_top_k_support_ratio,
            "post_hit_top_k_positive_count": post_hit_top_k_positive_count,
            "history_continue_rate": history_continue_rate,
            "hit_reward": hit_reward,
            "miss_penalty": miss_penalty,
            "score_breakdown": {
                "hit_reward": hit_reward,
                "miss_penalty": miss_penalty,
                "information_gain_bonus": info_bonus,
                "immediate_expected_value": immediate_expected_value,
                "continuation_value": continuation_value,
                "post_hit_continuation_value": post_hit_continuation_value,
                "next_best_immediate_value": next_best_immediate_ev,
                "continuation_likelihood": continuation_likelihood,
                "behavior_guidance_multiplier": behavior_guidance_multiplier,
                "behavior_guidance_support": behavior_guidance_support,
                "behavior_guidance_stable_ratio": behavior_guidance_stable_ratio,
                "behavior_guidance_signal_count": behavior_guidance_signal_count,
                "behavior_match_multiplier": behavior_match_multiplier,
                "behavior_match_bonus": behavior_match_bonus,
                "behavior_match_ranking_bonus": behavior_match_ranking_bonus,
                "behavior_match_net_structure": behavior_match_net_structure,
                "behavior_match_structure_adjustment": behavior_match_structure_adjustment,
                "behavior_match_support": behavior_match_support,
                "behavior_match_candidate_confidence": behavior_match_confidence_breakdown["candidate_confidence"],
                "behavior_match_component_support": behavior_match_confidence_breakdown["component_support"],
                "behavior_match_component_strength": behavior_match_confidence_breakdown["component_strength"],
                "behavior_match_component_penalty": behavior_match_confidence_breakdown["component_penalty"],
                "behavior_match_context_focus": behavior_match_confidence_breakdown["context_focus"],
                "ranking_score": ranking_score,
                "attackability_after_hit": attackability_after_hit,
                "post_hit_continue_score": post_hit_continue_score,
                "post_hit_stop_score": post_hit_stop_score,
                "post_hit_continue_margin": post_hit_continue_margin,
                "post_hit_best_gap": post_hit_best_gap,
                "post_hit_gap_adjustment": post_hit_gap_adjustment,
                "post_hit_guidance_multiplier": post_hit_guidance_multiplier,
                "post_hit_guidance_support": post_hit_guidance_support,
                "post_hit_guidance_stable_ratio": post_hit_guidance_stable_ratio,
                "post_hit_guidance_signal_count": post_hit_guidance_signal_count,
                "post_hit_behavior_support_adjustment": post_hit_behavior_support_adjustment,
                "post_hit_behavior_support_gain": post_hit_behavior_support_gain,
                "post_hit_behavior_fragility_drag": post_hit_behavior_fragility_drag,
                "post_hit_behavior_support_signal": post_hit_behavior_support_signal,
                "post_hit_behavior_fragility_signal": post_hit_behavior_fragility_signal,
                "post_hit_behavior_support_strength": post_hit_behavior_support_strength,
                "post_hit_behavior_fragility_strength": post_hit_behavior_fragility_strength,
                "post_hit_top_k_expected_continue_margin": post_hit_top_k_expected_continue_margin,
                "post_hit_top_k_continue_margin": post_hit_top_k_continue_margin,
                "post_hit_top_k_expected_support_ratio": post_hit_top_k_expected_support_ratio,
                "post_hit_top_k_support_ratio": post_hit_top_k_support_ratio,
                "post_hit_top_k_positive_count": post_hit_top_k_positive_count,
            },
            "recommendation_reason": self._build_reason(
                probability,
                info_gain,
                continuation_value,
                miss_penalty,
                continuation_likelihood,
            ),
            "target_scope": "player_slots",
        }

    def _evaluate_post_hit_rollout(
        self,
        *,
        success_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        behavior_guidance_profile: Optional[Dict[str, float]],
        game_state: Optional[GameState],
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
        rollout_depth: int,
    ) -> Dict[str, Any]:
        hidden_index_by_player = self._hidden_index_by_player_from_matrix(success_matrix)
        post_hit_game_state = None
        post_hit_guess_signals_by_player = guess_signals_by_player
        post_hit_behavior_map_hypothesis = None
        post_hit_behavior_guidance_profile = behavior_guidance_profile
        post_hit_behavior_guidance_debug = self._default_post_hit_guidance_rebuild_debug(
            base_profile=behavior_guidance_profile,
            acting_player_id=acting_player_id,
        )
        if game_state is not None and acting_player_id is not None:
            post_hit_context = self._build_post_hit_behavior_context(
                game_state=game_state,
                success_matrix=success_matrix,
                behavior_model=behavior_model,
                acting_player_id=acting_player_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_card=guessed_card,
            )
            post_hit_game_state = post_hit_context["game_state"]
            post_hit_guess_signals_by_player = post_hit_context["guess_signals_by_player"]
            post_hit_behavior_map_hypothesis = post_hit_context["behavior_map_hypothesis"]
            post_hit_guidance_rebuild = self._rebuild_post_hit_behavior_guidance_profile(
                base_profile=behavior_guidance_profile,
                behavior_model=behavior_model,
                full_probability_matrix=success_matrix,
                game_state=post_hit_game_state,
                guess_signals_by_player=post_hit_guess_signals_by_player,
                behavior_map_hypothesis=post_hit_behavior_map_hypothesis,
                acting_player_id=acting_player_id,
            )
            post_hit_behavior_guidance_profile = post_hit_guidance_rebuild["profile"]
            post_hit_behavior_guidance_debug = post_hit_guidance_rebuild["debug"]
        next_moves, next_risk_factor = self.evaluate_all_moves(
            full_probability_matrix=success_matrix,
            my_hidden_count=my_hidden_count,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=behavior_model,
            guess_signals_by_player=post_hit_guess_signals_by_player,
            acting_player_id=acting_player_id,
            behavior_guidance_profile=post_hit_behavior_guidance_profile,
            game_state=post_hit_game_state,
            behavior_map_hypothesis=post_hit_behavior_map_hypothesis,
            blocked_slots=set(),
            rollout_depth=rollout_depth,
        )
        next_best_move, next_summary = self.choose_best_move(
            next_moves,
            risk_factor=next_risk_factor,
            my_hidden_count=my_hidden_count,
        )
        top_k_moves = next_moves[: self.POST_HIT_TOP_K_COUNT]
        stop_score = next_summary.get("stop_score", next_summary.get("stop_threshold", 0.0))
        top_k_expected_continue_edges = [
            max(0.0, move["expected_value"] - stop_score)
            for move in top_k_moves
        ]
        top_k_continue_edges = [
            max(0.0, move.get("ranking_score", move["expected_value"]) - stop_score)
            for move in top_k_moves
        ]
        top_k_expected_continue_margin = (
            sum(top_k_expected_continue_edges) / len(top_k_expected_continue_edges)
            if top_k_expected_continue_edges
            else 0.0
        )
        top_k_continue_margin = (
            sum(top_k_continue_edges) / len(top_k_continue_edges)
            if top_k_continue_edges
            else 0.0
        )
        top_k_expected_positive_count = float(
            sum(1 for edge in top_k_expected_continue_edges if edge > 0.0)
        )
        top_k_positive_count = float(sum(1 for edge in top_k_continue_edges if edge > 0.0))
        top_k_expected_support_ratio = (
            top_k_expected_positive_count / len(top_k_expected_continue_edges)
            if top_k_expected_continue_edges
            else 0.0
        )
        top_k_support_ratio = (
            top_k_positive_count / len(top_k_continue_edges)
            if top_k_continue_edges
            else 0.0
        )
        return {
            "should_continue": next_best_move is not None,
            "continue_score": next_summary.get("continue_score", 0.0),
            "stop_score": stop_score,
            "continue_margin": next_summary.get("continue_margin", 0.0),
            "best_gap": next_summary.get("best_gap", 0.0),
            "behavior_guidance_multiplier": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "guidance_multiplier",
                    self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                )
            ),
            "behavior_guidance_support": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "average_posterior_support",
                    0.0,
                )
            ),
            "behavior_guidance_stable_ratio": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "stable_signal_ratio",
                    0.0,
                )
            ),
            "behavior_guidance_signal_count": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "signal_count",
                    0.0,
                )
            ),
            "top_k_expected_continue_margin": top_k_expected_continue_margin,
            "top_k_continue_margin": top_k_continue_margin,
            "top_k_expected_support_ratio": top_k_expected_support_ratio,
            "top_k_support_ratio": top_k_support_ratio,
            "top_k_positive_count": top_k_positive_count,
            "guidance_debug": post_hit_behavior_guidance_debug,
        }

    def _build_post_hit_behavior_context(
        self,
        *,
        game_state: GameState,
        success_matrix: FullProbabilityMatrix,
        behavior_model: BehavioralLikelihoodModel,
        acting_player_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, Any]:
        players: Dict[str, PlayerState] = {}
        for player_id in game_state.players:
            slots: List[CardSlot] = []
            for slot in game_state.resolved_ordered_slots(player_id):
                slot_card = slot.known_card()
                if player_id == target_player_id and slot.slot_index == target_slot_index:
                    slot_card = guessed_card
                slots.append(
                    CardSlot(
                        slot_index=slot.slot_index,
                        color=slot_card[0] if slot_card is not None else slot.color,
                        value=slot_card[1] if slot_card is not None else slot.value,
                        is_revealed=(slot_card is not None) or slot.is_revealed,
                        is_newly_drawn=slot.is_newly_drawn,
                    )
                )
            players[player_id] = PlayerState(player_id=player_id, slots=slots)

        post_hit_actions = list(game_state.actions)
        post_hit_actions.append(
            GuessAction(
                guesser_id=acting_player_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_color=guessed_card[0],
                guessed_value=guessed_card[1],
                result=True,
                continued_turn=True,
                revealed_player_id=target_player_id,
                revealed_slot_index=target_slot_index,
                revealed_color=guessed_card[0],
                revealed_value=guessed_card[1],
            )
        )
        post_hit_game_state = GameState(
            self_player_id=game_state.self_player_id,
            target_player_id=game_state.target_player_id,
            players=players,
            actions=post_hit_actions,
        )
        return {
            "game_state": post_hit_game_state,
            "guess_signals_by_player": behavior_model.build_guess_signals(post_hit_game_state),
            "behavior_map_hypothesis": self._map_hypothesis_from_matrix(success_matrix),
        }

    def _rebuild_post_hit_behavior_guidance_profile(
        self,
        *,
        base_profile: Optional[Dict[str, float]],
        behavior_model: BehavioralLikelihoodModel,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        acting_player_id: str,
    ) -> Dict[str, Any]:
        fallback_profile = {
            "signal_count": float((base_profile or {}).get("signal_count", 0.0)),
            "average_posterior_support": float((base_profile or {}).get("average_posterior_support", 0.0)),
            "average_weighted_strength": float((base_profile or {}).get("average_weighted_strength", 0.0)),
            "stable_signal_ratio": float((base_profile or {}).get("stable_signal_ratio", 0.0)),
            "guidance_multiplier": float(
                (base_profile or {}).get(
                    "guidance_multiplier",
                    self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                )
            ),
            "source_support_progressive": float((base_profile or {}).get("source_support_progressive", 0.0)),
            "source_support_same_color_anchor": float((base_profile or {}).get("source_support_same_color_anchor", 0.0)),
            "source_support_local_boundary": float((base_profile or {}).get("source_support_local_boundary", 0.0)),
        }
        default_debug = self._default_post_hit_guidance_rebuild_debug(
            base_profile=fallback_profile,
            acting_player_id=acting_player_id,
        )
        acting_signals = guess_signals_by_player.get(acting_player_id, ())
        if not acting_signals:
            return {
                "profile": fallback_profile,
                "debug": default_debug,
            }

        guidance_matrix_context = self._augment_behavior_debug_matrix(
            full_probability_matrix=full_probability_matrix,
            game_state=game_state,
            guess_signals_by_player=guess_signals_by_player,
        )
        guidance_matrix = guidance_matrix_context["matrix"]
        guidance_map_hypothesis = self._map_hypothesis_from_matrix(guidance_matrix)
        guidance_controller = GameController(game_state)
        behavior_debug = guidance_controller._build_behavior_debug(
            full_probability_matrix=guidance_matrix,
            guess_signals_by_player=guess_signals_by_player,
            map_hypothesis=guidance_map_hypothesis or behavior_map_hypothesis,
        )
        rebuilt_profile = guidance_controller._build_behavior_guidance_profile(
            behavior_debug_signals=behavior_debug["signals"],
            acting_player_id=acting_player_id,
        )
        if float(rebuilt_profile.get("signal_count", 0.0)) <= 0.0:
            return {
                "profile": fallback_profile,
                "debug": default_debug,
            }
        blended_profile = self._blend_behavior_guidance_profiles(
            base_profile=fallback_profile,
            updated_profile=rebuilt_profile,
        )
        signal_summaries = self._summarize_guidance_rebuild_signals(
            behavior_debug_signals=behavior_debug["signals"],
            acting_player_id=acting_player_id,
        )
        debug_payload = {
            "rebuild_applied": True,
            "acting_player_id": acting_player_id,
            "acting_signal_count": float(len(acting_signals)),
            "rebuilt_signal_count": float(rebuilt_profile.get("signal_count", 0.0)),
            "augmented_known_slot_count": float(len(guidance_matrix_context["augmented_known_slots"])),
            "augmented_known_slots": guidance_matrix_context["augmented_known_slots"],
            "signal_summaries": signal_summaries,
            "base_profile": dict(fallback_profile),
            "rebuilt_profile": dict(rebuilt_profile),
            "blended_profile": dict(blended_profile),
            "rebuilt_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=fallback_profile,
                updated_profile=rebuilt_profile,
            ),
            "blended_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=fallback_profile,
                updated_profile=blended_profile,
            ),
            "rebuilt_dominant_source_shift": self._dominant_guidance_source_shift(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=rebuilt_profile,
                )
            ),
            "rebuilt_dominant_source_shift_strength": self._dominant_guidance_source_shift_strength(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=rebuilt_profile,
                )
            ),
            "blended_dominant_source_shift": self._dominant_guidance_source_shift(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=blended_profile,
                )
            ),
            "blended_dominant_source_shift_strength": self._dominant_guidance_source_shift_strength(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=blended_profile,
                )
            ),
        }
        return {
            "profile": blended_profile,
            "debug": debug_payload,
        }

    def _augment_behavior_debug_matrix(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
    ) -> Dict[str, Any]:
        augmented_matrix: FullProbabilityMatrix = {
            player_id: {
                slot_index: dict(slot_distribution)
                for slot_index, slot_distribution in probability_matrix.items()
            }
            for player_id, probability_matrix in full_probability_matrix.items()
        }
        augmented_known_slots: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for signals in guess_signals_by_player.values():
            for signal in signals:
                try:
                    slot = game_state.get_slot(signal.target_player_id, signal.target_slot_index)
                except ValueError:
                    continue
                known_card = slot.known_card()
                if known_card is None:
                    continue
                augmented_matrix.setdefault(signal.target_player_id, {})[signal.target_slot_index] = {
                    known_card: 1.0,
                }
                augmented_known_slots[(signal.target_player_id, signal.target_slot_index)] = {
                    "player_id": signal.target_player_id,
                    "slot_index": signal.target_slot_index,
                    "card": serialize_card(known_card),
                }
        return {
            "matrix": augmented_matrix,
            "augmented_known_slots": [
                augmented_known_slots[key]
                for key in sorted(augmented_known_slots)
            ],
        }

    def _default_post_hit_guidance_rebuild_debug(
        self,
        *,
        base_profile: Optional[Dict[str, float]],
        acting_player_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "rebuild_applied": False,
            "acting_player_id": acting_player_id,
            "acting_signal_count": 0.0,
            "rebuilt_signal_count": 0.0,
            "augmented_known_slot_count": 0.0,
            "augmented_known_slots": [],
            "signal_summaries": [],
            "base_profile": dict(base_profile or {}),
            "rebuilt_profile": {
                "signal_count": 0.0,
                "average_posterior_support": 0.0,
                "average_weighted_strength": 0.0,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 0.0,
            },
            "blended_profile": dict(base_profile or {}),
            "rebuilt_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=base_profile or {},
                updated_profile=None,
            ),
            "blended_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=base_profile or {},
                updated_profile=base_profile or {},
            ),
            "rebuilt_dominant_source_shift": "neutral",
            "blended_dominant_source_shift": "neutral",
            "rebuilt_dominant_source_shift_strength": 0.0,
            "blended_dominant_source_shift_strength": 0.0,
        }

    def _summarize_guidance_rebuild_signals(
        self,
        *,
        behavior_debug_signals: Sequence[Dict[str, Any]],
        acting_player_id: str,
    ) -> List[Dict[str, Any]]:
        signal_summaries: List[Dict[str, Any]] = []
        for signal in behavior_debug_signals:
            if signal.get("guesser_id") != acting_player_id:
                continue
            dominant_signal = signal.get("value_selection", {}).get("dominant_signal", {})
            signal_summaries.append(
                {
                    "target_player_id": signal.get("target_player_id"),
                    "target_slot_index": signal.get("target_slot_index"),
                    "result": signal.get("result"),
                    "continued_turn": signal.get("continued_turn"),
                    "dominant_source": dominant_signal.get("source"),
                    "posterior_support": float(dominant_signal.get("posterior_support", 0.0)),
                    "weighted_strength": float(dominant_signal.get("weighted_strength", 0.0)),
                    "covered_probability": float(
                        signal.get("value_selection", {}).get("covered_probability", 0.0)
                    ),
                }
            )
        return signal_summaries

    def _behavior_guidance_profile_delta(
        self,
        *,
        base_profile: Dict[str, float],
        updated_profile: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        keys = (
            "signal_count",
            "average_posterior_support",
            "average_weighted_strength",
            "stable_signal_ratio",
            "guidance_multiplier",
            "source_support_progressive",
            "source_support_same_color_anchor",
            "source_support_local_boundary",
        )
        reference_profile = updated_profile or {}
        return {
            key: float(reference_profile.get(key, 0.0)) - float(base_profile.get(key, 0.0))
            for key in keys
        }

    def _dominant_guidance_source_shift(
        self,
        delta_profile: Dict[str, float],
    ) -> str:
        source_key_mapping = {
            "progressive": float(delta_profile.get("source_support_progressive", 0.0)),
            "same_color_anchor": float(delta_profile.get("source_support_same_color_anchor", 0.0)),
            "local_boundary": float(delta_profile.get("source_support_local_boundary", 0.0)),
        }
        dominant_source, dominant_delta = max(
            source_key_mapping.items(),
            key=lambda item: (item[1], item[0]),
        )
        if dominant_delta <= 0.0:
            return "neutral"
        return dominant_source

    def _dominant_guidance_source_shift_strength(
        self,
        delta_profile: Dict[str, float],
    ) -> float:
        return max(
            0.0,
            float(delta_profile.get("source_support_progressive", 0.0)),
            float(delta_profile.get("source_support_same_color_anchor", 0.0)),
            float(delta_profile.get("source_support_local_boundary", 0.0)),
        )

    def _build_post_hit_guidance_rebuild_reason(
        self,
        guidance_debug: Dict[str, Any],
    ) -> str:
        if not guidance_debug.get("rebuild_applied", False):
            return "post-hit rebuild unavailable"
        source_shift = str(guidance_debug.get("blended_dominant_source_shift", "neutral"))
        shift_strength = float(guidance_debug.get("blended_dominant_source_shift_strength", 0.0))
        multiplier_delta = float(
            guidance_debug.get("blended_delta_from_base", {}).get("guidance_multiplier", 0.0)
        )
        rebuilt_signal_count = float(guidance_debug.get("rebuilt_signal_count", 0.0))
        return (
            f"post-hit rebuild favors {source_shift}, "
            f"strength {shift_strength:.3f}, "
            f"signals {rebuilt_signal_count:.0f}, "
            f"multiplier delta {multiplier_delta:.3f}"
        )

    def _blend_behavior_guidance_profiles(
        self,
        *,
        base_profile: Dict[str, float],
        updated_profile: Dict[str, float],
    ) -> Dict[str, float]:
        base_count = max(0.0, float(base_profile.get("signal_count", 0.0)))
        updated_count = max(0.0, float(updated_profile.get("signal_count", 0.0)))
        if base_count <= 0.0:
            return dict(updated_profile)
        if updated_count <= 0.0:
            return dict(base_profile)

        total_count = base_count + updated_count
        blended_profile = {
            "signal_count": total_count,
            "average_posterior_support": (
                (float(base_profile.get("average_posterior_support", 0.0)) * base_count)
                + (float(updated_profile.get("average_posterior_support", 0.0)) * updated_count)
            ) / total_count,
            "average_weighted_strength": (
                (float(base_profile.get("average_weighted_strength", 0.0)) * base_count)
                + (float(updated_profile.get("average_weighted_strength", 0.0)) * updated_count)
            ) / total_count,
            "stable_signal_ratio": (
                (float(base_profile.get("stable_signal_ratio", 0.0)) * base_count)
                + (float(updated_profile.get("stable_signal_ratio", 0.0)) * updated_count)
            ) / total_count,
            "source_support_progressive": (
                (float(base_profile.get("source_support_progressive", 0.0)) * base_count)
                + (float(updated_profile.get("source_support_progressive", 0.0)) * updated_count)
            ) / total_count,
            "source_support_same_color_anchor": (
                (float(base_profile.get("source_support_same_color_anchor", 0.0)) * base_count)
                + (float(updated_profile.get("source_support_same_color_anchor", 0.0)) * updated_count)
            ) / total_count,
            "source_support_local_boundary": (
                (float(base_profile.get("source_support_local_boundary", 0.0)) * base_count)
                + (float(updated_profile.get("source_support_local_boundary", 0.0)) * updated_count)
            ) / total_count,
        }
        blended_profile["guidance_multiplier"] = clamp(
            0.95
            + (0.08 * blended_profile["average_posterior_support"])
            + (0.07 * blended_profile["stable_signal_ratio"])
            + (0.14 * blended_profile["average_weighted_strength"]),
            0.93,
            1.12,
        )
        return blended_profile

    def _map_hypothesis_from_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Dict[str, Dict[int, Card]]:
        hypothesis_by_player: Dict[str, Dict[int, Card]] = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_hypothesis: Dict[int, Card] = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if not slot_distribution:
                    continue
                best_card = max(
                    slot_distribution.items(),
                    key=lambda item: (item[1], -card_sort_key(item[0])[0], -card_sort_key(item[0])[1]),
                )[0]
                player_hypothesis[slot_index] = best_card
            if player_hypothesis:
                hypothesis_by_player[player_id] = player_hypothesis
        return hypothesis_by_player

    def _behavior_match_support(
        self,
        *,
        behavior_guidance_profile: Dict[str, float],
        behavior_candidate_signal: Dict[str, Any],
    ) -> float:
        source_weight_mapping = {
            "progressive": max(
                0.0,
                float(behavior_candidate_signal["progressive"].get("weight", 1.0)) - 1.0,
            ),
            "same_color_anchor": max(
                0.0,
                float(behavior_candidate_signal["anchor"].get("weight", 1.0)) - 1.0,
            ),
            "local_boundary": max(
                0.0,
                float(behavior_candidate_signal["boundary"].get("weight", 1.0)) - 1.0,
            ),
        }
        return sum(
            float(behavior_guidance_profile.get(f"source_support_{source}", 0.0)) * weight
            for source, weight in source_weight_mapping.items()
        )

    def _posterior_candidate_signal(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        behavior_model: BehavioralLikelihoodModel,
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, Any]:
        map_signal = behavior_model.describe_candidate_value_signal(
            game_state=game_state,
            hypothesis_by_player=behavior_map_hypothesis,
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
            guessed_card=guessed_card,
        )
        context_slots = self._neighbor_context_slot_indices(
            game_state,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
        )
        context_candidates: List[Tuple[int, List[Tuple[Card, float]], float]] = []
        covered_probability = 1.0

        for context_slot_index in context_slots:
            slot_candidates, slot_covered_probability = self._context_slot_candidates(
                full_probability_matrix=full_probability_matrix,
                game_state=game_state,
                behavior_map_hypothesis=behavior_map_hypothesis,
                player_id=target_player_id,
                slot_index=context_slot_index,
            )
            if len(slot_candidates) <= 1:
                continue
            context_candidates.append(
                (context_slot_index, slot_candidates, slot_covered_probability)
            )
            covered_probability *= slot_covered_probability

        if not context_candidates:
            return {
                **map_signal,
                "mode": "map_context_fallback",
                "context_candidate_count": 0,
                "context_covered_probability": 1.0,
                "map_signal": map_signal,
            }

        hypothesis_variants: List[Tuple[Dict[str, Dict[int, Card]], float]] = [
            (behavior_map_hypothesis, 1.0),
        ]
        for context_slot_index, slot_candidates, _ in context_candidates:
            next_variants: List[Tuple[Dict[str, Dict[int, Card]], float]] = []
            for hypothesis_variant, variant_weight in hypothesis_variants:
                for card, candidate_weight in slot_candidates:
                    next_variants.append(
                        (
                            self._replace_hypothesis_card(
                                hypothesis_variant,
                                target_player_id,
                                context_slot_index,
                                card,
                            ),
                            variant_weight * candidate_weight,
                        )
                    )
            hypothesis_variants = next_variants

        total_variant_weight = max(
            behavior_model.EPSILON,
            sum(weight for _, weight in hypothesis_variants),
        )
        component_weight_sums: DefaultDict[str, float] = defaultdict(float)
        reason_support_sums: Dict[str, DefaultDict[str, float]] = {
            "progressive": defaultdict(float),
            "anchor": defaultdict(float),
            "boundary": defaultdict(float),
        }

        for hypothesis_variant, variant_weight in hypothesis_variants:
            normalized_weight = variant_weight / total_variant_weight
            variant_signal = behavior_model.describe_candidate_value_signal(
                game_state=game_state,
                hypothesis_by_player=hypothesis_variant,
                guesser_id=guesser_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_card=guessed_card,
            )
            for component_name in ("progressive", "anchor", "boundary"):
                component_weight_sums[component_name] += (
                    normalized_weight * float(variant_signal[component_name]["weight"])
                )
                component_reason = str(variant_signal[component_name].get("reason", "neutral"))
                reason_support_sums[component_name][component_reason] += normalized_weight

        aggregated_components = {
            component_name: self._aggregate_candidate_signal_component(
                component_weight=component_weight_sums[component_name],
                reason_support=reason_support_sums[component_name],
            )
            for component_name in ("progressive", "anchor", "boundary")
        }
        source_mapping = {
            "progressive": aggregated_components["progressive"],
            "same_color_anchor": aggregated_components["anchor"],
            "local_boundary": aggregated_components["boundary"],
        }
        dominant_signal = behavior_model._dominant_signal(source_mapping)
        dominant_component_name = {
            "progressive": "progressive",
            "same_color_anchor": "anchor",
            "local_boundary": "boundary",
        }.get(dominant_signal["source"])
        if dominant_component_name is not None:
            dominant_signal["posterior_support"] = aggregated_components[dominant_component_name]["posterior_support"]
        signal_tags = [
            component["reason"]
            for component in aggregated_components.values()
            if component["reason"] != "neutral"
        ]
        return {
            "signal_tags": signal_tags,
            "dominant_signal": dominant_signal,
            "progressive": aggregated_components["progressive"],
            "anchor": aggregated_components["anchor"],
            "boundary": aggregated_components["boundary"],
            "mode": "neighbor_top_k_posterior",
            "context_candidate_count": len(hypothesis_variants),
            "context_covered_probability": covered_probability,
            "map_signal": map_signal,
        }

    def _aggregate_candidate_signal_component(
        self,
        *,
        component_weight: float,
        reason_support: DefaultDict[str, float],
    ) -> Dict[str, Any]:
        sorted_reasons = sorted(
            reason_support.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if not sorted_reasons:
            return {
                "weight": component_weight,
                "reason": "neutral",
                "posterior_support": 0.0,
            }

        top_reason, top_support = sorted_reasons[0]
        if top_reason == "neutral" and len(sorted_reasons) > 1:
            top_reason, top_support = sorted_reasons[1]
        if top_reason == "neutral":
            top_support = 0.0
        return {
            "weight": component_weight,
            "reason": top_reason,
            "posterior_support": top_support,
        }

    def _neighbor_context_slot_indices(
        self,
        game_state: GameState,
        *,
        target_player_id: str,
        target_slot_index: int,
    ) -> List[int]:
        slots = game_state.resolved_ordered_slots(target_player_id)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(slots)}
        order_index = index_by_slot.get(target_slot_index)
        if order_index is None:
            return []

        context_slot_indices: List[int] = []
        if order_index > 0:
            context_slot_indices.append(slots[order_index - 1].slot_index)
        if order_index + 1 < len(slots):
            context_slot_indices.append(slots[order_index + 1].slot_index)
        return context_slot_indices

    def _context_slot_candidates(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Tuple[List[Tuple[Card, float]], float]:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return [], 0.0

        known_card = slot.known_card()
        if known_card is not None:
            return [(known_card, 1.0)], 1.0

        slot_distribution = full_probability_matrix.get(player_id, {}).get(slot_index, {})
        if slot_distribution:
            ranked_candidates = sorted(
                slot_distribution.items(),
                key=lambda item: (-item[1], card_sort_key(item[0])),
            )[: self.BEHAVIOR_MATCH_CONTEXT_TOP_K]
            covered_probability = sum(probability for _, probability in ranked_candidates)
            if covered_probability > 0.0:
                return (
                    [
                        (card, probability / covered_probability)
                        for card, probability in ranked_candidates
                    ],
                    covered_probability,
                )

        fallback_card = behavior_map_hypothesis.get(player_id, {}).get(slot_index)
        if fallback_card is not None:
            return [(fallback_card, 1.0)], 1.0
        return [], 0.0

    def _replace_hypothesis_card(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
        card: Card,
    ) -> Dict[str, Dict[int, Card]]:
        cloned = {
            hypothesis_player_id: dict(cards_by_slot)
            for hypothesis_player_id, cards_by_slot in hypothesis_by_player.items()
        }
        player_hypothesis = cloned.setdefault(player_id, {})
        player_hypothesis[slot_index] = card
        return cloned

    def _behavior_match_multiplier(
        self,
        *,
        behavior_match_support: float,
        stable_signal_ratio: float,
    ) -> float:
        if behavior_match_support <= 0.0:
            if stable_signal_ratio >= 0.65:
                return 0.99
            return self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER
        return clamp(
            1.0 + ((0.32 + (0.16 * stable_signal_ratio)) * behavior_match_support),
            0.99,
            1.08,
        )

    def _hidden_index_by_player_from_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Dict[str, Dict[int, int]]:
        return {
            player_id: {
                slot_index: hidden_index
                for hidden_index, slot_index in enumerate(sorted(probability_matrix))
            }
            for player_id, probability_matrix in full_probability_matrix.items()
        }

    def _evaluate_continue_decision(
        self,
        *,
        best_move: Dict[str, Any],
        second_move: Optional[Dict[str, Any]],
        stop_threshold: float,
        my_hidden_count: int,
    ) -> Dict[str, Any]:
        best_gap = best_move["expected_value"] - (second_move["expected_value"] if second_move else 0.0)
        behavior_match_confidence_breakdown = self._behavior_match_candidate_confidence_breakdown(
            best_move=best_move,
        )
        behavior_match_candidate_confidence = behavior_match_confidence_breakdown["candidate_confidence"]
        behavior_match_component_support = behavior_match_confidence_breakdown["component_support"]
        behavior_match_component_strength = behavior_match_confidence_breakdown["component_strength"]
        behavior_match_component_penalty = behavior_match_confidence_breakdown["component_penalty"]
        behavior_match_context_focus = behavior_match_confidence_breakdown["context_focus"]
        behavior_match_net_structure = clamp(
            behavior_match_component_strength - behavior_match_component_penalty,
            -1.0,
            1.0,
        )
        behavior_match_decision_bonus = self._behavior_match_decision_bonus(
            best_move=best_move,
            candidate_confidence=behavior_match_candidate_confidence,
        )

        edge_pressure = 0.0
        if second_move is not None and best_gap < self.STOP_EDGE_REFERENCE:
            edge_pressure = self.STOP_MARGIN_WEAK_EDGE * (
                (self.STOP_EDGE_REFERENCE - max(0.0, best_gap)) / self.STOP_EDGE_REFERENCE
            )

        post_hit_continue_margin = best_move.get("post_hit_continue_margin", 0.0)
        rollout_pressure = 0.0
        if best_move.get("post_hit_stop_score", 0.0) > 0.0 and post_hit_continue_margin < 0.0:
            rollout_pressure = self.STOP_MARGIN_WEAK_ROLLOUT * min(
                1.0,
                (-post_hit_continue_margin) / self.ROLLOUT_MARGIN_REFERENCE,
            )

        post_hit_best_gap = best_move.get("post_hit_best_gap", 0.0)
        fragile_rollout_pressure = 0.0
        if (
            best_move.get("post_hit_stop_score", 0.0) > 0.0
            and post_hit_continue_margin > 0.0
            and post_hit_best_gap < self.POST_HIT_GAP_REFERENCE
        ):
            fragile_rollout_pressure = self.STOP_MARGIN_FRAGILE_POST_HIT * (
                (self.POST_HIT_GAP_REFERENCE - max(0.0, post_hit_best_gap)) / self.POST_HIT_GAP_REFERENCE
            )

        post_hit_top_k_continue_margin = best_move.get("post_hit_top_k_continue_margin", 0.0)
        top_k_rollout_pressure = 0.0
        if (
            best_move.get("post_hit_stop_score", 0.0) > 0.0
            and post_hit_continue_margin > 0.0
            and post_hit_top_k_continue_margin < post_hit_continue_margin
        ):
            top_k_rollout_pressure = self.STOP_MARGIN_TOP_K_SUPPORT * min(
                1.0,
                1.0 - (
                    post_hit_top_k_continue_margin
                    / max(1e-9, post_hit_continue_margin)
                ),
            )

        attackability_after_hit = best_move.get("attackability_after_hit", 0.0)
        attackability_pressure = 0.0
        if best_move.get("post_hit_stop_score", 0.0) <= 0.0 and attackability_after_hit < self.ATTACKABILITY_REFERENCE:
            attackability_pressure = self.STOP_MARGIN_LOW_ATTACKABILITY * (
                (self.ATTACKABILITY_REFERENCE - attackability_after_hit)
                / max(self.ATTACKABILITY_REFERENCE, 1e-9)
            )

        behavior_rollout_pressure = 0.0
        if (
            best_move.get("post_hit_stop_score", 0.0) > 0.0
            and post_hit_continue_margin > 0.0
            and best_move.get("continuation_value", 0.0) > 0.0
        ):
            behavior_rollout_pressure = self.STOP_MARGIN_BEHAVIOR_ROLLOUT * (
                1.0 - behavior_match_candidate_confidence
            )

        stop_score = (
            stop_threshold
            + edge_pressure
            + rollout_pressure
            + fragile_rollout_pressure
            + top_k_rollout_pressure
            + attackability_pressure
            + behavior_rollout_pressure
        )
        continue_score = best_move["expected_value"] + behavior_match_decision_bonus
        post_hit_behavior_support_breakdown = self._post_hit_behavior_support_breakdown(
            best_move=best_move,
        )
        post_hit_behavior_support_adjustment = post_hit_behavior_support_breakdown["adjustment"]
        continue_score += post_hit_behavior_support_adjustment
        pre_structure_continue_margin = continue_score - stop_score
        behavior_match_decision_structure_adjustment = self._behavior_match_decision_structure_adjustment(
            decision_bonus=behavior_match_decision_bonus,
            net_structure=behavior_match_net_structure,
            component_support=behavior_match_component_support,
            context_focus=behavior_match_context_focus,
            pre_structure_continue_margin=pre_structure_continue_margin,
        )
        continue_score += behavior_match_decision_structure_adjustment
        continue_margin = continue_score - stop_score

        low_confidence_guard = (
            my_hidden_count <= 2
            and best_move["win_probability"] < 0.45
            and best_move.get("continuation_likelihood", 0.0) < 0.55
            and continue_margin < self.LOW_CONFIDENCE_GUARD_MARGIN
        )
        weak_edge_guard = (
            second_move is not None
            and best_gap < 0.10
            and continue_margin < self.WEAK_EDGE_GUARD_MARGIN
        )

        should_continue = (
            continue_margin > 0.0
            and not low_confidence_guard
            and not weak_edge_guard
        )
        return {
            "should_continue": should_continue,
            "continue_score": continue_score,
            "stop_score": stop_score,
            "continue_margin": continue_margin,
            "best_gap": best_gap,
            "behavior_match_decision_bonus": behavior_match_decision_bonus,
            "behavior_match_decision_structure_adjustment": behavior_match_decision_structure_adjustment,
            "behavior_match_net_structure": behavior_match_net_structure,
            "behavior_match_candidate_confidence": behavior_match_candidate_confidence,
            "behavior_match_component_support": behavior_match_component_support,
            "behavior_match_component_strength": behavior_match_component_strength,
            "behavior_match_component_penalty": behavior_match_component_penalty,
            "behavior_match_context_focus": behavior_match_context_focus,
            "low_confidence_guard": low_confidence_guard,
            "weak_edge_guard": weak_edge_guard,
            "decision_score_breakdown": {
                "base_stop_threshold": stop_threshold,
                "edge_pressure": edge_pressure,
                "rollout_pressure": rollout_pressure,
                "fragile_rollout_pressure": fragile_rollout_pressure,
                "top_k_rollout_pressure": top_k_rollout_pressure,
                "attackability_pressure": attackability_pressure,
                "behavior_rollout_pressure": behavior_rollout_pressure,
                "post_hit_behavior_support_adjustment": post_hit_behavior_support_adjustment,
                "post_hit_behavior_support_gain": post_hit_behavior_support_breakdown["support_gain"],
                "post_hit_behavior_fragility_drag": post_hit_behavior_support_breakdown["fragility_drag"],
                "post_hit_behavior_support_signal": post_hit_behavior_support_breakdown["support_signal"],
                "post_hit_behavior_fragility_signal": post_hit_behavior_support_breakdown["fragility_signal"],
                "post_hit_behavior_support_strength": post_hit_behavior_support_breakdown["support_strength"],
                "post_hit_behavior_fragility_strength": post_hit_behavior_support_breakdown["fragility_strength"],
                "behavior_match_decision_bonus": behavior_match_decision_bonus,
                "behavior_match_decision_structure_adjustment": behavior_match_decision_structure_adjustment,
                "behavior_match_net_structure": behavior_match_net_structure,
                "behavior_match_candidate_confidence": behavior_match_candidate_confidence,
                "behavior_match_component_support": behavior_match_component_support,
                "behavior_match_component_strength": behavior_match_component_strength,
                "behavior_match_component_penalty": behavior_match_component_penalty,
                "behavior_match_context_focus": behavior_match_context_focus,
            },
        }

    def _behavior_match_candidate_confidence_breakdown(
        self,
        *,
        best_move: Dict[str, Any],
    ) -> Dict[str, float]:
        precomputed_keys = (
            "behavior_match_candidate_confidence",
            "behavior_match_component_support",
            "behavior_match_component_strength",
            "behavior_match_component_penalty",
            "behavior_match_context_focus",
        )
        if all(key in best_move for key in precomputed_keys):
            return {
                "candidate_confidence": float(best_move["behavior_match_candidate_confidence"]),
                "component_support": float(best_move["behavior_match_component_support"]),
                "component_strength": float(best_move["behavior_match_component_strength"]),
                "component_penalty": float(best_move["behavior_match_component_penalty"]),
                "context_focus": float(best_move["behavior_match_context_focus"]),
            }

        candidate_signal = best_move.get("behavior_candidate_signal")
        if not isinstance(candidate_signal, dict):
            return {
                "candidate_confidence": 1.0,
                "component_support": 1.0,
                "component_strength": 1.0,
                "component_penalty": 0.0,
                "context_focus": 1.0,
            }

        mode = str(candidate_signal.get("mode", ""))
        if mode == "map_context_fallback":
            return {
                "candidate_confidence": 1.0,
                "component_support": 1.0,
                "component_strength": 1.0,
                "component_penalty": 0.0,
                "context_focus": 1.0,
            }

        dominant_signal = candidate_signal.get("dominant_signal", {})
        posterior_support = clamp(
            float(dominant_signal.get("posterior_support", 0.0)),
            0.0,
            1.0,
        )
        context_covered_probability = clamp(
            float(candidate_signal.get("context_covered_probability", 0.0)),
            0.0,
            1.0,
        )
        component_strength = self._behavior_match_component_strength(
            candidate_signal=candidate_signal,
        )
        component_penalty = self._behavior_match_component_penalty(
            candidate_signal=candidate_signal,
        )
        component_support = self._behavior_match_component_support(
            candidate_signal=candidate_signal,
            dominant_signal_support=posterior_support,
            component_strength=component_strength,
            component_penalty=component_penalty,
        )
        context_focus = self._behavior_match_context_focus(candidate_signal=candidate_signal)
        if (
            mode != "neighbor_top_k_posterior"
            and posterior_support <= 0.0
            and context_covered_probability <= 0.0
        ):
            return {
                "candidate_confidence": 1.0,
                "component_support": 1.0,
                "component_strength": 1.0,
                "component_penalty": 0.0,
                "context_focus": 1.0,
            }
        candidate_confidence = clamp(
            (0.30 * posterior_support)
            + (0.30 * context_covered_probability)
            + (0.25 * component_support)
            + (0.15 * context_focus),
            0.0,
            1.0,
        )
        return {
            "candidate_confidence": candidate_confidence,
            "component_support": component_support,
            "component_strength": component_strength,
            "component_penalty": component_penalty,
            "context_focus": context_focus,
        }

    def _behavior_match_ranking_breakdown(
        self,
        *,
        best_move: Dict[str, Any],
    ) -> Dict[str, float]:
        confidence_breakdown = self._behavior_match_candidate_confidence_breakdown(
            best_move=best_move,
        )
        raw_bonus = max(0.0, float(best_move.get("behavior_match_bonus", 0.0)))
        net_structure = clamp(
            confidence_breakdown["component_strength"]
            - confidence_breakdown["component_penalty"],
            -1.0,
            1.0,
        )
        structure_adjustment = (
            raw_bonus
            * self.BEHAVIOR_MATCH_NET_STRUCTURE_SCALE
            * net_structure
        )
        ranking_bonus = max(
            0.0,
            (raw_bonus * confidence_breakdown["candidate_confidence"])
            + structure_adjustment,
        )
        return {
            "ranking_bonus": ranking_bonus,
            "net_structure": net_structure,
            "structure_adjustment": structure_adjustment,
        }

    def _behavior_match_component_strength(
        self,
        *,
        candidate_signal: Dict[str, Any],
    ) -> float:
        component_strengths = []
        for component_name in ("progressive", "anchor", "boundary"):
            component = candidate_signal.get(component_name)
            if not isinstance(component, dict):
                continue
            component_strengths.append(
                clamp(
                    (float(component.get("weight", 1.0)) - 1.0)
                    / self.BEHAVIOR_MATCH_COMPONENT_WEIGHT_REFERENCE,
                    0.0,
                    1.0,
                )
            )
        if component_strengths:
            return sum(component_strengths) / len(component_strengths)
        return 0.0

    def _behavior_match_component_penalty(
        self,
        *,
        candidate_signal: Dict[str, Any],
    ) -> float:
        component_penalties = []
        for component_name in ("progressive", "anchor", "boundary"):
            component = candidate_signal.get(component_name)
            if not isinstance(component, dict):
                continue
            component_penalties.append(
                clamp(
                    (1.0 - float(component.get("weight", 1.0)))
                    / self.BEHAVIOR_MATCH_COMPONENT_WEIGHT_REFERENCE,
                    0.0,
                    1.0,
                )
            )
        if component_penalties:
            return sum(component_penalties) / len(component_penalties)
        return 0.0

    def _behavior_match_component_support(
        self,
        *,
        candidate_signal: Dict[str, Any],
        dominant_signal_support: float,
        component_strength: float,
        component_penalty: float,
    ) -> float:
        component_supports = []
        for component_name in ("progressive", "anchor", "boundary"):
            component = candidate_signal.get(component_name)
            if not isinstance(component, dict):
                continue
            component_supports.append(
                clamp(float(component.get("posterior_support", 0.0)), 0.0, 1.0)
            )
        if component_supports:
            average_support = sum(component_supports) / len(component_supports)
        else:
            average_support = dominant_signal_support
        return clamp(
            (0.75 * average_support)
            + (0.25 * component_strength)
            - (self.BEHAVIOR_MATCH_COMPONENT_PENALTY_WEIGHT * component_penalty),
            0.0,
            1.0,
        )

    def _behavior_match_context_focus(
        self,
        *,
        candidate_signal: Dict[str, Any],
    ) -> float:
        context_candidate_count = max(
            1.0,
            float(candidate_signal.get("context_candidate_count", 1.0)),
        )
        return min(
            1.0,
            sqrt(self.BEHAVIOR_MATCH_CONTEXT_COUNT_REFERENCE / context_candidate_count),
        )

    def _behavior_match_decision_bonus(
        self,
        *,
        best_move: Dict[str, Any],
        candidate_confidence: float,
    ) -> float:
        raw_bonus = max(0.0, best_move.get("behavior_match_bonus", 0.0))
        if raw_bonus <= 0.0:
            return 0.0

        support_scale = min(
            1.0,
            max(0.0, float(best_move.get("behavior_match_support", 0.0)))
            / self.BEHAVIOR_MATCH_SUPPORT_REFERENCE,
        )
        stable_ratio = clamp(
            float(best_move.get("behavior_guidance_stable_ratio", 0.0)),
            0.0,
            1.0,
        )
        return (
            raw_bonus
            * self.BEHAVIOR_MATCH_DECISION_SCALE
            * support_scale
            * stable_ratio
            * clamp(candidate_confidence, 0.0, 1.0)
        )

    def _post_hit_behavior_support_breakdown(
        self,
        *,
        best_move: Dict[str, Any],
    ) -> Dict[str, float]:
        if (
            float(best_move.get("post_hit_stop_score", 0.0)) <= 0.0
            or float(best_move.get("post_hit_continue_margin", 0.0)) <= 0.0
            or float(best_move.get("continuation_value", 0.0)) <= 0.0
        ):
            return {
                "adjustment": 0.0,
                "support_gain": 0.0,
                "fragility_drag": 0.0,
                "support_signal": 0.0,
                "fragility_signal": 0.0,
                "support_strength": 0.0,
                "fragility_strength": 0.0,
            }

        ranking_edge = clamp(
            (
                float(best_move.get("post_hit_top_k_continue_margin", 0.0))
                - float(best_move.get("post_hit_top_k_expected_continue_margin", 0.0))
            ) / self.POST_HIT_BEHAVIOR_SUPPORT_REFERENCE,
            -1.0,
            1.0,
        )
        guidance_edge = clamp(
            (
                float(best_move.get("post_hit_guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER))
                - 1.0
            ) / 0.12,
            -1.0,
            1.0,
        )
        support_ratio_edge = clamp(
            float(best_move.get("post_hit_top_k_support_ratio", 0.0))
            - float(best_move.get("post_hit_top_k_expected_support_ratio", 0.0)),
            -1.0,
            1.0,
        )
        support_strength = clamp(
            (0.5 * float(best_move.get("post_hit_guidance_stable_ratio", 0.0)))
            + (0.5 * float(best_move.get("post_hit_guidance_support", 0.0))),
            0.0,
            1.0,
        )
        fragility_strength = clamp(
            (0.60 * (1.0 - float(best_move.get("post_hit_guidance_stable_ratio", 0.0))))
            + (0.40 * (1.0 - float(best_move.get("post_hit_guidance_support", 0.0)))),
            0.0,
            1.0,
        )
        support_signal = clamp(
            (0.55 * max(0.0, ranking_edge))
            + (0.25 * max(0.0, guidance_edge))
            + (0.20 * max(0.0, support_ratio_edge)),
            0.0,
            1.0,
        )
        fragility_signal = clamp(
            (0.50 * max(0.0, -ranking_edge))
            + (0.20 * max(0.0, -guidance_edge))
            + (0.30 * max(0.0, -support_ratio_edge)),
            0.0,
            1.0,
        )
        support_gain = self.POST_HIT_BEHAVIOR_SUPPORT_SCALE * support_signal * support_strength
        fragility_drag = self.POST_HIT_BEHAVIOR_SUPPORT_SCALE * fragility_signal * fragility_strength
        return {
            "adjustment": support_gain - fragility_drag,
            "support_gain": support_gain,
            "fragility_drag": fragility_drag,
            "support_signal": support_signal,
            "fragility_signal": fragility_signal,
            "support_strength": support_strength,
            "fragility_strength": fragility_strength,
        }

    def _behavior_match_decision_structure_adjustment(
        self,
        *,
        decision_bonus: float,
        net_structure: float,
        component_support: float,
        context_focus: float,
        pre_structure_continue_margin: float,
    ) -> float:
        decision_bonus = max(0.0, float(decision_bonus))
        if decision_bonus <= 0.0:
            return 0.0

        margin = abs(float(pre_structure_continue_margin))
        if margin >= self.BEHAVIOR_MATCH_DECISION_WINDOW:
            return 0.0

        support_gate = (
            clamp(component_support, 0.0, 1.0)
            * clamp(context_focus, 0.0, 1.0)
        )
        edge_scale = 1.0 - (margin / self.BEHAVIOR_MATCH_DECISION_WINDOW)
        return (
            decision_bonus
            * self.BEHAVIOR_MATCH_DECISION_NET_STRUCTURE_SCALE
            * clamp(net_structure, -1.0, 1.0)
            * support_gate
            * edge_scale
        )

    def _stop_threshold(
        self,
        *,
        risk_factor: float,
        my_hidden_count: int,
        best_move: Optional[Dict[str, Any]],
    ) -> float:
        threshold = self.STOP_MARGIN_BASE + max(0.0, (risk_factor - self.HIT_REWARD) * self.STOP_MARGIN_RISK_SCALE)
        if my_hidden_count <= 2:
            threshold += self.STOP_MARGIN_SHORT_HAND
        if my_hidden_count <= 1:
            threshold += self.STOP_MARGIN_SHORT_HAND

        if best_move is not None:
            if best_move["win_probability"] < 0.5:
                threshold += self.STOP_MARGIN_LOW_CONFIDENCE * (0.5 - best_move["win_probability"]) / 0.5
            if best_move.get("continuation_likelihood", 0.0) < 0.5:
                threshold += self.STOP_MARGIN_WEAK_CONTINUATION * (0.5 - best_move.get("continuation_likelihood", 0.0)) / 0.5
        return threshold

    def _build_stop_reason(
        self,
        *,
        should_continue: bool,
        best_move: Dict[str, Any],
        stop_threshold: float,
        stop_score: float,
        continue_score: float,
        continue_margin: float,
        low_confidence_guard: bool,
        weak_edge_guard: bool,
    ) -> str:
        if should_continue:
            if best_move.get("continuation_value", 0.0) > 0.0:
                return (
                    f"继续评分 {continue_score:.2f} 高于停手评分 {stop_score:.2f}，"
                    "命中后继续收益也支持进攻。"
                )
            return f"继续评分 {continue_score:.2f} 高于停手评分 {stop_score:.2f}，可以继续进攻。"
        if low_confidence_guard:
            return "当前属于高暴露局面，命中率与续压潜力都偏弱，建议停手。"
        if weak_edge_guard:
            return "最佳动作与次优动作差距过小，当前进攻边际优势不足，建议停手。"
        if stop_score > stop_threshold:
            return (
                f"继续评分 {continue_score:.2f} 仍低于停手评分 {stop_score:.2f}；"
                f"当前续压空间不足，净优势 {continue_margin:.2f}。"
            )
        return f"最佳动作期望收益 {best_move['expected_value']:.2f} 未超过停手阈值 {stop_threshold:.2f}。"

    def _build_reason(
        self,
        probability: float,
        info_gain: float,
        continuation_value: float,
        miss_penalty: float,
        continuation_likelihood: float,
    ) -> str:
        if continuation_value > max(0.35, info_gain * self.INFORMATION_GAIN_WEIGHT):
            if continuation_likelihood >= 0.60:
                return "命中后继续收益较高，且续压概率也偏高，适合主动压回合。"
            return "命中后存在续压价值，但更依赖后续局面兑现。"
        if probability >= 0.62 and miss_penalty < self.HIT_REWARD:
            return "当前命中率较高，且风险处于可接受范围。"
        if info_gain * self.INFORMATION_GAIN_WEIGHT >= max(0.45, continuation_value):
            return "即便不是最稳的一猜，也能显著压缩该位置的不确定性。"
        return "这步主要依靠当前命中率与风险平衡取得正收益。"

    def _best_immediate_ev(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
    ) -> float:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        best_value = 0.0
        for probability_matrix in full_probability_matrix.values():
            for slot_distribution in probability_matrix.values():
                for card, probability in slot_distribution.items():
                    info_gain = self._expected_slot_entropy_reduction(slot_distribution, card)
                    value = (probability * self.HIT_REWARD) - ((1.0 - probability) * risk_factor) + (info_gain * self.INFORMATION_GAIN_WEIGHT)
                    best_value = max(best_value, value)
        return best_value

    def _success_posterior(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> FullProbabilityMatrix:
        success_matrix: FullProbabilityMatrix = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_positions: ProbabilityMatrix = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if player_id == target_player_id and slot_index == target_slot_index:
                    continue

                adjusted = dict(slot_distribution)
                if guessed_card in adjusted:
                    adjusted.pop(guessed_card, None)
                    adjusted = normalize_card_distribution(adjusted)

                if adjusted:
                    player_positions[slot_index] = adjusted
            if player_positions:
                success_matrix[player_id] = player_positions
        return success_matrix

    def _expected_slot_entropy_reduction(
        self,
        slot_distribution: Dict[Card, float],
        guessed_card: Card,
    ) -> float:
        before = self._entropy(slot_distribution.values())
        hit_probability = slot_distribution.get(guessed_card, 0.0)
        if hit_probability <= 0.0:
            return 0.0
        if hit_probability >= 1.0:
            return before

        miss_distribution = {
            card: probability
            for card, probability in slot_distribution.items()
            if card != guessed_card
        }
        miss_distribution = normalize_card_distribution(miss_distribution)
        after_miss = self._entropy(miss_distribution.values())
        return before - ((1.0 - hit_probability) * after_miss)

    def _entropy(self, probabilities: Iterable[float]) -> float:
        entropy = 0.0
        for probability in probabilities:
            if probability <= 0.0:
                continue
            entropy -= probability * log2(probability)
        return entropy


class GameController:
    """Coordinate hard inference, soft weighting, posterior blending, and move scoring."""

    BEHAVIOR_DEBUG_TOP_K = 3

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_engine = DaVinciInferenceEngine(game_state)
        self.behavior_model = BehavioralLikelihoodModel()
        self.decision_engine = DaVinciDecisionEngine()

    def _build_draw_color_summary(
        self,
        full_probability_matrix: Optional[FullProbabilityMatrix] = None,
    ) -> Dict[str, Any]:
        available_counts = {
            color: sum(1 for card in self.inference_engine.available_cards if card[0] == color)
            for color in CARD_COLORS
        }
        self_counts = {
            color: sum(
                1
                for slot in self.game_state.self_player().ordered_slots()
                if getattr(slot, "color", None) == color
            )
            for color in CARD_COLORS
        }
        total_self_cards = max(1, sum(self_counts.values()))
        total_available = max(1, sum(available_counts.values()))

        defense_balance = {
            "B": (self_counts["W"] - self_counts["B"]) / total_self_cards,
            "W": (self_counts["B"] - self_counts["W"]) / total_self_cards,
        }
        hidden_color_mass = {"B": 0.0, "W": 0.0}
        total_hidden_positions = 0.0
        for probability_matrix in (full_probability_matrix or {}).values():
            for slot_distribution in probability_matrix.values():
                total_hidden_positions += 1.0
                for card, probability in slot_distribution.items():
                    hidden_color_mass[card[0]] += float(probability)
        if total_hidden_positions > 0.0:
            offense_pressure = {
                color: hidden_color_mass[color] / total_hidden_positions
                for color in CARD_COLORS
            }
        else:
            offense_pressure = {"B": 0.0, "W": 0.0}
        availability_pressure = {
            color: (
                available_counts[color]
                - available_counts["W" if color == "B" else "B"]
            )
            / total_available
            for color in CARD_COLORS
        }
        color_scores = {
            color: defense_balance[color]
            + (0.35 * offense_pressure[color])
            + (0.18 * availability_pressure[color])
            for color in CARD_COLORS
        }
        recommended_color = max(
            CARD_COLORS,
            key=lambda color: (color_scores[color], available_counts[color], color),
        )
        dominant_factor_margins = {
            "defense_balance": abs(defense_balance["B"] - defense_balance["W"]),
            "offense_pressure": abs(offense_pressure["B"] - offense_pressure["W"]),
            "availability_pressure": abs(
                availability_pressure["B"] - availability_pressure["W"]
            ),
        }
        return {
            "recommended_color": recommended_color,
            "black_score": color_scores["B"],
            "white_score": color_scores["W"],
            "defense_balance_black": defense_balance["B"],
            "defense_balance_white": defense_balance["W"],
            "offense_pressure_black": offense_pressure["B"],
            "offense_pressure_white": offense_pressure["W"],
            "availability_pressure_black": availability_pressure["B"],
            "availability_pressure_white": availability_pressure["W"],
            "available_black_count": available_counts["B"],
            "available_white_count": available_counts["W"],
            "self_black_count": self_counts["B"],
            "self_white_count": self_counts["W"],
            "dominant_factor": max(
                dominant_factor_margins,
                key=dominant_factor_margins.get,
            ),
        }

    def run_turn(self) -> Dict[str, Any]:
        has_any_hidden_slots = bool(self.inference_engine.search_positions) or any(
            self.inference_engine.preassigned_hidden.values()
        )
        target_hidden_slots = self.game_state.target_hidden_slots()
        blocked_target_slots = {key for key in self.inference_engine.publicly_collapsed_slots if key[0] == getattr(self.game_state, "target_player_id", None)}
        my_hidden_count = self.game_state.my_hidden_count()
        default_risk = self.decision_engine.calculate_risk_factor(my_hidden_count)
        draw_color_summary = self._build_draw_color_summary()

        if not has_any_hidden_slots:
            return {
                "best_move": None,
                "top_moves": [],
                "probability_matrix": [],
                "full_probability_matrix": [],
                "hard_probability_matrix": [],
                "hard_full_probability_matrix": [],
                "soft_full_probability_matrix": [],
                "search_space_size": 0,
                "effective_weight_sum": 0.0,
                "opponent_hidden_count": 0,
                "risk_factor": default_risk,
                "behavior_debug": {
                    "hypothesis_source": "target_slot_top_k_posterior_with_map_context",
                    "aggregation_top_k": self.BEHAVIOR_DEBUG_TOP_K,
                    "signal_count": 0,
                    "map_signals": [],
                    "signals": [],
                },
                "behavior_guidance_profile": {
                    "signal_count": 0.0,
                    "average_posterior_support": 0.0,
                    "average_weighted_strength": 0.0,
                    "stable_signal_ratio": 0.0,
                    "guidance_multiplier": 1.0,
                    "source_support_progressive": 0.0,
                    "source_support_same_color_anchor": 0.0,
                    "source_support_local_boundary": 0.0,
                },
                "draw_color_summary": draw_color_summary,
                "should_stop": True,
            }

        guess_signals_by_player = self.behavior_model.build_guess_signals(self.game_state)
        hard_full_probability_matrix, soft_full_probability_matrix, full_probability_matrix, search_space_size, total_soft_weight = self.inference_engine.infer_hidden_probabilities(
            guess_signals_by_player,
            self.behavior_model,
        )
        behavior_map_hypothesis = self._map_hypothesis_from_matrix(full_probability_matrix)
        behavior_debug = self._build_behavior_debug(
            full_probability_matrix=full_probability_matrix,
            guess_signals_by_player=guess_signals_by_player,
            map_hypothesis=behavior_map_hypothesis,
        )
        behavior_guidance_profile = self._build_behavior_guidance_profile(
            behavior_debug_signals=behavior_debug["signals"],
            acting_player_id=self.game_state.self_player_id,
        )
        draw_color_summary = self._build_draw_color_summary(full_probability_matrix)

        hidden_index_by_player = {
            player_id: self.game_state.hidden_index_by_slot(player_id)
            for player_id in full_probability_matrix
        }
        all_moves, risk_factor = self.decision_engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=my_hidden_count,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=self.behavior_model,
            guess_signals_by_player=guess_signals_by_player,
            acting_player_id=self.game_state.self_player_id,
            behavior_guidance_profile=behavior_guidance_profile,
            game_state=self.game_state,
            behavior_map_hypothesis=behavior_map_hypothesis,
            blocked_slots=self.inference_engine.publicly_collapsed_slots,
        )
        best_move, decision_summary = self.decision_engine.choose_best_move(
            all_moves,
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
        )

        target_player_id = getattr(self.game_state, "target_player_id", None)
        target_probability_matrix = full_probability_matrix.get(target_player_id, {})
        target_hard_matrix = hard_full_probability_matrix.get(target_player_id, {})

        return {
            "best_move": best_move,
            "top_moves": all_moves[:10],
            "probability_matrix": self._serialize_probability_matrix(
                target_probability_matrix,
                self.game_state.hidden_index_by_slot(target_player_id) if target_player_id is not None else {},
            ),
            "full_probability_matrix": self._serialize_full_probability_matrix(
                full_probability_matrix,
            ),
            "hard_probability_matrix": self._serialize_probability_matrix(
                target_hard_matrix,
                self.game_state.hidden_index_by_slot(target_player_id) if target_player_id is not None else {},
            ),
            "hard_full_probability_matrix": self._serialize_full_probability_matrix(
                hard_full_probability_matrix,
            ),
            "soft_full_probability_matrix": self._serialize_full_probability_matrix(
                soft_full_probability_matrix,
            ),
            "search_space_size": search_space_size,
            "opponent_hidden_count": max(0, len(target_hidden_slots) - len(blocked_target_slots)),
            "risk_factor": risk_factor,
            "effective_weight_sum": total_soft_weight,
            "behavior_blend": SOFT_BEHAVIOR_BLEND,
            "behavior_debug": behavior_debug,
            "behavior_guidance_profile": behavior_guidance_profile,
            "draw_color_summary": draw_color_summary,
            "decision_summary": decision_summary,
            "should_stop": best_move is None,
        }

    def _serialize_probability_matrix(
        self,
        probability_matrix: ProbabilityMatrix,
        hidden_index_by_slot: Dict[int, int],
    ) -> List[Dict[str, Any]]:
        serialized_positions: List[Dict[str, Any]] = []
        for target_slot_index in sorted(probability_matrix):
            sorted_candidates = sorted(
                probability_matrix[target_slot_index].items(),
                key=lambda item: (-item[1], card_sort_key(item[0])),
            )
            serialized_positions.append(
                {
                    "target_index": hidden_index_by_slot.get(target_slot_index, target_slot_index),
                    "target_slot_index": target_slot_index,
                    "target_scope": "player_slots",
                    "candidates": [
                        {
                            "card": serialize_card(card),
                            "probability": probability,
                        }
                        for card, probability in sorted_candidates
                    ],
                }
            )
        return serialized_positions

    def _serialize_full_probability_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> List[Dict[str, Any]]:
        serialized_players: List[Dict[str, Any]] = []
        for player_id in full_probability_matrix:
            hidden_index_by_slot = self.game_state.hidden_index_by_slot(player_id)
            serialized_players.append(
                {
                    "player_id": player_id,
                    "positions": self._serialize_probability_matrix(
                        full_probability_matrix[player_id],
                        hidden_index_by_slot,
                    ),
                }
            )
        return serialized_players

    def _map_hypothesis_from_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Dict[str, Dict[int, Card]]:
        hypothesis_by_player: Dict[str, Dict[int, Card]] = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_hypothesis: Dict[int, Card] = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if not slot_distribution:
                    continue
                best_card = max(
                    slot_distribution.items(),
                    key=lambda item: (item[1], -card_sort_key(item[0])[0], -card_sort_key(item[0])[1]),
                )[0]
                player_hypothesis[slot_index] = best_card
            if player_hypothesis:
                hypothesis_by_player[player_id] = player_hypothesis
        return hypothesis_by_player

    def _build_behavior_debug(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        map_hypothesis: Optional[Dict[str, Dict[int, Card]]] = None,
    ) -> Dict[str, Any]:
        map_hypothesis = map_hypothesis or self._map_hypothesis_from_matrix(full_probability_matrix)
        map_signals = self.behavior_model.explain_guess_signals(
            map_hypothesis,
            guess_signals_by_player,
            self.game_state,
        )
        aggregated_signals = [
            self._aggregate_behavior_signal_debug(
                signal=signal,
                full_probability_matrix=full_probability_matrix,
                map_hypothesis=map_hypothesis,
            )
            for player_id in sorted(guess_signals_by_player)
            for signal in guess_signals_by_player[player_id]
        ]
        return {
            "hypothesis_source": "target_slot_top_k_posterior_with_map_context",
            "aggregation_top_k": self.BEHAVIOR_DEBUG_TOP_K,
            "signal_count": sum(len(signals) for signals in guess_signals_by_player.values()),
            "map_signals": map_signals,
            "signals": aggregated_signals,
        }

    def _aggregate_behavior_signal_debug(
        self,
        *,
        signal: GuessSignal,
        full_probability_matrix: FullProbabilityMatrix,
        map_hypothesis: Dict[str, Dict[int, Card]],
    ) -> Dict[str, Any]:
        map_explanation = self.behavior_model.explain_signal(
            map_hypothesis,
            self.game_state,
            signal,
        )
        slot_distribution = (
            full_probability_matrix
            .get(signal.target_player_id, {})
            .get(signal.target_slot_index, {})
        )
        ranked_candidates = sorted(
            slot_distribution.items(),
            key=lambda item: (-item[1], card_sort_key(item[0])),
        )[: self.BEHAVIOR_DEBUG_TOP_K]
        covered_probability = sum(probability for _, probability in ranked_candidates)
        if not ranked_candidates or covered_probability <= 0.0:
            fallback_value_selection = dict(map_explanation["value_selection"])
            fallback_value_selection.update(
                {
                    "mode": "map_only_fallback",
                    "covered_probability": 0.0,
                    "candidate_count": 0,
                    "reason_support": [],
                    "source_support": [],
                }
            )
            return {
                **map_explanation,
                "aggregation_mode": "map_only_fallback",
                "candidate_count": 0,
                "covered_probability": 0.0,
                "candidate_explanations": [],
                "map_explanation": map_explanation,
                "value_selection": fallback_value_selection,
            }

        normalized_total = max(self.behavior_model.EPSILON, covered_probability)
        weighted_component_sums: DefaultDict[str, float] = defaultdict(float)
        weighted_value_component_sums: DefaultDict[str, float] = defaultdict(float)
        reason_support: DefaultDict[str, float] = defaultdict(float)
        source_support: DefaultDict[str, float] = defaultdict(float)
        source_strength: DefaultDict[str, float] = defaultdict(float)
        candidate_explanations: List[Dict[str, Any]] = []

        for card, probability in ranked_candidates:
            normalized_weight = probability / normalized_total
            candidate_hypothesis = self._replace_hypothesis_card(
                map_hypothesis,
                signal.target_player_id,
                signal.target_slot_index,
                card,
            )
            explanation = self.behavior_model.explain_signal(
                candidate_hypothesis,
                self.game_state,
                signal,
            )
            candidate_explanations.append(
                {
                    "hypothesis_target_card": serialize_card(card),
                    "posterior_probability": probability,
                    "normalized_weight": normalized_weight,
                    "dominant_signal": explanation["value_selection"]["dominant_signal"],
                    "signal_tags": explanation["value_selection"]["signal_tags"],
                    "value_selection_total_weight": explanation["value_selection"]["total_weight"],
                }
            )
            for component_name, weight in explanation["component_weights"].items():
                weighted_component_sums[component_name] += normalized_weight * weight

            value_selection = explanation["value_selection"]
            weighted_value_component_sums["total_weight"] += normalized_weight * value_selection["total_weight"]
            weighted_value_component_sums["progressive"] += normalized_weight * value_selection["progressive"]["weight"]
            weighted_value_component_sums["anchor"] += normalized_weight * value_selection["anchor"]["weight"]
            weighted_value_component_sums["boundary"] += normalized_weight * value_selection["boundary"]["weight"]
            for reason in value_selection["signal_tags"]:
                reason_support[reason] += normalized_weight

            dominant_signal = value_selection["dominant_signal"]
            source = str(dominant_signal.get("source", "neutral"))
            source_support[source] += normalized_weight
            source_strength[source] += normalized_weight * abs(float(dominant_signal.get("weight", 1.0)) - 1.0)

        aggregated_value_selection = {
            "mode": "target_slot_top_k_posterior",
            "covered_probability": covered_probability,
            "candidate_count": len(candidate_explanations),
            "total_weight": weighted_value_component_sums["total_weight"],
            "expected_component_weights": {
                "progressive": weighted_value_component_sums["progressive"],
                "anchor": weighted_value_component_sums["anchor"],
                "boundary": weighted_value_component_sums["boundary"],
            },
            "reason_support": [
                {
                    "reason": reason,
                    "posterior_support": support,
                }
                for reason, support in sorted(
                    reason_support.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "source_support": [
                {
                    "source": source,
                    "posterior_support": source_support[source],
                    "weighted_strength": source_strength[source],
                }
                for source in sorted(
                    source_support,
                    key=lambda item: (-source_strength[item], -source_support[item], item),
                )
            ],
        }
        aggregated_value_selection["dominant_signal"] = self._dominant_posterior_signal(
            aggregated_value_selection["source_support"],
            aggregated_value_selection["reason_support"],
        )

        return {
            **map_explanation,
            "aggregation_mode": "target_slot_top_k_posterior",
            "candidate_count": len(candidate_explanations),
            "covered_probability": covered_probability,
            "candidate_explanations": candidate_explanations,
            "component_weights": dict(weighted_component_sums),
            "map_explanation": map_explanation,
            "value_selection": aggregated_value_selection,
        }

    def _replace_hypothesis_card(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
        card: Card,
    ) -> Dict[str, Dict[int, Card]]:
        cloned = {
            hypothesis_player_id: dict(cards_by_slot)
            for hypothesis_player_id, cards_by_slot in hypothesis_by_player.items()
        }
        player_hypothesis = cloned.setdefault(player_id, {})
        player_hypothesis[slot_index] = card
        return cloned

    def _dominant_posterior_signal(
        self,
        source_support: Sequence[Dict[str, Any]],
        reason_support: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not source_support:
            return {
                "source": "neutral",
                "reason": "neutral",
                "posterior_support": 0.0,
                "weighted_strength": 0.0,
            }

        strongest_source = source_support[0]
        strongest_reason = reason_support[0]["reason"] if reason_support else "neutral"
        strongest_reason_support = reason_support[0]["posterior_support"] if reason_support else 0.0
        return {
            "source": strongest_source["source"],
            "reason": strongest_reason,
            "posterior_support": strongest_source["posterior_support"],
            "weighted_strength": strongest_source["weighted_strength"],
            "reason_support": strongest_reason_support,
        }

    def _build_behavior_guidance_profile(
        self,
        *,
        behavior_debug_signals: Sequence[Dict[str, Any]],
        acting_player_id: Optional[str],
    ) -> Dict[str, float]:
        if acting_player_id is None:
            return {
                "signal_count": 0.0,
                "average_posterior_support": 0.0,
                "average_weighted_strength": 0.0,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 0.0,
            }

        relevant_signals = [
            signal
            for signal in behavior_debug_signals
            if signal.get("guesser_id") == acting_player_id
        ]
        if not relevant_signals:
            return {
                "signal_count": 0.0,
                "average_posterior_support": 0.0,
                "average_weighted_strength": 0.0,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 0.0,
            }

        signal_count = float(len(relevant_signals))
        average_posterior_support = sum(
            float(signal["value_selection"]["dominant_signal"].get("posterior_support", 0.0))
            for signal in relevant_signals
        ) / signal_count
        average_weighted_strength = sum(
            float(signal["value_selection"]["dominant_signal"].get("weighted_strength", 0.0))
            for signal in relevant_signals
        ) / signal_count
        stable_signal_ratio = sum(
            1.0
            for signal in relevant_signals
            if float(signal["value_selection"]["dominant_signal"].get("posterior_support", 0.0)) >= 0.55
        ) / signal_count
        source_support_sums: DefaultDict[str, float] = defaultdict(float)
        for signal in relevant_signals:
            for source_support in signal["value_selection"].get("source_support", []):
                source_support_sums[str(source_support.get("source", "neutral"))] += float(
                    source_support.get("posterior_support", 0.0)
                )
        guidance_multiplier = clamp(
            0.95
            + (0.08 * average_posterior_support)
            + (0.07 * stable_signal_ratio)
            + (0.14 * average_weighted_strength),
            0.93,
            1.12,
        )
        return {
            "signal_count": signal_count,
            "average_posterior_support": average_posterior_support,
            "average_weighted_strength": average_weighted_strength,
            "stable_signal_ratio": stable_signal_ratio,
            "guidance_multiplier": guidance_multiplier,
            "source_support_progressive": source_support_sums["progressive"] / signal_count,
            "source_support_same_color_anchor": source_support_sums["same_color_anchor"] / signal_count,
            "source_support_local_boundary": source_support_sums["local_boundary"] / signal_count,
        }
