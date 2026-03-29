from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
DATA_ROOT = BACKEND_ROOT / "data"
EXTERNAL_TRACE_DIR = DATA_ROOT / "behavior_trace_catalogs"
for candidate_path in (str(BACKEND_ROOT), str(REPO_ROOT)):
    if candidate_path not in sys.path:
        sys.path.insert(0, candidate_path)

from app.core.engine import (  # noqa: E402
    BehavioralLikelihoodModel,
    DaVinciDecisionEngine,
    GameController,
    GameState,
    GuessAction,
    GuessSignal,
)
from backend.tests.test_regression_cases import BEHAVIOR_REGRESSION_CASES  # noqa: E402


def _clone_truth_slots_by_player(
    engine: DaVinciDecisionEngine,
    truth_slots_by_player: Dict[str, Sequence[Any]],
) -> Dict[str, List[Any]]:
    return {
        player_id: engine._clone_truth_slots(slots)
        for player_id, slots in truth_slots_by_player.items()
    }


def _hidden_truth_hypothesis(
    public_state: GameState,
    truth_slots_by_player: Dict[str, Sequence[Any]],
) -> Dict[str, Dict[int, Tuple[str, int]]]:
    hypothesis: Dict[str, Dict[int, Tuple[str, int]]] = {}
    for player_id in public_state.players:
        if player_id == public_state.self_player_id:
            continue
        hidden_slots: Dict[int, Tuple[str, int]] = {}
        for truth_slot in truth_slots_by_player[player_id]:
            if truth_slot.is_revealed:
                continue
            truth_card = truth_slot.known_card()
            if truth_card is None:
                continue
            hidden_slots[truth_slot.slot_index] = truth_card
        if hidden_slots:
            hypothesis[player_id] = hidden_slots
    return hypothesis


def _count_guess_actions(actions: Sequence[GuessAction]) -> int:
    return sum(
        1 for action in actions if getattr(action, "action_type", None) == "guess"
    )


def _candidate_catalog_from_state(
    model: BehavioralLikelihoodModel,
    public_state: GameState,
    truth_slots_by_player: Optional[Dict[str, Sequence[Any]]],
    action: GuessAction,
    hypothesis_by_player: Optional[Dict[str, Dict[int, Tuple[str, int]]]] = None,
) -> List[Dict[str, Any]]:
    guessed_card = action.guessed_card()
    if guessed_card is None:
        return []
    hypothesis = (
        dict(hypothesis_by_player or {})
        if hypothesis_by_player is not None
        else _hidden_truth_hypothesis(public_state, truth_slots_by_player or {})
    )
    signal = GuessSignal(
        action_index=_count_guess_actions(public_state.actions),
        guesser_id=action.guesser_id,
        target_player_id=action.target_player_id,
        target_slot_index=action.target_slot_index,
        guessed_card=guessed_card,
        result=bool(action.result),
        continued_turn=action.continued_turn,
    )
    guesser_cards = model._player_cards(
        public_state,
        signal.guesser_id,
        hypothesis.get(signal.guesser_id, {}),
    )
    breakdown = model._joint_action_generative_probability_breakdown(
        public_state,
        hypothesis,
        signal,
        guesser_cards=guesser_cards,
        include_candidate_catalog=True,
    )
    return list(breakdown.get("candidate_catalog", ()))


def _collect_regression_samples(
    model: BehavioralLikelihoodModel,
) -> List[List[Dict[str, Any]]]:
    samples: List[List[Dict[str, Any]]] = []
    for behavior_case in BEHAVIOR_REGRESSION_CASES:
        for state, hypothesis in (
            (behavior_case.preferred_state, behavior_case.preferred_hypothesis),
            (behavior_case.alternative_state, behavior_case.alternative_hypothesis),
        ):
            guess_actions = [
                action
                for action in state.actions
                if getattr(action, "action_type", None) == "guess"
            ]
            for guess_index, action in enumerate(guess_actions):
                prefix_state = GameState(
                    self_player_id=state.self_player_id,
                    target_player_id=state.target_player_id,
                    players=deepcopy(state.players),
                    actions=list(guess_actions[:guess_index]),
                )
                catalog = _candidate_catalog_from_state(
                    model,
                    prefix_state,
                    None,
                    action,
                    hypothesis_by_player=hypothesis,
                )
                if catalog:
                    samples.append(catalog)
    return samples


def _collect_self_play_samples(
    model: BehavioralLikelihoodModel,
    *,
    seeds: Sequence[int] = tuple(range(11, 107, 8)),
    games_per_seed: int = 5,
) -> List[List[Dict[str, Any]]]:
    engine = DaVinciDecisionEngine()
    samples: List[List[Dict[str, Any]]] = []
    for seed in seeds:
        rng = Random(seed)
        for game_index in range(games_per_seed):
            world = engine._build_long_self_play_world(rng)
            truth_slots_by_player = {
                player_id: engine._clone_truth_slots(slots)
                for player_id, slots in world["truth_slots_by_player"].items()
            }
            remaining_deck = list(world["remaining_deck"])
            actions: List[GuessAction] = list(world["actions"])
            guess_public_states: List[GameState] = []
            guess_truth_snapshots: List[Dict[str, List[Any]]] = []
            acting_player_id = "p0" if game_index % 2 == 0 else "p1"
            winner = None

            for _ in range(engine.LONG_SELF_PLAY_MAX_TURNS):
                target_player_id = "p1" if acting_player_id == "p0" else "p0"
                requested_color = engine._benchmark_recommended_draw_color(
                    truth_slots_by_player=truth_slots_by_player,
                    actions=actions,
                    acting_player_id=acting_player_id,
                    target_player_id=target_player_id,
                    remaining_deck=remaining_deck,
                )
                drawn_card = engine._draw_remaining_card_by_color(
                    remaining_deck,
                    color=requested_color,
                    rng=rng,
                )
                if drawn_card is None:
                    fallback_color = "W" if requested_color == "B" else "B"
                    drawn_card = engine._draw_remaining_card_by_color(
                        remaining_deck,
                        color=fallback_color,
                        rng=rng,
                    )
                if drawn_card is None:
                    break

                previous_slots = engine._clone_truth_slots(
                    truth_slots_by_player[acting_player_id]
                )
                updated_slots = [
                    replace(slot, is_newly_drawn=False)
                    for slot in truth_slots_by_player[acting_player_id]
                ]
                updated_slots.append(
                    replace(
                        previous_slots[0],
                        slot_index=-1,
                        color=drawn_card[0],
                        value=drawn_card[1],
                        is_revealed=False,
                        is_newly_drawn=True,
                    )
                )
                updated_slots = engine._sorted_truth_slots(updated_slots)
                actions = engine._remap_actions_for_reindexed_player(
                    actions,
                    player_id=acting_player_id,
                    previous_slots=previous_slots,
                    next_slots=updated_slots,
                )
                truth_slots_by_player[acting_player_id] = updated_slots
                drawn_slot_index = next(
                    slot.slot_index for slot in updated_slots if slot.is_newly_drawn
                )
                last_success_action_index = None

                while True:
                    public_state = engine._build_perspective_state_from_truth_slots(
                        truth_slots_by_player=truth_slots_by_player,
                        actions=actions,
                        self_player_id=acting_player_id,
                        target_player_id=target_player_id,
                    )
                    post_draw_controller = GameController(public_state)
                    post_draw_controller.decision_engine.DEEP_ROLLOUT_DEPTH = 1
                    post_draw_result = post_draw_controller.run_turn(
                        include_draw_color_summary=False,
                    )
                    best_move = post_draw_result.get("best_move")
                    if (
                        best_move is None
                        and post_draw_result.get("strategy_phase") == "post_draw_opening"
                    ):
                        top_moves = list(post_draw_result.get("top_moves", ()))
                        if top_moves:
                            best_move = top_moves[0]
                    if best_move is None:
                        if last_success_action_index is not None:
                            actions[last_success_action_index] = replace(
                                actions[last_success_action_index],
                                continued_turn=False,
                            )
                        break

                    guess_card = best_move.get("guess_card")
                    target_slot_index = int(best_move.get("target_slot_index", -1))
                    target_truth_slot = next(
                        (
                            slot
                            for slot in truth_slots_by_player[target_player_id]
                            if slot.slot_index == target_slot_index
                        ),
                        None,
                    )
                    if (
                        target_truth_slot is None
                        or not isinstance(guess_card, (list, tuple))
                        or len(guess_card) != 2
                    ):
                        break
                    guessed_card = (guess_card[0], guess_card[1])
                    actual_card = target_truth_slot.known_card()
                    guess_public_states.append(public_state)
                    guess_truth_snapshots.append(
                        _clone_truth_slots_by_player(engine, truth_slots_by_player)
                    )
                    if actual_card == guessed_card:
                        actions.append(
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
                                revealed_color=actual_card[0],
                                revealed_value=actual_card[1],
                            )
                        )
                        last_success_action_index = len(actions) - 1
                        updated_target_slots = []
                        for slot in truth_slots_by_player[target_player_id]:
                            if slot.slot_index == target_slot_index:
                                updated_target_slots.append(
                                    replace(slot, is_revealed=True)
                                )
                            else:
                                updated_target_slots.append(slot)
                        truth_slots_by_player[target_player_id] = updated_target_slots
                        if engine._all_revealed(truth_slots_by_player[target_player_id]):
                            winner = acting_player_id
                            break
                        continue

                    revealed_drawn_card = None
                    updated_self_slots = []
                    for slot in truth_slots_by_player[acting_player_id]:
                        if slot.slot_index == drawn_slot_index:
                            revealed_drawn_card = slot.known_card()
                            updated_self_slots.append(
                                replace(slot, is_revealed=True)
                            )
                        else:
                            updated_self_slots.append(slot)
                    truth_slots_by_player[acting_player_id] = updated_self_slots
                    if revealed_drawn_card is not None:
                        actions.append(
                            GuessAction(
                                guesser_id=acting_player_id,
                                target_player_id=target_player_id,
                                target_slot_index=target_slot_index,
                                guessed_color=guessed_card[0],
                                guessed_value=guessed_card[1],
                                result=False,
                                continued_turn=False,
                                revealed_player_id=acting_player_id,
                                revealed_slot_index=drawn_slot_index,
                                revealed_color=revealed_drawn_card[0],
                                revealed_value=revealed_drawn_card[1],
                            )
                        )
                    if engine._all_revealed(truth_slots_by_player[acting_player_id]):
                        winner = target_player_id
                    break

                for player_id, slots in truth_slots_by_player.items():
                    truth_slots_by_player[player_id] = [
                        replace(slot, is_newly_drawn=False)
                        for slot in slots
                    ]
                if winner is not None:
                    break
                acting_player_id = target_player_id

            guess_actions = [
                action
                for action in actions
                if getattr(action, "action_type", None) == "guess"
            ]
            for public_state, truth_snapshot, action in zip(
                guess_public_states,
                guess_truth_snapshots,
                guess_actions,
            ):
                catalog = _candidate_catalog_from_state(
                    model,
                    public_state,
                    truth_snapshot,
                    action,
                )
                if catalog:
                    samples.append(catalog)
    return samples


def _normalize_catalog_payload(
    payload: Any,
) -> Optional[List[Dict[str, Any]]]:
    catalog = payload
    if isinstance(payload, dict):
        catalog = payload.get("candidate_catalog", payload.get("catalog"))
    if not isinstance(catalog, list):
        return None
    normalized_catalog: List[Dict[str, Any]] = []
    for candidate in catalog:
        if not isinstance(candidate, dict):
            return None
        normalized_catalog.append(dict(candidate))
    if len(normalized_catalog) <= 1:
        return None
    if not any(bool(candidate.get("observed")) for candidate in normalized_catalog):
        return None
    return normalized_catalog


def _load_external_trace_samples(
    trace_dir: Path = EXTERNAL_TRACE_DIR,
) -> Tuple[List[List[Dict[str, Any]]], int]:
    if not trace_dir.exists():
        return [], 0

    samples: List[List[Dict[str, Any]]] = []
    file_count = 0
    for trace_path in sorted(trace_dir.rglob("*.jsonl")):
        file_count += 1
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            catalog = _normalize_catalog_payload(payload)
            if catalog is not None:
                samples.append(catalog)
    for trace_path in sorted(trace_dir.rglob("*.json")):
        file_count += 1
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            for item in payload:
                catalog = _normalize_catalog_payload(item)
                if catalog is not None:
                    samples.append(catalog)
        else:
            catalog = _normalize_catalog_payload(payload)
            if catalog is not None:
                samples.append(catalog)
    return samples, file_count


def _fit_softmax_weights(
    samples: Sequence[List[Dict[str, Any]]],
    *,
    feature_key: str,
    initial_weights: Dict[str, float],
    epoch_count: int = 60,
    learning_rate: float = 0.35,
    regularization: float = 0.02,
) -> Dict[str, float]:
    weights = dict(initial_weights)
    feature_names = list(initial_weights.keys())
    for epoch_index in range(epoch_count):
        gradient = {feature_name: 0.0 for feature_name in feature_names}
        sample_count = 0.0
        for catalog in samples:
            positive_candidates = [
                candidate for candidate in catalog if candidate.get("observed")
            ]
            if len(catalog) <= 1 or not positive_candidates:
                continue
            centered_rows: List[Dict[str, float]] = []
            logits: List[float] = []
            for candidate in catalog:
                feature_row = candidate.get(feature_key, {})
                centered_row = {
                    feature_name: float(feature_row.get(feature_name, 0.0)) - 0.5
                    for feature_name in feature_names
                }
                centered_rows.append(centered_row)
                logits.append(
                    sum(
                        weights[feature_name] * centered_row[feature_name]
                        for feature_name in feature_names
                    )
                )
            max_logit = max(logits)
            probabilities = [math.exp(logit - max_logit) for logit in logits]
            total_probability = sum(probabilities)
            if total_probability <= 0.0:
                continue
            probabilities = [
                probability / total_probability for probability in probabilities
            ]
            for candidate_index, candidate in enumerate(catalog):
                target = 1.0 if candidate.get("observed") else 0.0
                error = probabilities[candidate_index] - target
                for feature_name in feature_names:
                    gradient[feature_name] += (
                        error * centered_rows[candidate_index][feature_name]
                    )
            sample_count += 1.0
        if sample_count <= 0.0:
            break
        step_size = learning_rate / math.sqrt(epoch_index + 1.0)
        for feature_name in feature_names:
            average_gradient = gradient[feature_name] / sample_count
            average_gradient += regularization * (
                weights[feature_name] - initial_weights[feature_name]
            )
            weights[feature_name] -= step_size * average_gradient
    return weights


def main() -> None:
    model = BehavioralLikelihoodModel(load_trained_weights=False)
    regression_samples = _collect_regression_samples(model)
    self_play_seed_schedule = tuple(range(11, 107, 8))
    self_play_games_per_seed = 5
    self_play_samples = _collect_self_play_samples(
        model,
        seeds=self_play_seed_schedule,
        games_per_seed=self_play_games_per_seed,
    )
    external_trace_samples, external_trace_file_count = _load_external_trace_samples()
    all_samples = regression_samples + self_play_samples + external_trace_samples

    generative_weights = _fit_softmax_weights(
        all_samples,
        feature_key="parametric_features",
        initial_weights=model.PARAMETRIC_GENERATIVE_FEATURE_WEIGHTS,
    )
    posterior_weights = _fit_softmax_weights(
        all_samples,
        feature_key="posterior_parametric_features",
        initial_weights=model.PARAMETRIC_POSTERIOR_FEATURE_WEIGHTS,
    )

    output_path = BACKEND_ROOT / "app" / "core" / "parametric_weights.json"
    training_sources = [
        "regression_cases",
        "large_self_play_trajectory_corpus",
    ]
    if external_trace_samples:
        training_sources.append("external_trace_catalogs")
    payload = {
        "metadata": {
            "training_source": " + ".join(training_sources),
            "sample_count": len(all_samples),
            "regression_sample_count": len(regression_samples),
            "self_play_sample_count": len(self_play_samples),
            "self_play_seed_count": len(self_play_seed_schedule),
            "self_play_games_per_seed": self_play_games_per_seed,
            "external_trace_sample_count": len(external_trace_samples),
            "external_trace_file_count": external_trace_file_count,
        },
        "generative": {
            "bias": model.PARAMETRIC_GENERATIVE_BIAS,
            "weights": generative_weights,
        },
        "posterior": {
            "bias": model.PARAMETRIC_POSTERIOR_BIAS,
            "weights": posterior_weights,
        },
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
