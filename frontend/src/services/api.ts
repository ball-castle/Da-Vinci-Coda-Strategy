import type { GameAction, GuessValue, PlayerState, ProbabilityPreview, TileState } from '../types';

type FrontendColor = 'black' | 'white';
type BackendColor = 'B' | 'W';
type BackendValue = number | '-';

export interface GameStatePayload {
  sessionId?: string;
  playerCount: number;
  opponents: PlayerState[];
  myHand: TileState[];
  actionHistory: GameAction[];
}

export interface DrawRecommendation {
  color: FrontendColor;
  reasoning: string;
}

export interface AttackTarget {
  playerId: string;
  tileIndex: number;
  expectedNumber: GuessValue;
  confidence: number;
}

export interface AIAnalysisResponse {
  sessionId?: string;
  recommendedAction: 'DRAW' | 'GUESS' | 'STOP';
  reasoning: string;
  drawRecommendation?: DrawRecommendation;
  attackTarget?: AttackTarget;
  posteriorProbabilities: Record<string, ProbabilityPreview[]>;
  error?: string;
  isDeadEnd?: boolean;
}

interface BackendMove {
  target_player_id: string;
  target_slot_index: number;
  guess_card?: [BackendColor, BackendValue];
  win_probability?: number;
  recommendation_reason?: string;
}

interface BackendProbabilityCandidate {
  card: [BackendColor, BackendValue];
  probability: number;
}

interface BackendProbabilityPosition {
  target_slot_index: number;
  candidates: BackendProbabilityCandidate[];
}

interface BackendProbabilityPlayer {
  player_id: string;
  positions: BackendProbabilityPosition[];
}

interface BackendDecisionSummary {
  stop_reason?: string;
}

interface BackendTurnResponse {
  session_id?: string;
  recommended_action?: string;
  recommended_draw_color?: BackendColor | null;
  best_move?: BackendMove | null;
  decision_summary?: BackendDecisionSummary;
  full_probability_matrix?: BackendProbabilityPlayer[];
  search_space_size?: number;
}

function toBackendColor(color: FrontendColor): BackendColor {
  return color === 'black' ? 'B' : 'W';
}

function toFrontendColor(color: BackendColor): FrontendColor {
  return color === 'B' ? 'black' : 'white';
}

function toBackendGuessValue(value?: number): BackendValue | null {
  if (value === undefined || Number.isNaN(value)) {
    return null;
  }

  return value === -1 ? '-' : value;
}

function resolveTargetPlayerId(payload: GameStatePayload): string {
  const latestGuessTarget = [...payload.actionHistory]
    .reverse()
    .find(action => action.type === 'GUESS' && action.targetId && action.targetId !== 'me')
    ?.targetId;

  return latestGuessTarget ?? payload.opponents[0]?.id ?? 'opponent';
}

function resolveNewlyDrawnTileId(myHand: TileState[], actionHistory: GameAction[]): string | undefined {
  const latestSelfAction = [...actionHistory]
    .reverse()
    .find(action => action.actorId === 'me');

  if (latestSelfAction?.type !== 'DRAW' || !latestSelfAction.color) {
    return undefined;
  }

  const candidates = myHand.filter(tile => tile.color === latestSelfAction.color);
  return candidates.at(-1)?.id;
}

function buildStructuredActions(actionHistory: GameAction[]) {
  return actionHistory.flatMap((action, index) => {
    if (action.type !== 'GUESS' || !action.targetId) {
      return [];
    }

    const parsedTargetIndex = action.targetTileId !== undefined
      ? Number.parseInt(action.targetTileId, 10)
      : Number.NaN;
    const nextAction = actionHistory[index + 1];

    return [{
      guesser_id: action.actorId,
      target_player_id: action.targetId,
      target_slot_index: Number.isNaN(parsedTargetIndex) ? null : parsedTargetIndex,
      guessed_color: action.targetColor ? toBackendColor(action.targetColor) : null,
      guessed_value: toBackendGuessValue(action.guessNumber),
      result: action.isHit ?? false,
      continued_turn: Boolean(
        action.isHit
        && nextAction?.actorId === action.actorId
        && nextAction?.type === 'GUESS'
      ),
      action_type: 'guess',
    }];
  });
}

function buildProbabilityMap(data: BackendTurnResponse): Record<string, ProbabilityPreview[]> {
  const probabilityMap: Record<string, ProbabilityPreview[]> = {};

  for (const player of data.full_probability_matrix ?? []) {
    for (const position of player.positions) {
      probabilityMap[`${player.player_id}_${position.target_slot_index}`] = position.candidates.map(candidate => ({
        number: candidate.card[1],
        prob: Math.round(candidate.probability * 100),
      }));
    }
  }

  return probabilityMap;
}

function mapRecommendedAction(data: BackendTurnResponse): 'DRAW' | 'GUESS' | 'STOP' {
  const rawAction = data.recommended_action ?? '';

  if (rawAction === 'guess') {
    return 'GUESS';
  }

  if (rawAction === 'stop') {
    return 'STOP';
  }

  if (rawAction.startsWith('draw_')) {
    return 'DRAW';
  }

  return data.best_move ? 'GUESS' : 'STOP';
}

export async function fetchAIAnalysis(
  payload: GameStatePayload,
  signal?: AbortSignal,
): Promise<AIAnalysisResponse> {
  try {
    const targetPlayerId = resolveTargetPlayerId(payload);
    const newlyDrawnTileId = resolveNewlyDrawnTileId(payload.myHand, payload.actionHistory);

    const requestBody = {
      session_id: payload.sessionId,
      state: {
        self_player_id: 'me',
        target_player_id: targetPlayerId,
        players: [
          {
            player_id: 'me',
            slots: payload.myHand.map((tile, index) => ({
              slot_index: index,
              color: toBackendColor(tile.color),
              value: tile.isJoker ? '-' : (tile.number ?? null),
              is_revealed: tile.isKnown,
              is_newly_drawn: tile.id === newlyDrawnTileId,
            })),
          },
          ...payload.opponents.map(opponent => ({
            player_id: opponent.id,
            slots: opponent.tiles.map((tile, index) => ({
              slot_index: index,
              color: toBackendColor(tile.color),
              value: tile.isJoker ? '-' : (tile.number ?? null),
              is_revealed: tile.isKnown,
              is_newly_drawn: false,
            })),
          })),
        ],
        actions: buildStructuredActions(payload.actionHistory),
      },
    };

    const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
    const response = await fetch(`${baseUrl}/api/turn`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
      signal,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as BackendTurnResponse;
    const recommendedAction = mapRecommendedAction(data);
    const probabilityMap = buildProbabilityMap(data);
    const drawColor = data.recommended_draw_color ? toFrontendColor(data.recommended_draw_color) : undefined;
    const bestMove = data.best_move ?? undefined;

    return {
      sessionId: data.session_id,
      recommendedAction,
      reasoning: bestMove?.recommendation_reason
        ?? data.decision_summary?.stop_reason
        ?? (recommendedAction === 'DRAW'
          ? '当前更适合先补牌，再观察下一步进攻窗口。'
          : 'AI 已完成当前局面的策略评估。'),
      drawRecommendation: recommendedAction === 'DRAW' && drawColor
        ? {
          color: drawColor,
          reasoning: `当前建议优先补${drawColor === 'black' ? '黑' : '白'}牌，以保持后续进攻与防守弹性。`,
        }
        : undefined,
      attackTarget: recommendedAction === 'GUESS' && bestMove
        ? {
          playerId: bestMove.target_player_id,
          tileIndex: bestMove.target_slot_index,
          expectedNumber: bestMove.guess_card?.[1] ?? 0,
          confidence: bestMove.win_probability ?? 0,
        }
        : undefined,
      posteriorProbabilities: probabilityMap,
      isDeadEnd: data.search_space_size === 0,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error('[API] Failed to fetch AI analysis', error);

    return {
      recommendedAction: 'STOP',
      reasoning: `Error connecting to AI Backend: ${errorMessage}`,
      posteriorProbabilities: {},
      error: errorMessage,
    };
  }
}
