import type { PlayerState, TileState, GameAction } from '../types';

export interface GameStatePayload {
  playerCount: number;
  opponents: PlayerState[];
  myHand: TileState[];
  actionHistory: GameAction[];
}

export interface AIAnalysisResponse {
  recommendedAction: 'DRAW' | 'GUESS' | 'STOP';
  reasoning: string;
  attackTarget?: {
    playerId: string;
    tileIndex: number;
    expectedNumber: number;
    confidence: number;
  };
  posteriorProbabilities: Record<string, { number: number; prob: number }[]>; 
}

export async function fetchAIAnalysis(payload: GameStatePayload): Promise<AIAnalysisResponse> {
  try {
    const targetPlayerId = payload.opponents[0]?.id || 'opponent';

    const requestBody = {
      state: {
        self_player_id: 'me',
        target_player_id: targetPlayerId,
        players: [
          {
            player_id: 'me',
            slots: payload.myHand.map((t, i) => ({
              slot_index: i,
              color: t.color === 'black' ? 'B' : 'W',
              value: t.isJoker ? '-' : (t.number !== undefined ? t.number : -1),
              is_revealed: t.isKnown, 
              is_newly_drawn: false
            }))
          },
          ...payload.opponents.map(op => ({
            player_id: op.id,
            slots: op.tiles.map((t, i) => ({
              slot_index: i,
              color: t.color === 'black' ? 'B' : 'W',
              value: t.isJoker ? '-' : (t.number !== undefined ? t.number : -1),
              is_revealed: t.isKnown,
              is_newly_drawn: false
            }))
          }))
        ],
        actions: payload.actionHistory.map(a => ({
          guesser_id: a.type === 'GUESS' ? (a.targetId === 'me' ? targetPlayerId : 'me') : 'me', // simplistic mapping
          target_player_id: a.targetId || targetPlayerId,
          target_slot_index: a.targetTileId ? parseInt(a.targetTileId, 10) : 0,
          guessed_color: null,
          guessed_value: a.guessNumber ?? null,
          result: a.isHit ?? false,
          continued_turn: false,
          action_type: a.type.toLowerCase()
        }))
      }
    };

    console.log('[API] Sending request to POMDP Backend...', requestBody);

    const response = await fetch('/api/turn', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[API] Received response:', data);

    const bestMove = data.best_move;
    
    // Parse backend response to frontend format
    const isStop = !bestMove || data.recommended_action === 'stop' || !bestMove.target_player_id;
    
    return {
      recommendedAction: isStop ? 'STOP' : 'GUESS',
      reasoning: data.behavior_debug?.hypothesis_source || 'MCTS Full-game Search computed',
      attackTarget: !isStop ? {
        playerId: bestMove.target_player_id,
        tileIndex: bestMove.target_slot_index,
        expectedNumber: bestMove.guess_card?.[1] || 0,
        confidence: bestMove.win_probability || 0
      } : undefined,
      posteriorProbabilities: {} // Optionally map `data.probability_matrix` here if needed
    };
  } catch (error) {
    console.error('[API] Failed to fetch AI analysis', error);
    // Fallback in case of error
    return {
      recommendedAction: 'STOP',
      reasoning: 'Error connecting to AI Backend: ' + (error as Error).message,
      posteriorProbabilities: {}
    };
  }
}

