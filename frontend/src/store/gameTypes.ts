/**
 * 游戏状态管理类型定义
 */
import { TileState, PlayerState, GameAction } from '../types';

export interface GameState {
  playerCount: number;
  opponents: PlayerState[];
  myHand: TileState[];
  actionHistory: GameAction[];
  sessionId?: string;
}

export type GameActionType =
  | { type: 'SET_PLAYER_COUNT'; payload: number }
  | { type: 'ADD_TILE'; payload: { playerId: string; color: 'black' | 'white' } }
  | { type: 'REMOVE_TILE'; payload: { playerId: string } }
  | { type: 'REORDER_TILES'; payload: { playerId: string; oldIndex: number; newIndex: number } }
  | { type: 'MOVE_TILE'; payload: { playerId: string; tileIndex: number; direction: 'left' | 'right' } }
  | { type: 'UPDATE_TILE'; payload: { playerId: string; tileId: string; updates: Partial<TileState> } }
  | { type: 'ADD_ACTION'; payload: GameAction }
  | { type: 'CLEAR_ACTIONS' }
  | { type: 'RESTORE_STATE'; payload: GameState }
  | { type: 'SET_SESSION_ID'; payload: string };
