/**
 * 游戏状态管理Hook
 */
import { useReducer, useCallback, useMemo, useRef, useState } from 'react';
import { gameReducer, initialGameState } from '../store/gameReducer';
import type { GameState } from '../store/gameTypes';
import type { TileState, GameAction } from '../types';

const MAX_HISTORY_SIZE = 50; // 限制历史记录大小防止内存泄漏

export function useGameState() {
  const [state, dispatch] = useReducer(gameReducer, initialGameState);
  const historyStack = useRef<GameState[]>([]);
  const [historyDepth, setHistoryDepth] = useState(0);

  // 保存当前状态到历史栈
  const saveToHistory = useCallback(() => {
    historyStack.current.push(state);
    // 限制历史栈大小
    if (historyStack.current.length > MAX_HISTORY_SIZE) {
      historyStack.current = historyStack.current.slice(-MAX_HISTORY_SIZE);
    }
    setHistoryDepth(historyStack.current.length);
  }, [state]);

  // 撤销操作
  const undo = useCallback(() => {
    if (historyStack.current.length === 0) return false;

    const previousState = historyStack.current.pop();
    setHistoryDepth(historyStack.current.length);
    if (previousState) {
      dispatch({ type: 'RESTORE_STATE', payload: previousState });
      return true;
    }
    return false;
  }, []);

  // 玩家数量设置
  const setPlayerCount = useCallback((count: number) => {
    saveToHistory();
    dispatch({ type: 'SET_PLAYER_COUNT', payload: count });
  }, [saveToHistory]);

  // 添加瓦片
  const addTile = useCallback((playerId: string, color: 'black' | 'white') => {
    saveToHistory();
    dispatch({ type: 'ADD_TILE', payload: { playerId, color } });
  }, [saveToHistory]);

  // 移除瓦片
  const removeTile = useCallback((playerId: string) => {
    saveToHistory();
    dispatch({ type: 'REMOVE_TILE', payload: { playerId } });
  }, [saveToHistory]);

  // 重排瓦片
  const reorderTiles = useCallback((playerId: string, oldIndex: number, newIndex: number) => {
    saveToHistory();
    dispatch({ type: 'REORDER_TILES', payload: { playerId, oldIndex, newIndex } });
  }, [saveToHistory]);

  // 移动瓦片
  const moveTile = useCallback((playerId: string, tileIndex: number, direction: 'left' | 'right') => {
    saveToHistory();
    dispatch({ type: 'MOVE_TILE', payload: { playerId, tileIndex, direction } });
  }, [saveToHistory]);

  // 更新瓦片属性
  const updateTile = useCallback((playerId: string, tileId: string, updates: Partial<TileState>) => {
    saveToHistory();
    dispatch({ type: 'UPDATE_TILE', payload: { playerId, tileId, updates } });
  }, [saveToHistory]);

  // 添加游戏动作
  const addAction = useCallback((action: GameAction) => {
    saveToHistory();
    dispatch({ type: 'ADD_ACTION', payload: action });
  }, [saveToHistory]);

  // 清空动作历史
  const clearActions = useCallback(() => {
    saveToHistory();
    dispatch({ type: 'CLEAR_ACTIONS' });
  }, [saveToHistory]);

  // 重置整个对局
  const resetGame = useCallback(() => {
    saveToHistory();
    dispatch({ type: 'RESET_GAME' });
  }, [saveToHistory]);

  // 设置会话ID
  const setSessionId = useCallback((sessionId: string) => {
    dispatch({ type: 'SET_SESSION_ID', payload: sessionId });
  }, []);

  // 计算派生状态
  const derivedState = useMemo(() => ({
    totalTiles: state.myHand.length + state.opponents.reduce((sum, opp) => sum + opp.tiles.length, 0),
    canUndo: historyDepth > 0,
    opponentNames: state.opponents.reduce((acc, opp) => {
      acc[opp.id] = opp.name;
      return acc;
    }, {} as Record<string, string>),
  }), [historyDepth, state.myHand.length, state.opponents]);

  return {
    // State
    state,
    ...derivedState,

    // Actions
    setPlayerCount,
    addTile,
    removeTile,
    reorderTiles,
    moveTile,
    updateTile,
    addAction,
    clearActions,
    resetGame,
    setSessionId,
    undo,
  };
}
