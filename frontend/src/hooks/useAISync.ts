/**
 * AI同步Hook
 * 负责与后端MCTS引擎的通信
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchAIAnalysis } from '../services/api';
import { GameState } from '../store/gameTypes';

interface UseAISyncOptions {
  enabled: boolean;
  debounceMs?: number;
}

export function useAISync(gameState: GameState, options: UseAISyncOptions) {
  const { enabled, debounceMs = 800 } = options;
  const [isLoading, setIsLoading] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  const syncWithAI = useCallback(async (signal: AbortSignal) => {
    try {
      setIsLoading(true);
      setError(null);

      // 构建请求负载
      const payload = {
        session_id: gameState.sessionId,
        state: {
          self_player_id: 'me',
          target_player_id: gameState.opponents[0]?.id || 'opp1',
          players: [
            {
              player_id: 'me',
              slots: gameState.myHand.map((tile, idx) => ({
                slot_index: idx,
                color: tile.color === 'black' ? 'B' : 'W',
                value: tile.isJoker ? '-' : tile.number,
                is_revealed: tile.number !== null || tile.isJoker,
                is_newly_drawn: false,
              })),
            },
            ...gameState.opponents.map(opp => ({
              player_id: opp.id,
              slots: opp.tiles.map((tile, idx) => ({
                slot_index: idx,
                color: tile.color === 'black' ? 'B' : 'W',
                value: tile.isJoker ? '-' : tile.number,
                is_revealed: tile.number !== null || tile.isJoker,
                is_newly_drawn: false,
              })),
            })),
          ],
          actions: gameState.actionHistory,
        },
      };

      const result = await fetchAIAnalysis(payload, signal);

      if (result.isDeadEnd || result.error) {
        setError(result.isDeadEnd ? '矛盾的录入！推演树崩溃' : '连接服务器失败');
        setAiAnalysis(null);
      } else {
        setAiAnalysis(result);
        setError(null);
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('AI Sync failed:', err);
        setError('连接服务器失败，请检查网络');
        setAiAnalysis(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, [gameState]);

  useEffect(() => {
    if (!enabled) {
      setAiAnalysis(null);
      setError(null);
      return;
    }

    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // 清除之前的防抖定时器
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // 设置新的防抖定时器
    debounceTimerRef.current = setTimeout(() => {
      abortControllerRef.current = new AbortController();
      syncWithAI(abortControllerRef.current.signal);
    }, debounceMs);

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [enabled, syncWithAI, debounceMs, gameState.actionHistory.length, gameState.myHand.length, gameState.opponents.length]);

  const refresh = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    syncWithAI(abortControllerRef.current.signal);
  }, [syncWithAI]);

  return {
    isLoading,
    aiAnalysis,
    error,
    refresh,
  };
}
