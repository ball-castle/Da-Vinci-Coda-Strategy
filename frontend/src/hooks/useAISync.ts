/**
 * AI同步Hook
 * 负责与后端MCTS引擎的通信
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchAIAnalysis, type AIAnalysisResponse } from '../services/api';
import type { GameState } from '../store/gameTypes';

export function useAISync(gameState: GameState) {
  const [isLoading, setIsLoading] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const latestGameStateRef = useRef(gameState);

  useEffect(() => {
    latestGameStateRef.current = gameState;
  }, [gameState]);

  const syncWithAI = useCallback(async (nextGameState?: GameState) => {
    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      setIsLoading(true);
      setError(null);

      const activeGameState = nextGameState ?? latestGameStateRef.current;
      const payload = {
        sessionId: activeGameState.sessionId,
        playerCount: activeGameState.playerCount,
        opponents: activeGameState.opponents,
        myHand: activeGameState.myHand,
        actionHistory: activeGameState.actionHistory,
      };

      const [result] = await Promise.all([
        fetchAIAnalysis(payload, abortControllerRef.current.signal),
        new Promise(resolve => setTimeout(resolve, 800)),
      ]);

      if (result.isDeadEnd || result.error) {
        setError(result.isDeadEnd ? '矛盾的录入，推演树无法展开' : '连接服务器失败或后端尚未准备好');
        setAiAnalysis(null);
      } else {
        setAiAnalysis(result);
        setError(null);
      }
    } catch (errorValue) {
      if (!(errorValue instanceof Error) || errorValue.name !== 'AbortError') {
        console.error('AI Sync failed:', errorValue);
        setError('连接服务器失败，请检查后端是否启动');
        setAiAnalysis(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setAiAnalysis(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    isLoading,
    aiAnalysis,
    error,
    clearAnalysis,
    syncWithAI,
  };
}
