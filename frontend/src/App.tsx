import { useState, useMemo, useCallback } from 'react';
import './index.css';
import { Hand } from './components/Hand';
import { AICenter } from './components/AICenter';
import { ActionHistory } from './components/ActionHistory';
import { TileEditorModal } from './components/TileEditorModal';
import { ActionLogger } from './components/ActionLogger';
import { useGameState } from './hooks/useGameState';
import { useAISync } from './hooks/useAISync';
import type { TileState, PlayerState, GameAction } from './types';

const INITIAL_OPPONENTS_DATA = [
  {
    id: 'p-a',
    name: 'Player A',
    tiles: []
  },
  {
    id: 'p-b',
    name: 'Player B',
    tiles: []
  },
  {
    id: 'p-c',
    name: 'Player C',
    tiles: []
  }
] as PlayerState[];

function App() {
  // 使用新的游戏状态管理 hook
  const {
    state,
    addMyTile,
    addOpponentTile,
    removeMyTile,
    removeOpponentTile,
    updateTile,
    reorderTiles,
    addAction,
    clearHistory,
    setPlayerCount,
    sortPlayerTiles,
    undo,
    canUndo,
    opponentNames,
    totalTiles
  } = useGameState({
    initialPlayerCount: 3,
    initialOpponents: INITIAL_OPPONENTS_DATA.slice(0, 2)
  });

  // 使用 AI 同步 hook
  const { aiAnalysis, isLoading: isAILoading, error: aiError } = useAISync(state, {
    enabled: true,
    debounceMs: 500
  });

  // UI Modal State
  const [editingTileId, setEditingTileId] = useState<{playerId: string, tileId: string} | null>(null);

  // Memoize computed values
  const canAddOpponentTile = useMemo(() => 
    state.opponents.every(opp => opp.tiles.length < 12),
    [state.opponents]
  );

  // Get currently editing tile
  const getEditingTile = useCallback((): TileState | null => {
    if (!editingTileId) return null;
    if (editingTileId.playerId === 'me') {
      return state.myHand.find(t => t.id === editingTileId.tileId) || null;
    }
    const opp = state.opponents.find(o => o.id === editingTileId.playerId);
    return opp?.tiles.find(t => t.id === editingTileId.tileId) || null;
  }, [editingTileId, state.myHand, state.opponents]);

  // Handle player count change
  const handlePlayerCountChange = useCallback((count: number) => {
    setPlayerCount(count);
  }, [setPlayerCount]);

  // Handle Add Tile Actions
  const handleAddMyTile = useCallback(() => {
    addMyTile();
  }, [addMyTile]);

  const handleAddOpponentTile = useCallback(() => {
    addOpponentTile();
  }, [addOpponentTile]);

  const handleRemoveMyTile = useCallback((tileId: string) => {
    removeMyTile(tileId);
  }, [removeMyTile]);

  const handleRemoveOpponentTile = useCallback((playerId: string, tileId: string) => {
    removeOpponentTile(playerId, tileId);
  }, [removeOpponentTile]);

  // Handle Sort
  const handleSortMyHand = useCallback(() => {
    sortPlayerTiles('me');
  }, [sortPlayerTiles]);

  const handleSortOpponentHand = useCallback((playerId: string) => {
    sortPlayerTiles(playerId);
  }, [sortPlayerTiles]);

  // Handle tile reorder (drag and drop)
  const handleReorderTile = useCallback((playerId: string, oldIndex: number, newIndex: number) => {
    if (oldIndex !== newIndex) {
      reorderTiles(playerId, oldIndex, newIndex);
    }
  }, [reorderTiles]);

  // Handle tile update (from modal)
  const handleUpdateTile = useCallback((playerId: string, tileId: string, updates: Partial<TileState>) => {
    updateTile(playerId, tileId, updates);
    setEditingTileId(null);
  }, [updateTile]);

  // Handle action logging
  const handleAddAction = useCallback((action: GameAction) => {
    addAction(action);
  }, [addAction]);

  // Handle undo
  const handleUndo = useCallback(() => {
    undo();
  }, [undo]);

  // Handle clear all
  const handleClearAll = useCallback(() => {
    if (window.confirm('确定要清空所有数据吗？')) {
      clearHistory();
    }
  }, [clearHistory]);

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">DaVinci Code Assistant</h1>
          
          {/* Controls */}
          <div className="flex flex-wrap gap-4 items-center">
            {/* Player Count Selector */}
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-gray-700">玩家数量:</label>
              <select
                value={state.playerCount}
                onChange={(e) => handlePlayerCountChange(Number(e.target.value))}
                className="border border-gray-300 rounded px-3 py-1"
              >
                <option value={2}>2人</option>
                <option value={3}>3人</option>
                <option value={4}>4人</option>
              </select>
            </div>

            {/* Add Tile Buttons */}
            <button
              onClick={handleAddMyTile}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              添加我的牌
            </button>
            
            <button
              onClick={handleAddOpponentTile}
              disabled={!canAddOpponentTile}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              添加对手牌
            </button>

            {/* Undo Button */}
            <button
              onClick={handleUndo}
              disabled={!canUndo}
              className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              撤销 ({state.historyStack?.length || 0})
            </button>

            {/* Clear Button */}
            <button
              onClick={handleClearAll}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              清空
            </button>

            {/* Stats */}
            <div className="ml-auto text-sm text-gray-600">
              总牌数: {totalTiles} | AI状态: {isAILoading ? '计算中...' : '就绪'}
            </div>
          </div>
        </div>

        {/* Main Game Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Left: Opponents */}
          <div className="lg:col-span-2 space-y-4">
            {/* My Hand */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-xl font-semibold text-gray-800">我的手牌</h2>
                <button
                  onClick={handleSortMyHand}
                  className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                >
                  排序
                </button>
              </div>
              <Hand
                playerId="me"
                tiles={state.myHand}
                onTileClick={(tileId) => setEditingTileId({ playerId: 'me', tileId })}
                onRemoveTile={handleRemoveMyTile}
                onReorder={handleReorderTile}
              />
            </div>

            {/* Opponents */}
            {state.opponents.map((opponent) => (
              <div key={opponent.id} className="bg-white rounded-lg shadow-lg p-4">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-xl font-semibold text-gray-800">{opponent.name}</h2>
                  <button
                    onClick={() => handleSortOpponentHand(opponent.id)}
                    className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                  >
                    排序
                  </button>
                </div>
                <Hand
                  playerId={opponent.id}
                  tiles={opponent.tiles}
                  onTileClick={(tileId) => setEditingTileId({ playerId: opponent.id, tileId })}
                  onRemoveTile={(tileId) => handleRemoveOpponentTile(opponent.id, tileId)}
                  onReorder={handleReorderTile}
                />
              </div>
            ))}
          </div>

          {/* Right: AI Analysis & Actions */}
          <div className="space-y-4">
            {/* AI Analysis */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">AI分析</h2>
              <AICenter 
                aiAnalysis={aiAnalysis} 
                isLoading={isAILoading}
                error={aiError}
              />
            </div>

            {/* Action History */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">动作历史</h2>
              <ActionHistory actions={state.actionHistory} />
            </div>

            {/* Action Logger */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">记录动作</h2>
              <ActionLogger 
                players={[
                  { id: 'me', name: '我' },
                  ...state.opponents.map(o => ({ id: o.id, name: o.name }))
                ]}
                onAddAction={handleAddAction}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Tile Editor Modal */}
      {editingTileId && (
        <TileEditorModal
          tile={getEditingTile()}
          onClose={() => setEditingTileId(null)}
          onSave={(updates) => handleUpdateTile(
            editingTileId.playerId, 
            editingTileId.tileId, 
            updates
          )}
        />
      )}
    </div>
  );
}

export default App;
