import { useState, useCallback, useEffect } from 'react';
import './index.css';
import { Hand } from './components/Hand';
import { AICenter } from './components/AICenter';
import { ActionHistory } from './components/ActionHistory';
import { TileEditorModal } from './components/TileEditorModal';
import { ActionLogger } from './components/ActionLogger';
import { useGameState } from './hooks/useGameState';
import { useAISync } from './hooks/useAISync';
import type { TileState, GameAction } from './types';

function App() {
  // 使用新的游戏状态管理 hook
  const {
    state,
    addTile,
    removeTile,
    updateTile,
    reorderTiles,
    moveTile,
    addAction,
    resetGame,
    setPlayerCount,
    setSessionId,
    undo,
    canUndo,
    totalTiles,
    opponentNames,
  } = useGameState();

  // 使用 AI 同步 hook
  const { aiAnalysis, isLoading: isAILoading, error, clearAnalysis, syncWithAI } = useAISync(state);

  // Clear AI analysis when game state changes (so we don't look at stale manual output)
  // Or handle differently if we want it preserved.
  // Actually, let's let the user keep it until they trigger again.

  // UI Modal State
  const [editingTileId, setEditingTileId] = useState<{playerId: string, tileId: string} | null>(null);

  useEffect(() => {
    if (aiAnalysis?.sessionId && aiAnalysis.sessionId !== state.sessionId) {
      setSessionId(aiAnalysis.sessionId);
    }
  }, [aiAnalysis?.sessionId, setSessionId, state.sessionId]);

  const displayOpponents = state.opponents.map(opponent => ({
    ...opponent,
    tiles: opponent.tiles.map((tile, index) => ({
      ...tile,
      probabilities: aiAnalysis?.posteriorProbabilities[`${opponent.id}_${index}`] ?? tile.probabilities,
    })),
  }));

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
  const handleAddTile = useCallback((playerId: string, color: 'black' | 'white') => {
    addTile(playerId, color);
  }, [addTile]);

  const handleRemoveTile = useCallback((playerId: string) => {
    removeTile(playerId);
  }, [removeTile]);

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
    window.setTimeout(() => {
      syncWithAI();
    }, 0);
  }, [addAction, syncWithAI]);

  // Handle undo
  const handleUndo = useCallback(() => {
    undo();
  }, [undo]);

  // Handle clear all
  const handleClearAll = useCallback(() => {
    if (window.confirm('确定要清空所有数据吗？')) {
      clearAnalysis();
      resetGame();
    }
  }, [clearAnalysis, resetGame]);

  return (
    <div className="min-h-screen bg-[#0B1120] text-gray-200 font-sans selection:bg-indigo-500/30 relative overflow-hidden">
      {/* Decorative ambient blurred blobs */}
      <div className="fixed top-[-20%] left-[-10%] w-[50vw] h-[50vw] bg-indigo-600/10 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="fixed bottom-[-20%] right-[-10%] w-[40vw] h-[40vw] bg-blue-600/10 rounded-full blur-[150px] pointer-events-none"></div>

      <div className="max-w-[1500px] mx-auto p-4 sm:p-6 relative z-10 w-full h-full flex flex-col">
        {/* Sleek Glassmorphism Header */}
        <header className="bg-gray-900/60 backdrop-blur-xl border border-white/10 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.5)] p-4 sm:p-6 mb-6 flex flex-col md:flex-row items-center justify-between gap-6 transition-all">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-[0_0_20px_rgba(99,102,241,0.4)] border border-white/20">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 tracking-tight">
                DaVinci AI
              </h1>
              <p className="text-xs sm:text-sm text-indigo-300/70 font-medium tracking-widest mt-1 uppercase">POMDP Tactical Decision Engine</p>
            </div>
          </div>

          {/* Control Strip */}
          <div className="flex flex-wrap items-center gap-2 sm:gap-3 bg-black/30 p-2 sm:p-2.5 rounded-xl border border-white/5 shadow-inner">
            {/* Player Count */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/60 rounded-lg border border-gray-700/50 group hover:border-gray-600 transition-colors">
              <svg className="w-4 h-4 text-indigo-400 group-hover:text-indigo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" /></svg>
              <select
                value={state.playerCount}
                onChange={(e) => handlePlayerCountChange(Number(e.target.value))}
                className="bg-transparent border-none text-sm font-semibold text-gray-200 focus:ring-0 cursor-pointer outline-none w-20 appearance-none"
              >
                <option className="bg-gray-800 text-white" value={2}>2 玩家</option>
                <option className="bg-gray-800 text-white" value={3}>3 玩家</option>
                <option className="bg-gray-800 text-white" value={4}>4 玩家</option>
              </select>
            </div>

            <div className="h-6 w-[1px] bg-gray-700/50 hidden sm:block"></div>

            <button
              onClick={handleUndo}
              disabled={!canUndo}
              className="flex items-center gap-1.5 px-3 sm:px-4 py-1.5 sm:py-2 bg-gray-800/80 hover:bg-gray-700 text-gray-300 hover:text-white rounded-lg border border-gray-700/50 transition-all shadow-sm disabled:opacity-40 disabled:cursor-not-allowed text-xs sm:text-sm font-medium"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" /></svg>
              撤销操作
            </button>

            <button
              onClick={handleClearAll}
              className="flex items-center gap-1.5 px-3 sm:px-4 py-1.5 sm:py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 rounded-lg border border-red-500/20 transition-all shadow-sm text-xs sm:text-sm font-medium"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
              重置
            </button>

            <div className="h-6 w-[1px] bg-gray-700/50 hidden sm:block mx-1"></div>

            <div className="flex items-center gap-4 px-2 sm:px-4">
              <div className="flex flex-col">
                <span className="text-[9px] sm:text-[10px] text-gray-500 uppercase tracking-widest font-bold">总牌数</span>
                <span className="text-xs sm:text-sm font-bold text-gray-200 leading-tight">{totalTiles}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[9px] sm:text-[10px] text-gray-500 uppercase tracking-widest font-bold">AI 状态</span>
                <div className="flex items-center gap-1.5 leading-tight">
                  <span className={`w-2 h-2 rounded-full ${isAILoading ? 'bg-indigo-400 animate-pulse shadow-[0_0_8px_rgba(129,140,248,0.8)]' : 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]'}`}></span>
                  <span className="text-xs sm:text-sm font-bold text-gray-200">{isAILoading ? '计算中...' : '已就绪'}</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 flex-1 min-h-0">
          {/* Left/Center: Playing Area */}
          <div className="xl:col-span-8 flex flex-col gap-6">
            {/* Opponents Area */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-6">
              {displayOpponents.map((opponent) => (
                <div key={opponent.id} className="transform transition-transform duration-300 hover:-translate-y-1">
                  <Hand
                    playerName={opponent.name}
                    isMe={false}
                    tiles={opponent.tiles}
                    aiTargetTileIndex={aiAnalysis?.attackTarget?.playerId === opponent.id ? aiAnalysis.attackTarget.tileIndex : undefined}
                    onTileClick={(tileId) => setEditingTileId({ playerId: opponent.id, tileId })}
                    onAddTile={(color) => handleAddTile(opponent.id, color)}
                    onRemoveTile={() => handleRemoveTile(opponent.id)}
                    onReorderTile={(oldIndex, newIndex) => handleReorderTile(opponent.id, oldIndex, newIndex)}
                  />
                </div>
              ))}
            </div>

            {/* Spacer */}
            <div className="flex-1 min-h-[40px] flex items-center justify-center pointer-events-none">
              <div className="w-full flex items-center gap-4 opacity-40">
                <div className="h-[1px] flex-1 bg-gradient-to-r from-transparent via-indigo-500/50 to-indigo-500"></div>
                <span className="text-[10px] uppercase tracking-[0.3em] text-indigo-300 font-bold border border-indigo-500/30 px-4 py-1.5 rounded-full bg-indigo-500/5 shadow-[0_0_15px_rgba(99,102,241,0.2)]">VS</span>
                <div className="h-[1px] flex-1 bg-gradient-to-l from-transparent via-indigo-500/50 to-indigo-500"></div>
              </div>
            </div>

            {/* My Hand */}
            <div className="transform transition-transform duration-300 hover:-translate-y-1 scale-[1.01] origin-bottom mb-4">
              <Hand
                playerName="我的手牌"
                isMe={true}
                tiles={state.myHand}
                onTileClick={(tileId) => setEditingTileId({ playerId: 'me', tileId })}
                onAddTile={(color) => handleAddTile('me', color)}
                onRemoveTile={() => handleRemoveTile('me')}
                onReorderTile={(oldIndex, newIndex) => handleReorderTile('me', oldIndex, newIndex)}
              />
            </div>
          </div>

          {/* Right Sidebar: AI Dashboard */}
          <div className="xl:col-span-4 flex flex-col gap-6 h-full min-h-[800px] xl:min-h-0 relative z-10 space-y-0">
            {/* AI Top Center */}
            <div className="flex-none h-auto sm:h-[400px] rounded-2xl overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.5)] border border-indigo-500/20 bg-gray-900/60 backdrop-blur-xl relative group">
              <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none"></div>
              <AICenter
                aiAnalysis={aiAnalysis}
                isAILoading={isAILoading}
                error={error}
                opponentNames={opponentNames || {}}
                onTrigger={syncWithAI}
              />
            </div>

            {/* Action Log Form */}
            <div className="flex-none bg-gray-900/60 backdrop-blur-xl rounded-2xl border border-white/5 shadow-xl overflow-hidden mt-6">
               <div className="p-3 bg-gray-800/40 border-b border-gray-700/50">
                  <h2 className="text-[13px] font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                    <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>
                    记录新动作
                  </h2>
               </div>
               <div className="p-4">
                 <ActionLogger 
                     players={[
                       { id: 'me', name: '我', tiles: state.myHand },
                       ...displayOpponents.map(o => ({ id: o.id, name: o.name, tiles: o.tiles }))
                     ]}
                     onLogAction={handleAddAction}
                   />
               </div>
            </div>

            {/* Event Timeline / History */}
            <div className="flex-1 min-h-[250px] bg-gray-900/60 backdrop-blur-xl rounded-2xl border border-white/5 shadow-xl overflow-hidden flex flex-col mt-6">
               <div className="p-3 bg-gray-800/40 border-b border-gray-700/50 flex flex-shrink-0 items-center justify-between">
                  <h2 className="text-[13px] font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                    <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    动作历史
                  </h2>
               </div>
               <div className="flex-1 overflow-y-auto custom-scrollbar p-2 relative">
                 <div className="absolute top-0 left-6 bottom-0 w-[1px] bg-gray-800 pointer-events-none"></div>
                 <ActionHistory logs={state.actionHistory} />
               </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tile Editor Modal */}
      {editingTileId && getEditingTile() && (
        <TileEditorModal
          tile={getEditingTile()!}
          onClose={() => setEditingTileId(null)}
          onSave={(updates) => handleUpdateTile(
            editingTileId.playerId,
            editingTileId.tileId,
            updates
          )}
          onMove={(direction) => {
            const tiles = editingTileId.playerId === 'me' ? state.myHand : state.opponents.find(o => o.id === editingTileId.playerId)?.tiles || [];
            const idx = tiles.findIndex(t => t.id === editingTileId.tileId);
            if (idx !== -1) {
              moveTile(editingTileId.playerId, idx, direction === -1 ? 'left' : 'right');
            }
          }}
        />
      )}
    </div>
  );
}

export default App;
