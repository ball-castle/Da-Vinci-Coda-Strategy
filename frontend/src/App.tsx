import { useState, useEffect } from 'react';
import './index.css';
import { Hand } from './components/Hand';
import { AICenter } from './components/AICenter';
import { ActionHistory } from './components/ActionHistory';
import { TileEditorModal } from './components/TileEditorModal';
import { ActionLogger } from './components/ActionLogger';
import { fetchAIAnalysis } from './services/api';
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
  const [playerCount, setPlayerCount] = useState<number>(3); // 2, 3, or 4 (including You)

  const [opponents, setOpponents] = useState<PlayerState[]>(INITIAL_OPPONENTS_DATA.slice(0, 2));

  const [myHand, setMyHand] = useState<TileState[]>([]);

  // Action History State
  const [actionHistory, setActionHistory] = useState<GameAction[]>([]);

  // History Stack for Undo
  const [historyStack, setHistoryStack] = useState<{opponents: PlayerState[], myHand: TileState[], actionHistory: GameAction[]}[]>([]);

  // AI 建议状态
  const [isAILoading, setIsAILoading] = useState<boolean>(false);
  const [aiAnalysis, setAiAnalysis] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);

  // UI Modal State
  const [editingTileId, setEditingTileId] = useState<{playerId: string, tileId: string} | null>(null);

  // Capture current state for Undo
  const saveStateToHistory = () => {
    setHistoryStack(prev => [...prev, { opponents, myHand, actionHistory }]);
  };

  const handleUndo = () => {
    if (historyStack.length === 0) return;
    const lastState = historyStack[historyStack.length - 1];
    setOpponents(lastState.opponents);
    setMyHand(lastState.myHand);
    setActionHistory(lastState.actionHistory);
    setHistoryStack(prev => prev.slice(0, -1));
  };

  // Sorting Helper Function (黑左白右，从小到大)
  // 只对已知且非百搭的牌进行原位排序，保留未知牌和百搭牌的当前相对物理槽位
  const sortTiles = (tiles: TileState[]) => {
    const result = [...tiles];
    const knownPositions: number[] = [];
    const knownTiles: TileState[] = [];
    
    result.forEach((t, i) => {
      // 提取所有可以确定绝对大小的牌
      if (t.isKnown && !t.isJoker && t.number !== undefined) {
        knownPositions.push(i);
        knownTiles.push(t);
      }
    });

    // 对提取出的牌进行标准规则排序
    knownTiles.sort((a, b) => {
      if (a.number !== b.number) return a.number! - b.number!;
      if (a.color === 'black' && b.color === 'white') return -1;
      if (a.color === 'white' && b.color === 'black') return 1;
      return 0;
    });

    // 填回原位，这样就能完美保留 `?` 和 `-` 的用户手动排序插槽
    knownPositions.forEach((pos, index) => {
      result[pos] = knownTiles[index];
    });

    return result;
  };

  // Handle changing total players
  const handlePlayerCountChange = (count: number) => {
    saveStateToHistory();
    setPlayerCount(count);
    const requiredOpponents = count - 1;
    setOpponents(INITIAL_OPPONENTS_DATA.slice(0, requiredOpponents));
  };

  // Add a tile to a specific player
  const addTileToPlayer = (playerId: string, color: 'black' | 'white') => {
    saveStateToHistory();
    const newTile: TileState = {
      id: `${playerId}-${Date.now()}`,
      color,
      isKnown: false, // Default to unknown when just drawn
    };

    if (playerId === 'me') {
      setMyHand([...myHand, { ...newTile, isKnown: true }]); 
    } else {
      setOpponents(prev => prev.map(opp => 
        opp.id === playerId ? { ...opp, tiles: [...opp.tiles, newTile] } : opp
      ));
    }
  };

  // Remove the last tile from a specific player
  const removeTileFromPlayer = (playerId: string) => {
    saveStateToHistory();
    if (playerId === 'me') {
      setMyHand(prev => prev.slice(0, -1));
    } else {
      setOpponents(prev => prev.map(opp => 
        opp.id === playerId ? { ...opp, tiles: opp.tiles.slice(0, -1) } : opp
      ));
    }
  };

// Reorder tile (for drag and drop)
    const reorderTile = (playerId: string, oldIndex: number, newIndex: number) => {
      if (oldIndex === newIndex) return;
      saveStateToHistory();

      const reorderInArray = (arr: TileState[]) => {
        const newArr = [...arr];
        const [movedItem] = newArr.splice(oldIndex, 1);
        newArr.splice(newIndex, 0, movedItem);
        return sortTiles(newArr);
      };

      if (playerId === 'me') {
        setMyHand(reorderInArray);
      } else {
        setOpponents(prev => prev.map(opp =>
          opp.id === playerId ? { ...opp, tiles: reorderInArray(opp.tiles) } : opp
      ));
    }
  };

  // Find Tile to Edit
  const getTileToEdit = () => {
    if (!editingTileId) return null;
    if (editingTileId.playerId === 'me') {
      return myHand.find(t => t.id === editingTileId.tileId) || null;
    }
    const opp = opponents.find(o => o.id === editingTileId.playerId);
    return opp?.tiles.find(t => t.id === editingTileId.tileId) || null;
  };

  // 监听动作历史，自动调用 AI 分析
  useEffect(() => {
    // 增加防抖 (Debounce) 和 AbortController 避免并发轰炸
    const abortController = new AbortController();
    
    const syncAI = async () => {
      setIsAILoading(true);
      try {
        const result = await fetchAIAnalysis({ sessionId, playerCount, opponents, myHand, actionHistory }, abortController.signal);
        
        if (result.isDeadEnd || result.error) {
          window.alert(result.isDeadEnd ? "矛盾的录入！推演树崩溃，请点击撤销 (Undo)" : "连接服务器失败! 请检查网络");
          setIsAILoading(false);
          return;
        }

        setAiAnalysis(result);
        if (result.sessionId) {
          setSessionId(result.sessionId);
        }

        if (result.posteriorProbabilities) {
          setOpponents(prev => prev.map(opp => {
            let Changed = false;
            const newTiles = opp.tiles.map((t, idx) => {
              const key = `${opp.id}_${idx}`;
              const probs = result.posteriorProbabilities[key];
              if (probs) {
                Changed = true;
                return { ...t, probabilities: probs };
              }
              return t;
            });
            return Changed ? { ...opp, tiles: newTiles } : opp;
          }));
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error("AI Sync failed", err);
          window.alert("连接服务器失败! 请检查网络");
        }
      } finally {
        if (!abortController.signal.aborted) {
          setIsAILoading(false);
        }
      }
    };

    const timer = setTimeout(() => {
      syncAI();
    }, 500); // 500ms debounce

    return () => {
      clearTimeout(timer);
      abortController.abort();
    };
  }, [actionHistory, opponents.length, myHand.length]); // Re-fetch whenever history or hand amounts change

  // Handle move tile manually
  const handleMoveTile = (direction: -1 | 1) => {
    if (!editingTileId) return;
    saveStateToHistory();

    const moveInArray = (arr: TileState[]) => {
      const idx = arr.findIndex(t => t.id === editingTileId.tileId);
      if (idx < 0) return arr;
      const newIdx = idx + direction;
      if (newIdx < 0 || newIdx >= arr.length) return arr;
      const newArr = [...arr];
      [newArr[idx], newArr[newIdx]] = [newArr[newIdx], newArr[idx]];
      return sortTiles(newArr);
    };

    if (editingTileId.playerId === 'me') {
      setMyHand(prev => moveInArray(prev));
    } else {
      setOpponents(prev => prev.map(opp => {
        if (opp.id === editingTileId.playerId) {
          return { ...opp, tiles: moveInArray(opp.tiles) };
        }
        return opp;
      }));
    }
  };

  // Handle Save Edited Tile
  const handleSaveTile = (updatedTile: TileState) => {
    if (!editingTileId) return;
    saveStateToHistory();

    if (editingTileId.playerId === 'me') {
      setMyHand(prev => sortTiles(prev.map(t => t.id === updatedTile.id ? updatedTile : t)));
    } else {
      setOpponents(prev => prev.map(opp => {
        if (opp.id === editingTileId.playerId) {
          return {
            ...opp,
            tiles: sortTiles(opp.tiles.map(t => t.id === updatedTile.id ? updatedTile : t))
          };
        }
        return opp;
      }));
    }
    setEditingTileId(null);
  };

  const handleResetGame = () => {
    if (window.confirm("确定要重置当前对局吗？所有的牌将被清空。")) {
      saveStateToHistory();
      setMyHand([]);
      setOpponents(prev => prev.map(opp => ({...opp, tiles: []})));
      setActionHistory([]);
    }
  };

  const handleLogAction = (action: GameAction) => {
    saveStateToHistory();
    // 1. Log to history
    setActionHistory(prev => [action, ...prev]);

    // 2. Auto-sync state
    if (action.type === 'DRAW' && action.color) {
      // Auto add tile
      const newTile: TileState = { id: `${action.actorId}-${Date.now()}`, color: action.color, isKnown: action.actorId === 'me' };
      if (action.actorId === 'me') {
        setMyHand(prev => [...prev, { ...newTile, isKnown: true }]);
      } else {
        setOpponents(prev => prev.map(opp => opp.id === action.actorId ? { ...opp, tiles: [...opp.tiles, newTile] } : opp));
      }
    }
    else if (action.type === 'GUESS' && action.isHit) {
      // Auto reveal tile if guessed correctly
      if (action.targetId && action.targetTileId && action.guessNumber !== undefined) {
        const targetIndex = parseInt(action.targetTileId, 10);
        if (action.targetId === 'me') {
          // Typically my hand is known, do nothing
        } else {
          setOpponents(prev => prev.map(opp => {
            if (opp.id === action.targetId) {
              const newTiles = [...opp.tiles];
              if (newTiles[targetIndex]) {
                newTiles[targetIndex] = { ...newTiles[targetIndex], isKnown: true, number: action.guessNumber };
              }
              return { ...opp, tiles: sortTiles(newTiles) };
            }
            return opp;
          }));
        }
      }
    }
  };

  return (
    <div className="min-h-screen p-2 sm:p-6 flex flex-col lg:flex-row gap-6 max-w-[1600px] mx-auto h-full lg:h-screen lg:items-stretch relative">
      {/* 模态框渲染 */}
      {editingTileId && getTileToEdit() && (
        <TileEditorModal 
          tile={getTileToEdit()!} 
          onClose={() => setEditingTileId(null)} 
          onSave={handleSaveTile}
            onMove={handleMoveTile}
        />
      )}

      {/* 左侧：桌面状态沙盘 (60%) */}
      <div className="flex-[6] flex flex-col gap-6 h-full">
        <header className="mb-2 shrink-0 flex flex-col xl:flex-row justify-between items-start gap-4">
          <div>
            <h1 className="text-3xl font-black text-white tracking-widest uppercase items-center flex gap-3">
              DA VINCI CODA <span className="text-blue-500 bg-blue-500/10 px-2 py-0.5 rounded text-2xl border border-blue-500/30">AI</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">部分可观察马尔可夫决策辅助系统</p>
          </div>
          
          {/* 游戏设置区 */}
          <div className="flex gap-4">
            <div className="bg-gray-800 p-2 rounded-lg border border-gray-700 shadow-sm flex items-center gap-3">
              <span className="text-sm font-semibold text-gray-300">对局总人数:</span>
              <div className="flex gap-1 bg-gray-900 rounded p-1">
                {[2, 3, 4].map(num => (
                  <button
                    key={num}
                    onClick={() => handlePlayerCountChange(num)}
                    className={`px-3 py-1 rounded text-sm font-bold transition-colors ${
                      playerCount === num 
                        ? 'bg-blue-600 text-white shadow' 
                        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                    }`}
                  >
                    {num}人
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={handleUndo}
              disabled={historyStack.length === 0}
              className={`px-4 py-2 rounded-lg font-bold transition shadow-sm border ${
                historyStack.length === 0 
                  ? 'bg-gray-800 text-gray-600 border-gray-700 cursor-not-allowed' 
                  : 'bg-yellow-900/40 text-yellow-400 border-yellow-800/50 hover:bg-yellow-900/60 hover:text-yellow-300'
              }`}
              title="撤销上一步操作"
            >
              ↩ 撤销
            </button>
            <button
              onClick={handleResetGame}
              className="bg-red-900/40 text-red-400 border border-red-800/50 hover:bg-red-900/60 hover:text-red-300 px-4 py-2 rounded-lg font-bold transition shadow-sm"
              title="清空所有玩家状态"
            >
              重置对局
            </button>
          </div>
        </header>

        {/* 敌方手牌区域 */}
        <div className="flex flex-col gap-4 flex-1 overflow-y-auto pr-2 custom-scrollbar">
          {opponents.map(opp => (
            <Hand 
              key={opp.id}
              playerName={opp.name} 
              isMe={false} 
              tiles={opp.tiles}
              aiTargetTileIndex={
                aiAnalysis?.attackTarget?.playerId === opp.id 
                  ? aiAnalysis.attackTarget.tileIndex 
                  : undefined
              }
              onAddTile={(color) => addTileToPlayer(opp.id, color)}
              onRemoveTile={() => removeTileFromPlayer(opp.id)}
              onTileClick={(tileId) => setEditingTileId({ playerId: opp.id, tileId })}
              onReorderTile={(oldIndex, newIndex) => reorderTile(opp.id, oldIndex, newIndex)}
            />
          ))}
        </div>

        {/* 桌面中央界线 */}
        <div className="w-full border-t-2 border-dashed border-gray-700 my-2 relative shrink-0">
          <span className="absolute left-1/2 -top-3 -translate-x-1/2 bg-gray-900 px-4 text-gray-500 font-bold text-sm tracking-widest">
            YOUR HAND
          </span>
        </div>

        {/* 我方手牌区域 */}
        <div className="shrink-0 mb-4">
          <Hand 
            playerName="My Player" 
            isMe={true} 
            tiles={myHand}
            onAddTile={(color) => addTileToPlayer('me', color)}
            onRemoveTile={() => removeTileFromPlayer('me')}
            onTileClick={(tileId) => setEditingTileId({ playerId: 'me', tileId })}
            onReorderTile={(oldIndex, newIndex) => reorderTile('me', oldIndex, newIndex)}
          />
        </div>
      </div>

      {/* 右侧：控制台与事件区 (40%) */}
      <div className="flex-[4] flex flex-col gap-6 h-full lg:pt-[64px]">
        {/* AI 决策面板 (占上部) */}
        <div className="flex-[4] min-h-0 flex flex-col gap-4">
          <AICenter 
            aiAnalysis={aiAnalysis} 
            isAILoading={isAILoading}
            opponentNames={opponents.reduce((acc, opp) => ({...acc, [opp.id]: opp.name}), {} as Record<string, string>)}
          />
          <ActionLogger 
            players={[...opponents, { id: 'me', name: 'You', tiles: myHand }]} 
            onLogAction={handleLogAction} 
          />
        </div>
        
        {/* 事件时间线 (占下部) */}
        <div className="flex-[2] min-h-0 mb-4">
          <ActionHistory logs={actionHistory} />
        </div>
      </div>

    </div>
  );
}

export default App;
