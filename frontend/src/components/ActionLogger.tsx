import { useState } from 'react';
import type { ActionType, GameAction } from '../types';

interface ActionLoggerProps {
  players: { id: string; name: string }[];
  onLogAction: (action: GameAction) => void;
}

export function ActionLogger({ players, onLogAction }: ActionLoggerProps) {
  const [actionType, setActionType] = useState<ActionType>('DRAW');
  const [actorId, setActorId] = useState<string>(players[0]?.id || '');
  
  // Specific to Draw
  const [color, setColor] = useState<'black' | 'white'>('black');
  
  // Specific to Guess
  const [targetId, setTargetId] = useState<string>(players[1]?.id || '');
  const [tileIndex, setTileIndex] = useState<number>(0);
  const [guessNumber, setGuessNumber] = useState<number>(0);
  const [isHit, setIsHit] = useState<boolean>(false);

  const handleSubmit = () => {
    const actorName = players.find(p => p.id === actorId)?.name || 'Unknown';
    const targetName = players.find(p => p.id === targetId)?.name || 'Unknown';
    
    let humanReadable: React.ReactNode = '';
    
    if (actionType === 'DRAW') {
      humanReadable = (
        <p>
          <span className="font-semibold text-white">{actorName}</span> 抽了一张 
          <span className={`px-1.5 py-0.5 font-bold rounded text-xs ml-1 ${color === 'black' ? 'bg-black text-white border border-gray-600' : 'bg-white text-black border border-gray-300'}`}>
            {color === 'black' ? '黑牌' : '白牌'}
          </span>
        </p>
      );
    } else if (actionType === 'GUESS') {
      humanReadable = (
        <div>
          <p>
            <span className="font-semibold text-white">{actorName}</span> 猜 <span className="font-semibold text-white">{targetName}</span> 的第 {tileIndex + 1} 张为 <span className="font-bold text-yellow-400">{guessNumber}</span>
          </p>
          <p className={`text-xs mt-1.5 p-1.5 rounded border ${isHit ? 'bg-green-900/20 border-green-900/50 text-green-400' : 'bg-red-900/20 border-red-900/50 text-red-500'}`}>
            {isHit ? '🎯 猜测成功！' : '❌ 失败，猜测结束。'}
          </p>
        </div>
      );
    } else if (actionType === 'STOP') {
      humanReadable = (
        <p>
          <span className="font-semibold text-white">{actorName}</span> 选择了 <span className="text-gray-400 font-bold">终止猜测 (Pass)</span>
        </p>
      );
    }

    const action: GameAction = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      actorId,
      type: actionType,
      color: actionType === 'DRAW' ? color : undefined,
      targetId: actionType === 'GUESS' ? targetId : undefined,
      targetTileId: actionType === 'GUESS' ? `${tileIndex}` : undefined,
      guessNumber: actionType === 'GUESS' ? guessNumber : undefined,
      isHit: actionType === 'GUESS' ? isHit : undefined,
      humanReadable
    };

    onLogAction(action);
  };

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-lg p-4 flex flex-col gap-4">
      <div className="flex justify-between items-center mb-1">
        <h3 className="text-sm uppercase tracking-wider font-bold text-gray-400">回合控制器</h3>
        <div className="flex gap-1 bg-gray-900 p-1 rounded">
          {(['DRAW', 'GUESS', 'STOP'] as ActionType[]).map(type => (
            <button
              key={type}
              onClick={() => setActionType(type)}
              className={`px-3 py-1 text-xs font-bold rounded transition ${actionType === type ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
            >
              {type === 'DRAW' ? '摸牌' : type === 'GUESS' ? '猜测' : '过卡'}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2 text-sm text-gray-300">
        <span className="w-12 shrink-0">行动者:</span>
        <select 
          className="flex-1 bg-gray-900 border border-gray-700 rounded p-1.5 focus:ring focus:ring-blue-500 outline-none text-white"
          value={actorId} 
          onChange={(e) => setActorId(e.target.value)}
        >
          {players.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </div>

      {actionType === 'DRAW' && (
        <div className="flex items-center gap-2 text-sm text-gray-300">
          <span className="w-12 shrink-0">花色:</span>
          <div className="flex gap-2">
            <button 
              className={`px-4 py-1.5 rounded font-bold border transition ${color === 'black' ? 'bg-black text-white border-gray-500 shadow-lg scale-105' : 'bg-gray-800 text-gray-400 border-gray-700 hover:bg-gray-700'}`}
              onClick={() => setColor('black')}
            >
              黑牌
            </button>
            <button 
              className={`px-4 py-1.5 rounded font-bold border transition ${color === 'white' ? 'bg-white text-black border-gray-300 shadow-lg scale-105' : 'bg-gray-800 text-gray-400 border-gray-700 hover:bg-gray-700'}`}
              onClick={() => setColor('white')}
            >
              白牌
            </button>
          </div>
        </div>
      )}

      {actionType === 'GUESS' && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <span className="w-12 shrink-0">目标:</span>
            <select 
              className="flex-1 bg-gray-900 border border-gray-700 rounded p-1.5 focus:ring focus:ring-blue-500 outline-none text-white"
              value={targetId} 
              onChange={(e) => setTargetId(e.target.value)}
            >
              {players.filter(p => p.id !== actorId).map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <span className="w-12 shrink-0">位置:</span>
            <input 
              type="number" min={1} max={20} 
              className="w-16 bg-gray-900 border border-gray-700 rounded p-1.5 font-mono text-center outline-none focus:ring focus:ring-blue-500 text-white"
              value={tileIndex + 1} 
              onChange={(e) => setTileIndex(Math.max(0, parseInt(e.target.value) - 1))}
            />
            <span className="shrink-0 ml-2 text-gray-400">数字:</span>
            <input 
              type="number" min={0} max={11} 
              className="w-16 bg-gray-900 border border-gray-700 rounded p-1.5 font-mono text-center outline-none focus:ring focus:ring-blue-500 text-white"
              value={guessNumber} 
              onChange={(e) => setGuessNumber(parseInt(e.target.value))}
            />
            <span className="shrink-0 ml-1 text-xs text-gray-500">百搭用 -1</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-300 mt-1">
            <span className="w-12 shrink-0">结果:</span>
            <div className="flex gap-2">
              <button 
                className={`px-3 py-1 rounded font-bold border transition ${isHit ? 'bg-green-600 text-white border-green-500' : 'bg-gray-800 text-gray-500 border-gray-700 hover:bg-gray-700'}`}
                onClick={() => setIsHit(true)}
              >
                命中
              </button>
              <button 
                className={`px-3 py-1 rounded font-bold border transition ${!isHit ? 'bg-red-600 text-white border-red-500' : 'bg-gray-800 text-gray-500 border-gray-700 hover:bg-gray-700'}`}
                onClick={() => setIsHit(false)}
              >
                失败
              </button>
            </div>
          </div>
        </div>
      )}

      <button 
        onClick={handleSubmit}
        className="mt-2 w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 rounded-lg shadow transition"
      >
        录入系统并触发推演
      </button>
    </div>
  );
}