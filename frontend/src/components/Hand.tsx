import React from 'react';

type TileState = {
  id: string;
  color: 'black' | 'white';
  isKnown: boolean;
  number?: number;
  isJoker?: boolean;
  probabilities?: { number: number; prob: number }[]; // For unknown tiles
};

interface HandProps {
  playerName: string;
  isMe: boolean;
  tiles: TileState[];
  onTileClick?: (id: string) => void;
  onAddTile?: (color: 'black' | 'white') => void;
}

export function Hand({ playerName, isMe, tiles, onTileClick, onAddTile }: HandProps) {
  return (
    <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-200">
          {playerName} {isMe && <span className="text-sm text-green-400 font-normal ml-2">(You)</span>}
        </h2>
        {onAddTile && (
          <div className="flex gap-2">
            <button
              onClick={() => onAddTile('black')}
              className="px-3 py-1 bg-black text-white rounded hover:bg-gray-900 border border-gray-600 transition"
            >
              + 黑
            </button>
            <button
              onClick={() => onAddTile('white')}
              className="px-3 py-1 bg-white text-black rounded hover:bg-gray-200 border border-gray-300 transition shadow-sm"
            >
              + 白
            </button>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-3 min-h-[80px]">
        {tiles.map((tile) => (
          <div
            key={tile.id}
            onClick={() => onTileClick && onTileClick(tile.id)}
            className={`
              relative flex flex-col items-center justify-center w-14 h-20 rounded-md shadow-md cursor-pointer transition transform hover:-translate-y-1
              ${tile.color === 'black' ? 'bg-black text-white border border-gray-600' : 'bg-gray-100 text-gray-900 border border-gray-300'}
            `}
          >
            {/* 牌面数字显示 */}
            <span className="text-2xl font-bold">
              {tile.isKnown ? (tile.isJoker ? '-' : tile.number) : '?'}
            </span>

            {/* AI 概率预览（针对暗牌） */}
            {!tile.isKnown && tile.probabilities && (
              <div className="absolute -bottom-6 w-24 text-center text-xs text-blue-400 font-mono tracking-tighter">
                {tile.probabilities.slice(0, 2).map(p => `${p.number}:${p.prob}%`).join(' ')}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
