import type { TileState } from '../types';

interface HandProps {
  playerName: string;
  isMe: boolean;
  tiles: TileState[];
  aiTargetTileIndex?: number;
  onTileClick?: (id: string) => void;
  onAddTile?: (color: 'black' | 'white') => void;
  onRemoveTile?: () => void;
  onMoveTile?: (id: string, direction: 'left' | 'right') => void;
}

export function Hand({ playerName, isMe, tiles, aiTargetTileIndex, onTileClick, onAddTile, onRemoveTile, onMoveTile }: HandProps) {
  return (
    <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 shadow-lg relative">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-200">
          {playerName} {isMe && <span className="text-sm text-green-400 font-normal ml-2">(You)</span>}
        </h2>
        <div className="flex gap-2">
          {onAddTile && (
            <>
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
            </>
          )}
          {onRemoveTile && tiles.length > 0 && (
            <button
              onClick={onRemoveTile}
              className="px-3 py-1 bg-red-900/40 text-red-400 rounded hover:bg-red-900/60 border border-red-800/50 transition ml-2"
              title="移除最新摸的一张牌"
            >
              - 撤销
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-3 min-h-[80px]">
        {tiles.map((tile, index) => {
          const isTarget = aiTargetTileIndex === index;
          return (
          <div
            key={tile.id}
            className={`
              group relative flex flex-col items-center justify-center w-14 h-20 rounded-md shadow-md transition transform hover:-translate-y-1
              ${tile.color === 'black' ? 'bg-black text-white' : 'bg-gray-100 text-gray-900'}
              ${isTarget ? 'border-2 border-red-500 shadow-[0_0_12px_rgba(239,68,68,0.8)] scale-110' : tile.color === 'black' ? 'border border-gray-600' : 'border border-gray-300'}
            `}
          >
            {isTarget && (
              <div className="absolute -top-6 text-red-500 text-xs font-bold animate-bounce flex items-center gap-1">
                <span>🎯 Target</span>
              </div>
            )}
            
            {/* 点击区域占满整个牌的正中心，但边角留给移动按钮 */}
            <div 
              className="absolute inset-0 flex items-center justify-center cursor-pointer"
              onClick={() => onTileClick && onTileClick(tile.id)}
            >
              <span className="text-2xl font-bold">
                {tile.isKnown ? (tile.isJoker ? '-' : tile.number) : '?'}
              </span>
            </div>

            {/* AI 概率预览（针对暗牌） */}
            {!tile.isKnown && tile.probabilities && (
              <div className="absolute -bottom-6 w-24 text-center text-xs text-blue-400 font-mono tracking-tighter pointer-events-none">
                {tile.probabilities.slice(0, 2).map(p => `${p.number}:${p.prob}%`).join(' ')}
              </div>
            )}

            {/*左右移动控制 (Hover显示) */}
            {onMoveTile && (
              <div className="absolute -top-3 flex justify-between w-full px-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                <button 
                  onClick={(e) => { e.stopPropagation(); onMoveTile(tile.id, 'left'); }}
                  className="bg-gray-700/80 hover:bg-blue-600 text-white w-5 h-5 rounded-full flex items-center justify-center text-xs"
                >
                  ◀
                </button>
                <button 
                  onClick={(e) => { e.stopPropagation(); onMoveTile(tile.id, 'right'); }}
                  className="bg-gray-700/80 hover:bg-blue-600 text-white w-5 h-5 rounded-full flex items-center justify-center text-xs"
                >
                  ▶
                </button>
              </div>
            )}
          </div>
        )})}
      </div>
    </div>
  );
}
