import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import type { TileState } from '../types';

interface SortableTileProps {
  tile: TileState;
  isTarget: boolean;
  onTileClick?: (id: string) => void;
}

export function SortableTile({ tile, isTarget, onTileClick }: SortableTileProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: tile.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 100 : 1,
    opacity: isDragging ? 0.8 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={`
        relative flex flex-col items-center justify-center w-14 h-20 rounded-md shadow-md transition-shadow
        ${tile.color === 'black' ? 'bg-black text-white' : 'bg-gray-100 text-gray-900'}
        ${isTarget ? 'border-2 border-red-500 shadow-[0_0_12px_rgba(239,68,68,0.8)] scale-110' : tile.color === 'black' ? 'border border-gray-600' : 'border border-gray-300'}
      `}
      onClick={() => onTileClick && onTileClick(tile.id)}
    >
      {isTarget && (
        <div className="absolute -top-6 text-red-500 text-xs font-bold animate-bounce flex items-center gap-1 pointer-events-none">
          <span>🎯 Target</span>
        </div>
      )}

      {/* 点击区域 */}
      <div className="absolute inset-0 flex items-center justify-center cursor-pointer pointer-events-none">
        <span className="text-2xl font-bold">
          {tile.isKnown ? (tile.isJoker ? '-' : tile.number) : '?'}
        </span>
      </div>

      {/* AI 概率预览（针对暗牌） */}
      {!tile.isKnown && tile.probabilities && (
        <div className="absolute -bottom-6 w-24 text-center text-xs text-blue-400 font-mono tracking-tighter pointer-events-none mt-2">
          {tile.probabilities.slice(0, 2).map(p => `${p.number}:${p.prob}%`).join(' ')}
        </div>
      )}
    </div>
  );
}
