import { DndContext, closestCenter, KeyboardSensor, PointerSensor, TouchSensor, useSensor, useSensors } from '@dnd-kit/core';
import { SortableContext, horizontalListSortingStrategy } from '@dnd-kit/sortable';
import type { TileState } from '../types';
import { SortableTile } from './SortableTile';

interface HandProps {
  playerName: string;
  isMe: boolean;
  tiles: TileState[];
  aiTargetTileIndex?: number;
  onTileClick?: (id: string) => void;
  onAddTile?: (color: 'black' | 'white') => void;
  onRemoveTile?: () => void;
  onReorderTile: (oldIndex: number, newIndex: number) => void;
}

export function Hand({ playerName, isMe, tiles, aiTargetTileIndex, onTileClick, onAddTile, onRemoveTile, onReorderTile }: HandProps) {
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5, // 5px tolerance to distinguish between click and drag
      },
    }),
    useSensor(TouchSensor, {
      activationConstraint: {
        delay: 250, // 250ms press to drag on touch devices to prevent scroll capturing
        tolerance: 5,
      },
    }),
    useSensor(KeyboardSensor)
  );

  const handleDragEnd = (event: any) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const oldIndex = tiles.findIndex((t) => t.id === active.id);
      const newIndex = tiles.findIndex((t) => t.id === over.id);
      if (oldIndex !== -1 && newIndex !== -1) {
        onReorderTile(oldIndex, newIndex);
      }
    }
  };

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

      <DndContext 
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <div className="flex flex-wrap gap-3 min-h-[80px]">
          <SortableContext 
            items={tiles.map(t => t.id)}
            strategy={horizontalListSortingStrategy}
          >
            {tiles.map((tile, index) => (
              <SortableTile 
                key={tile.id} 
                tile={tile} 
                isTarget={aiTargetTileIndex === index} 
                onTileClick={onTileClick}
              />
            ))}
          </SortableContext>
        </div>
      </DndContext>
    </div>
  );
}
