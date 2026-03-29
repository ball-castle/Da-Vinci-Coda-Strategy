import { useState } from 'react';
import type { TileState } from '../types';

interface TileEditorModalProps {
  tile: TileState;
  onClose: () => void;
  onSave: (updatedTile: TileState) => void;
}

export function TileEditorModal({ tile, onClose, onSave }: TileEditorModalProps) {
  const [num, setNum] = useState<string>(tile.number !== undefined ? tile.number.toString() : '');
  const [isJoker, setIsJoker] = useState<boolean>(tile.isJoker || false);

  if (!tile) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6 w-full max-w-sm border border-gray-600">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          {tile.color === 'black' ? (
            <span className="w-4 h-4 bg-black border border-gray-500 rounded-sm inline-block"></span>
          ) : (
            <span className="w-4 h-4 bg-white border border-gray-300 rounded-sm inline-block"></span>
          )}
          编辑此牌状态
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm font-semibold">设置数字 (0-11)</label>
            <input 
              type="number" 
              min="0" max="11"
              value={num}
              onChange={(e) => setNum(e.target.value)}
              disabled={isJoker}
              className="bg-gray-900 border border-gray-600 text-white rounded px-3 py-1.5 w-24 text-center disabled:opacity-50"
              placeholder="未知?"
            />
          </div>

          <div className="flex items-center justify-between pt-2">
            <label className="text-gray-300 text-sm font-semibold flex items-center gap-2 cursor-pointer">
               百搭牌 (-)
            </label>
            <input 
              type="checkbox" 
              checked={isJoker}
              onChange={(e) => setIsJoker(e.target.checked)}
              className="w-5 h-5 bg-gray-900 border-gray-600 rounded cursor-pointer"
            />
          </div>
        </div>

        <div className="mt-8 flex justify-end gap-3">
          <button 
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 font-bold text-white rounded transition"
          >
            取消
          </button>
          <button 
            onClick={() => {
              const updatedTile = {
                ...tile,
                number: num !== '' && !isJoker ? parseInt(num, 10) : undefined,
                isJoker: isJoker,
                isKnown: isJoker || num !== '' // 只要填了数字或勾了百搭就是明牌
              };
              onSave(updatedTile);
            }}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-500 font-bold text-white rounded shadow transition"
          >
            保存并重排
          </button>
        </div>
      </div>
    </div>
  );
}

