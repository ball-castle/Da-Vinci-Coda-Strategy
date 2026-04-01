/**
 * 瓦片工具函数
 */
import { TileState } from '../types';

/**
 * 对瓦片进行排序
 */
export function sortTiles(tiles: TileState[]): TileState[] {
  const knownTiles = tiles.filter(t => t.number !== null && !t.isJoker);
  const unknownAndJokers = tiles.filter(t => t.number === null || t.isJoker);

  knownTiles.sort((a, b) => {
    if (a.number !== b.number) {
      return (a.number ?? 0) - (b.number ?? 0);
    }
    if (a.color === 'black' && b.color === 'white') return -1;
    if (a.color === 'white' && b.color === 'black') return 1;
    return 0;
  });

  const result: TileState[] = [];
  let knownIndex = 0;
  let unknownIndex = 0;

  tiles.forEach(tile => {
    if (tile.number === null || tile.isJoker) {
      if (unknownIndex < unknownAndJokers.length) {
        result.push(unknownAndJokers[unknownIndex++]);
      }
    } else {
      if (knownIndex < knownTiles.length) {
        result.push(knownTiles[knownIndex++]);
      }
    }
  });

  return result;
}

export function generateTileId(): string {
  return `tile-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function reorderArray<T>(array: T[], oldIndex: number, newIndex: number): T[] {
  const newArr = [...array];
  const [removed] = newArr.splice(oldIndex, 1);
  newArr.splice(newIndex, 0, removed);
  return newArr;
}

export function moveInArray<T>(array: T[], index: number, direction: 'left' | 'right'): T[] {
  if (direction === 'left' && index === 0) return array;
  if (direction === 'right' && index === array.length - 1) return array;

  const newIndex = direction === 'left' ? index - 1 : index + 1;
  return reorderArray(array, index, newIndex);
}
