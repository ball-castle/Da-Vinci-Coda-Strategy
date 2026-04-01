/**
 * 玩家工具函数
 */
import { PlayerState, TileState } from '../types';

export function findPlayer(players: PlayerState[], playerId: string): PlayerState | undefined {
  return players.find(p => p.id === playerId);
}

export function findTile(player: PlayerState | undefined, tileId: string): TileState | undefined {
  return player?.tiles.find(t => t.id === tileId);
}

export function updatePlayer(
  players: PlayerState[],
  playerId: string,
  updater: (player: PlayerState) => PlayerState
): PlayerState[] {
  return players.map(p => (p.id === playerId ? updater(p) : p));
}

export function updatePlayerTiles(
  player: PlayerState,
  updater: (tiles: TileState[]) => TileState[]
): PlayerState {
  return { ...player, tiles: updater(player.tiles) };
}
