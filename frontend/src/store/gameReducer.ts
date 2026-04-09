/**
 * 游戏状态Reducer
 */
import type { GameState, GameActionType } from './gameTypes';
import { sortTiles, reorderArray, moveInArray, generateTileId } from '../utils/tileUtils';
import { updatePlayer, updatePlayerTiles } from '../utils/playerUtils';
import type { TileState } from '../types';

const INITIAL_OPPONENTS_DATA = [
  { id: 'opp1', name: '对手A', tiles: [] },
  { id: 'opp2', name: '对手B', tiles: [] },
  { id: 'opp3', name: '对手C', tiles: [] },
];

function createInitialOpponents(count = 3) {
  return INITIAL_OPPONENTS_DATA.slice(0, count - 1).map(opponent => ({
    ...opponent,
    tiles: [],
  }));
}

export function createInitialGameState(): GameState {
  return {
    playerCount: 3,
    opponents: createInitialOpponents(3),
    myHand: [],
    actionHistory: [],
    sessionId: undefined,
  };
}

export const initialGameState: GameState = createInitialGameState();

export function gameReducer(state: GameState, action: GameActionType): GameState {
  switch (action.type) {
    case 'SET_PLAYER_COUNT': {
      const count = action.payload;
      const newOpponents = createInitialOpponents(count);
      return { ...state, playerCount: count, opponents: newOpponents };
    }

    case 'ADD_TILE': {
      const { playerId, color } = action.payload;
      const newTile: TileState = {
        id: generateTileId(),
        color,
        isKnown: false,
        isJoker: false,
      };

      if (playerId === 'me') {
        return { ...state, myHand: sortTiles([...state.myHand, newTile]) };
      }

      return {
        ...state,
        opponents: updatePlayer(state.opponents, playerId, player =>
          updatePlayerTiles(player, tiles => sortTiles([...tiles, newTile]))
        ),
      };
    }

    case 'REMOVE_TILE': {
      const { playerId } = action.payload;

      if (playerId === 'me') {
        if (state.myHand.length === 0) return state;
        return { ...state, myHand: state.myHand.slice(0, -1) };
      }

      return {
        ...state,
        opponents: updatePlayer(state.opponents, playerId, player =>
          updatePlayerTiles(player, tiles => (tiles.length > 0 ? tiles.slice(0, -1) : tiles))
        ),
      };
    }

    case 'REORDER_TILES': {
      const { playerId, oldIndex, newIndex } = action.payload;

      if (playerId === 'me') {
        const reordered = reorderArray(state.myHand, oldIndex, newIndex);
        return { ...state, myHand: sortTiles(reordered) };
      }

      return {
        ...state,
        opponents: updatePlayer(state.opponents, playerId, player =>
          updatePlayerTiles(player, tiles => sortTiles(reorderArray(tiles, oldIndex, newIndex)))
        ),
      };
    }

    case 'MOVE_TILE': {
      const { playerId, tileIndex, direction } = action.payload;

      if (playerId === 'me') {
        const moved = moveInArray(state.myHand, tileIndex, direction);
        return { ...state, myHand: sortTiles(moved) };
      }

      return {
        ...state,
        opponents: updatePlayer(state.opponents, playerId, player =>
          updatePlayerTiles(player, tiles => sortTiles(moveInArray(tiles, tileIndex, direction)))
        ),
      };
    }

    case 'UPDATE_TILE': {
      const { playerId, tileId, updates } = action.payload;

      const updateTileById = (tiles: TileState[]) =>
        tiles.map(t => (t.id === tileId ? { ...t, ...updates } : t));

      if (playerId === 'me') {
        return { ...state, myHand: updateTileById(state.myHand) };
      }

      return {
        ...state,
        opponents: updatePlayer(state.opponents, playerId, player =>
          updatePlayerTiles(player, updateTileById)
        ),
      };
    }

    case 'ADD_ACTION':
      return { ...state, actionHistory: [...state.actionHistory, action.payload] };

    case 'CLEAR_ACTIONS':
      return { ...state, actionHistory: [] };

    case 'RESET_GAME':
      return createInitialGameState();

    case 'RESTORE_STATE':
      return action.payload;

    case 'SET_SESSION_ID':
      return { ...state, sessionId: action.payload };

    default:
      return state;
  }
}
