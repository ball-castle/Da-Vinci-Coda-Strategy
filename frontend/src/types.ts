export type GuessValue = number | '-';

export type ProbabilityPreview = {
  number: GuessValue;
  prob: number;
};

export type TileState = {
  id: string;
  color: 'black' | 'white';
  isKnown: boolean;
  number?: number;
  isJoker?: boolean;
  probabilities?: ProbabilityPreview[];
};

export type PlayerState = {
  id: string;
  name: string;
  tiles: TileState[];
};

export type ActionType = 'DRAW' | 'GUESS' | 'STOP';

export type GameAction = {
  id: string;
  timestamp: number;
  actorId: string;
  type: ActionType;
  
  // Specific to DRAW
  color?: 'black' | 'white';
  
  // Specific to GUESS
  targetId?: string;       // Target player
  targetTileId?: string;   // the tile being guessed
  targetColor?: 'black' | 'white'; // The color of the tile being guessed
  guessNumber?: number;    // guessed number
  isHit?: boolean;         // guess result
  
  // Potentially for text reconstruction
  humanReadable: React.ReactNode; 
};
