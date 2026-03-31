const fs = require('fs');
let c = fs.readFileSync('src/App.tsx', 'utf8');

const moveCode = \
  const handleMoveTile = (direction: -1 | 1) => {
    if (!editingTileId) return;
    saveStateToHistory();

    const moveInArray = (arr: TileState[]) => {
      const idx = arr.findIndex(t => t.id === editingTileId.tileId);
      if (idx < 0) return arr;
      const newIdx = idx + direction;
      if (newIdx < 0 || newIdx >= arr.length) return arr;
      const newArr = [...arr];
      [newArr[idx], newArr[newIdx]] = [newArr[newIdx], newArr[idx]];
      return sortTiles(newArr);
    };

    if (editingTileId.playerId === 'me') {
      setMyHand(prev => moveInArray(prev));
    } else {
      setOpponents(prev => prev.map(opp => {
        if (opp.id === editingTileId.playerId) {
          return { ...opp, tiles: moveInArray(opp.tiles) };
        }
        return opp;
      }));
    }
  };

  const handleSaveTile = \;

c = c.replace('  const handleSaveTile = ', moveCode);

c = c.replace('onSave={handleSaveTile}', 'onSave={handleSaveTile}\\n            onMove={handleMoveTile}');

fs.writeFileSync('src/App.tsx', c);

