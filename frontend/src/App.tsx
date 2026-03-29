import React, { useState } from 'react';
import './index.css';
import { Hand } from './components/Hand';
import { AICenter } from './components/AICenter';
import { ActionHistory } from './components/ActionHistory';

function App() {
  const [opponents] = useState([
    {
      id: 'p-a',
      name: 'Player A',
      tiles: [
        { id: 'a1', color: 'black', isKnown: true, number: 0 },
        { id: 'a2', color: 'white', isKnown: false, probabilities: [{ number: 5, prob: 60 }, { number: 6, prob: 40 }] },
        { id: 'a3', color: 'black', isKnown: false, probabilities: [{ number: 8, prob: 90 }, { number: 9, prob: 10 }] },
      ]
    },
    {
      id: 'p-b',
      name: 'Player B',
      tiles: [
        { id: 'b1', color: 'white', isKnown: false, probabilities: [{ number: 1, prob: 50 }, { number: 2, prob: 50 }] },
        { id: 'b2', color: 'white', isKnown: false, probabilities: [{ number: 4, prob: 75 }, { number: 5, prob: 25 }] },
        { id: 'b3', color: 'black', isKnown: true, number: 11 },
      ]
    }
  ]);

  const [myHand] = useState([
    { id: 'm1', color: 'black', isKnown: true, number: 2 },
    { id: 'm2', color: 'white', isKnown: true, number: 7 },
    { id: 'm3', color: 'black', isKnown: true, number: 10 },
  ]);

  return (
    <div className="min-h-screen p-6 flex gap-6 max-w-7xl mx-auto h-screen items-stretch">
      
      {/* 左侧：桌面状态沙盘 (70%) */}
      <div className="flex-[7] flex flex-col gap-6 h-full">
        <header className="mb-2 shrink-0">
          <h1 className="text-3xl font-black text-white tracking-widest uppercase items-center flex gap-3">
            DA VINCI CODA <span className="text-blue-500 bg-blue-500/10 px-2 py-0.5 rounded text-2xl border border-blue-500/30">AI</span>
          </h1>
          <p className="text-sm text-gray-400 mt-1">部分可观察马尔可夫决策辅助系统</p>
        </header>

        {/* 敌方手牌区域 */}
        <div className="flex flex-col gap-4 flex-1 overflow-y-auto pr-2 custom-scrollbar">
          {opponents.map(opp => (
            <Hand 
              key={opp.id}
              playerName={opp.name} 
              isMe={false} 
              // @ts-ignore (简化处理)
              tiles={opp.tiles} 
            />
          ))}
        </div>

        {/* 桌面中央界线 */}
        <div className="w-full border-t-2 border-dashed border-gray-700 my-2 relative shrink-0">
          <span className="absolute left-1/2 -top-3 -translate-x-1/2 bg-gray-900 px-4 text-gray-500 font-bold text-sm tracking-widest">
            YOUR HAND
          </span>
        </div>

        {/* 我方手牌区域 */}
        <div className="shrink-0 mb-4">
          <Hand 
            playerName="My Player" 
            isMe={true} 
            // @ts-ignore
            tiles={myHand}
            onAddTile={(color) => console.log('Add tile', color)}
            onTileClick={(id) => console.log('Clicked', id)}
          />
        </div>
      </div>

      {/* 右侧：控制台与事件区 (30%) */}
      <div className="flex-[3] flex flex-col gap-6 h-full pt-[64px]">
        {/* AI 决策面板 (占上部) */}
        <div className="flex-[3] min-h-0">
          <AICenter />
        </div>
        
        {/* 事件时间线 (占下部) */}
        <div className="flex-[2] min-h-0 mb-4">
          <ActionHistory />
        </div>
      </div>

    </div>
  );
}

export default App;
