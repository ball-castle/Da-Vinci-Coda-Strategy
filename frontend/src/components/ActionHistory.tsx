import React from 'react';

export function ActionHistory() {
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-lg h-full flex flex-col overflow-hidden">
      <div className="p-4 bg-gray-900 border-b border-gray-700">
        <h3 className="text-sm uppercase tracking-wider font-bold text-gray-400">Action History</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="relative pl-4 border-l-2 border-gray-600 space-y-5">
          
          <div className="relative">
            <span className="absolute -left-[23px] top-1 w-2.5 h-2.5 bg-blue-500 rounded-full border-2 border-gray-800"></span>
            <p className="text-[10px] text-gray-500 mb-0.5">Round 3</p>
            <p className="text-sm text-gray-300">
              <span className="font-semibold text-white">Player B</span> 抽了一张 <span className="px-1.5 py-0.5 bg-white text-black font-bold rounded text-xs ml-1">白牌</span>
            </p>
          </div>

          <div className="relative">
            <span className="absolute -left-[23px] top-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-gray-800"></span>
            <p className="text-sm text-gray-300">
              <span className="font-semibold text-white">Player B</span> 猜 <span className="font-semibold text-white">Player A</span> 的第2张为 <span className="font-bold text-red-400">白5</span>
            </p>
            <p className="text-xs text-red-500 mt-1.5 p-1.5 bg-red-900/10 rounded border border-red-900/30">
              ❌ 失败，Player B 展示了惩罚牌 <b>黑3</b>
            </p>
          </div>

          <div className="relative">
             <span className="absolute -left-[23px] top-1 w-2.5 h-2.5 bg-green-400 rounded-full border-2 border-gray-800 animate-pulse shadow-[0_0_8px_rgba(74,222,128,0.6)]"></span>
             <p className="text-xs text-green-400 mt-0.5 font-mono">
               系统完成贝叶斯更新坍缩...
             </p>
          </div>

        </div>
      </div>
    </div>
  );
}
