import React from 'react';

export function AICenter() {
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-lg flex flex-col h-full overflow-hidden">
      <div className="p-4 bg-gray-900 border-b border-gray-700">
        <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400 flex items-center">
          <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
          AI Decision Engine
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Module A: Draw Advisor */}
        <div className="space-y-2">
          <button className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow transition font-semibold">
            [请求抽牌建议]
          </button>
          <div className="p-3 bg-gray-700/50 rounded-lg border border-gray-600">
            <div className="text-sm text-gray-300 flex items-center gap-2">
              建议抽 <span className="text-white text-xs font-bold px-2 py-1 bg-black rounded border border-gray-600">黑牌</span>
            </div>
            <p className="text-xs text-gray-400 mt-2 italic">
              *防守推演：你手中的黑牌存在断层，抽黑牌可增加对手推断难度。*
            </p>
          </div>
        </div>

        <hr className="border-gray-700" />

        {/* Module B: Attack Advisor */}
        <div className="space-y-2">
          <button className="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow transition font-semibold">
            [请求猜牌建议]
          </button>
          <div className="p-4 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg border border-indigo-500/30 shadow-inner">
            <h3 className="text-xs font-bold text-indigo-300 mb-2 uppercase tracking-wider">🎯 最佳攻击目标</h3>
            <ul className="text-sm text-gray-200 space-y-1">
              <li>对象: <span className="font-semibold text-white">Player A</span></li>
              <li>槽位: <span className="font-semibold text-white">左数第3张 (暗黑)</span></li>
              <li className="mt-1">提议猜测: <span className="font-bold text-red-400 text-lg ml-1">黑 8</span></li>
            </ul>
            <div className="mt-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">预测置信度</span>
                <span className="text-green-400 font-bold">90%</span>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-1.5 overflow-hidden">
                <div className="bg-green-400 h-1.5 rounded-full" style={{ width: '90%' }}></div>
              </div>
            </div>
          </div>
        </div>

        <hr className="border-gray-700" />

        {/* Module C: Continuation Analyzer */}
        <div className="space-y-2">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">⚠️ 连击风险研判</h3>
          <div className="p-3 bg-red-900/20 rounded-lg border border-red-800/50 transition">
            <h4 className="text-base font-bold text-red-400 text-center mb-1">见好就收，终止回合</h4>
            <p className="text-xs text-gray-300 text-center">
              后续安全攻击确信度仅为 42%，猜错暴露新牌代价过高。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
