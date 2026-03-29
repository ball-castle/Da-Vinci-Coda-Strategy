interface AICenterProps {
  aiAnalysis: any;
  isAILoading: boolean;
  opponentNames: Record<string, string>;
}

export function AICenter({ aiAnalysis, isAILoading, opponentNames }: AICenterProps) {
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

        {/* Module B: Attack Advisor & Continuation */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center justify-between">
            <span>⚔️ 攻击与防守研判</span>
            {isAILoading && <span className="text-blue-400 text-[10px] animate-pulse">Computing MCTS...</span>}
          </h3>

          {!aiAnalysis && !isAILoading && (
            <div className="p-4 bg-gray-800 rounded-lg text-sm text-gray-500 text-center italic border border-gray-700">
              录入任意动作以触发分析...
            </div>
          )}

          {aiAnalysis?.attackTarget && (
            <div className="p-4 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg border border-indigo-500/50 shadow-inner relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-indigo-500/10 blur-2xl"></div>
              <h4 className="text-xs font-bold text-indigo-300 mb-3 uppercase tracking-wider flex items-center gap-2">
                <span>🎯 最佳攻击目标</span>
                <span className="px-1.5 py-0.5 bg-indigo-500/20 text-indigo-300 rounded text-[10px]">Highest Value</span>
              </h4>
              <ul className="text-sm text-gray-200 space-y-1.5 mb-4">
                <li>对象: <span className="font-semibold text-white">{opponentNames[aiAnalysis.attackTarget.playerId] || aiAnalysis.attackTarget.playerId}</span></li>
                <li>槽位: <span className="font-semibold text-white">第 {aiAnalysis.attackTarget.tileIndex + 1} 张盲牌</span></li>
                <li className="mt-2 text-base">预测: <span className="font-bold text-red-400 text-lg ml-1">{aiAnalysis.attackTarget.expectedNumber}</span></li>
              </ul>
              
              <div className="mt-auto">
                <div className="flex justify-between text-[10px] mb-1.5 uppercase font-bold tracking-wider">
                  <span className="text-gray-400">击杀置信度</span>
                  <span className="text-green-400">{(aiAnalysis.attackTarget.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-900 rounded-full h-2 overflow-hidden border border-gray-700">
                  <div className="bg-gradient-to-r from-green-500 to-green-400 h-2 rounded-full transition-all duration-1000" style={{ width: `${aiAnalysis.attackTarget.confidence * 100}%` }}></div>
                </div>
              </div>
            </div>
          )}

          {aiAnalysis?.recommendedAction === 'STOP' && (
            <div className="p-3 bg-red-900/20 rounded-lg border border-red-800/50 transition relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-red-500/10 blur-2xl"></div>
              <h4 className="text-base font-bold text-red-400 text-center mb-1 flex justify-center items-center gap-2">
                <span>🛑</span> 触发防守边界：建议停手
              </h4>
              <p className="text-xs text-red-300/80 text-center mt-2 leading-relaxed">
                {aiAnalysis.reasoning}
              </p>
            </div>
          )}
          
          {aiAnalysis?.recommendedAction === 'GUESS' && (
            <div className="p-3 bg-green-900/20 rounded-lg border border-green-800/50 transition">
              <h4 className="text-sm font-bold text-green-400 text-center flex justify-center items-center gap-2">
                <span>⚡</span> 继续攻击 (Continue)
              </h4>
              <p className="text-xs text-green-300/80 text-center mt-1">
                收益期望仍为正，存在明显高置信度目标。
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
