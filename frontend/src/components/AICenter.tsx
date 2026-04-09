import type { AIAnalysisResponse } from '../services/api';

interface AICenterProps {
  aiAnalysis: AIAnalysisResponse | null;
  isAILoading: boolean;
  error?: string | null;
  opponentNames: Record<string, string>;
  onTrigger: () => void;
}

export function AICenter({ aiAnalysis, isAILoading, error, opponentNames, onTrigger }: AICenterProps) {
  const drawLabel = aiAnalysis?.recommendedAction === 'DRAW'
    ? (aiAnalysis.drawRecommendation?.color === 'white' ? '白牌 (White)' : '黑牌 (Black)')
    : '本回合无需补牌';

  return (
    <div className="flex flex-col h-full overflow-hidden text-gray-200">
      <div className="p-4 bg-gray-800/40 border-b border-gray-700/50 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 flex items-center tracking-tight">
            <svg className="w-5 h-5 mr-3 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
            战术决策终端
          </h2>
          <span className="text-[10px] uppercase font-mono font-bold text-indigo-400/80 bg-indigo-400/10 px-2 py-1 rounded shadow-[0_0_10px_rgba(99,102,241,0.2)] border border-indigo-500/20">
            Nexus.Online
          </span>
        </div>

        <button
          onClick={onTrigger}
          disabled={isAILoading}
          className="w-full relative overflow-hidden group py-3 px-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 active:scale-[0.98] transition-all rounded-xl shadow-[0_0_20px_rgba(99,102,241,0.3)] disabled:opacity-50 disabled:cursor-not-allowed border border-white/10"
        >
          <div className="absolute inset-0 w-full h-full bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <div className="absolute top-0 left-0 w-full h-[2px] bg-white/40 blur-[1px] -translate-y-full group-hover:animate-[scan_2s_ease-in-out_infinite]"></div>

          <div className="flex items-center justify-center gap-2 relative z-10">
            {isAILoading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                <span className="font-bold text-sm tracking-widest text-shadow-sm">建立精神链接中...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                <span className="font-bold text-sm tracking-widest text-shadow-sm uppercase">执行深度推演分析</span>
              </>
            )}
          </div>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-5 space-y-6 relative">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:20px_20px] [mask-image:linear-gradient(to_bottom,transparent,black,transparent)] pointer-events-none"></div>

        {!aiAnalysis && !isAILoading && !error && (
          <div className="flex flex-col items-center justify-center py-10 space-y-4 opacity-60">
            <svg className="w-12 h-12 text-gray-500 drop-shadow-[0_0_15px_rgba(107,114,128,0.5)]" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
            <span className="text-xs font-mono tracking-widest text-gray-400">等待操作指令录入</span>
          </div>
        )}

        {error && !isAILoading && (
          <div className="flex flex-col items-center justify-center py-10 space-y-4">
            <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center border border-red-500/20">
              <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
            </div>
            <span className="text-sm font-bold tracking-widest text-red-400">{error}</span>
            <span className="text-[10px] text-gray-500">无法连接至决策引擎 / 树崩溃</span>
          </div>
        )}

        {(aiAnalysis || isAILoading) && (
          <div className="relative pl-4 border-l-2 border-blue-500/50 pb-2">
            <h3 className="text-[11px] font-black text-blue-400/80 uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full shadow-[0_0_5px_rgba(59,130,246,0.8)]"></span>
              第一阶段 // 补牌策略
            </h3>

            {!aiAnalysis ? (
              <div className="h-16 flex items-center justify-center bg-gray-800/30 rounded border border-gray-700/50">
                <span className="text-xs text-blue-400 animate-pulse font-mono tracking-wider">计算中...</span>
              </div>
            ) : (
              <div className="p-4 bg-gray-800/40 backdrop-blur-sm rounded-xl border border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.05)] relative overflow-hidden group transition-all hover:bg-gray-800/60">
                <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 blur-2xl group-hover:bg-blue-500/10 transition-all"></div>
                <div className="text-sm text-gray-200 flex items-center gap-3 relative z-10">
                  <span className="text-gray-400 font-medium">建议抽取</span>
                  <span className="px-3 py-1 bg-black text-white font-extrabold text-xs tracking-wider rounded border border-gray-600 shadow-[2px_2px_0px_rgba(255,255,255,0.1)]">
                    {drawLabel}
                  </span>
                </div>
                <p className="text-[11px] text-blue-300/70 mt-3 font-mono leading-relaxed relative z-10 border-t border-blue-500/10 pt-2">
                  <span className="text-blue-400 font-bold">» 推演记录:</span>{' '}
                  {aiAnalysis.drawRecommendation?.reasoning ?? '当前回合不需要补牌，系统已直接给出后续动作建议。'}
                </p>
              </div>
            )}
          </div>
        )}

        {(aiAnalysis || isAILoading) && (
          <div className="relative pl-4 border-l-2 border-purple-500/50 pb-2 mt-4">
            <h3 className="text-[11px] font-black text-purple-400/80 uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-purple-500 rounded-full shadow-[0_0_5px_rgba(168,85,247,0.8)]"></span>
              第二阶段 // 攻击嗅探
            </h3>

            {!aiAnalysis ? (
              <div className="h-32 flex items-center justify-center bg-gray-800/30 rounded border border-gray-700/50">
                <span className="text-xs text-purple-400 animate-pulse font-mono tracking-wider">搜寻弱点目标...</span>
              </div>
            ) : (
              <div className="space-y-4">
                {aiAnalysis.recommendedAction === 'DRAW' && (
                  <div className="p-4 bg-blue-900/10 backdrop-blur-sm rounded-xl border border-blue-500/30 shadow-[0_0_20px_rgba(59,130,246,0.08)] relative overflow-hidden">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                      <h4 className="text-sm font-black text-blue-300 tracking-wider">当前先补牌，再进入攻击判断</h4>
                    </div>
                    <p className="text-[11px] text-blue-200/80 font-mono mt-2 leading-relaxed">
                      {aiAnalysis.reasoning}
                    </p>
                  </div>
                )}

                {aiAnalysis.recommendedAction === 'STOP' && (
                  <div className="p-4 bg-red-900/10 backdrop-blur-sm rounded-xl border border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.1)] relative overflow-hidden">
                    <div className="absolute -right-4 -top-4 text-7xl opacity-5">🛑</div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-ping"></div>
                      <h4 className="text-sm font-black text-red-400 tracking-wider">防守阈值触发 // 建议停手</h4>
                    </div>
                    <p className="text-[11px] text-red-300/80 font-mono mt-2 leading-relaxed">
                      {aiAnalysis.reasoning}
                    </p>
                  </div>
                )}

                {aiAnalysis.recommendedAction === 'GUESS' && (
                  <div className="p-4 bg-emerald-900/10 backdrop-blur-sm rounded-xl border border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.1)] transition-all">
                    <div className="flex items-center gap-2 mb-3 border-b border-emerald-500/20 pb-2">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full shadow-[0_0_5px_#34d399]"></div>
                      <h4 className="text-sm font-black text-emerald-400 tracking-wider">发现机会 // 进攻许可</h4>
                      <span className="ml-auto text-[10px] text-emerald-500/60 font-mono">EXPECTED(V) {'>'} 0</span>
                    </div>

                    {aiAnalysis.attackTarget && (
                      <div className="mt-3 relative z-10 space-y-3">
                        <div className="flex justify-between items-end">
                          <div className="flex flex-col gap-1.5">
                            <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Target Entity</span>
                            <span className="text-sm font-black text-white bg-white/5 px-2 py-1 rounded border border-white/10 uppercase">
                              {opponentNames[aiAnalysis.attackTarget.playerId] || aiAnalysis.attackTarget.playerId}
                              <span className="text-gray-400 mx-1">/</span>
                              SLOT {aiAnalysis.attackTarget.tileIndex + 1}
                            </span>
                          </div>
                          <div className="flex flex-col gap-1 items-end">
                            <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Pred. Value</span>
                            <span className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-br from-emerald-300 to-cyan-300 drop-shadow-[0_0_5px_rgba(52,211,153,0.5)]">
                              {aiAnalysis.attackTarget.expectedNumber === '-' ? 'JOKER' : aiAnalysis.attackTarget.expectedNumber}
                            </span>
                          </div>
                        </div>

                        <div className="pt-2">
                          <div className="flex justify-between text-[10px] mb-1.5 uppercase font-bold tracking-widest text-emerald-500/80">
                            <span>击杀概率 (Confidence)</span>
                            <span>{(aiAnalysis.attackTarget.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-[#0B1120] rounded-full h-1.5 overflow-hidden border border-emerald-900/50">
                            <div
                              className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full rounded-full relative"
                              style={{ width: `${aiAnalysis.attackTarget.confidence * 100}%` }}
                            >
                              <div className="absolute top-0 right-0 w-4 h-full bg-white/50 blur-[2px]"></div>
                            </div>
                          </div>
                        </div>

                        <p className="text-[11px] text-emerald-200/80 font-mono pt-2 border-t border-emerald-500/10">
                          {aiAnalysis.reasoning}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
