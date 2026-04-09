import type { GameAction } from '../types';

interface ActionHistoryProps {
  logs: GameAction[];
}

export function ActionHistory({ logs }: ActionHistoryProps) {
  return (
    <div className="w-full h-full flex flex-col pl-4 mt-2">
      <div className="relative border-l-2 border-indigo-500/30 space-y-[22px] min-h-[100px] pb-4">
        {logs.length === 0 && (
          <p className="text-xs text-gray-500 italic relative -left-4 pl-4 pt-1">暂无对局记录，请使用上方控制器录入动作...</p>     
        )}

        {logs.map((log) => (
          <div key={log.id} className="relative group">
            {/* Timeline bullet */}
            <span className={`absolute -left-[23px] top-1 w-3 h-3 rounded-full border-[3px] border-[#0B1120] ${
              log.type === 'DRAW' ? 'bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)]' :
              log.type === 'GUESS' && log.isHit ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' :
              log.type === 'GUESS' && !log.isHit ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]' :
              'bg-gray-500 shadow-[0_0_8px_rgba(107,114,128,0.8)]'
            } transition-all duration-300 group-hover:scale-125`}></span>

            <div className="text-sm text-gray-300 pl-4 py-1.5 px-3 bg-gray-800/40 rounded-lg border border-white/5 shadow-sm group-hover:bg-gray-800/60 transition-colors ml-2 -mt-1.5">
              {log.humanReadable}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
