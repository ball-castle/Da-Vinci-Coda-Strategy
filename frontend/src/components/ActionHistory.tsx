import type { GameAction } from '../types';

interface ActionHistoryProps {
  logs: GameAction[];
}

export function ActionHistory({ logs }: ActionHistoryProps) {
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-lg h-full flex flex-col overflow-hidden">
      <div className="p-4 bg-gray-900 border-b border-gray-700">
        <h3 className="text-sm uppercase tracking-wider font-bold text-gray-400">Action History</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        <div className="relative pl-4 border-l-2 border-gray-600 space-y-5">
          {logs.length === 0 && (
            <p className="text-xs text-gray-500 italic">暂无事件记录...</p>
          )}

          {logs.map((log) => (
            <div key={log.id} className="relative">
              <span className={`absolute -left-[23px] top-1 w-2.5 h-2.5 rounded-full border-2 border-gray-800 ${
                log.type === 'DRAW' ? 'bg-blue-500' :
                log.type === 'GUESS' && log.isHit ? 'bg-green-500' :
                log.type === 'GUESS' && !log.isHit ? 'bg-red-500' :
                'bg-gray-400'
              }`}></span>
              
              <div className="text-sm text-gray-300">
                {log.humanReadable}
              </div>
            </div>
          ))}

        </div>
      </div>
    </div>
  );
}
