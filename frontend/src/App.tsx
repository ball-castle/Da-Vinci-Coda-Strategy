import { useState } from 'react'
import './App.css'

type CardTuple = [string, number | string]

interface OpponentAction {
  type: string
  target_color: string
  target_num: number | string
  result: boolean
}

function App() {
  const [myCards, setMyCards] = useState<CardTuple[]>([['B', 2], ['W', 5], ['B', 7]])
  const [opponentCardCount, setOpponentCardCount] = useState<number>(3)
  const [mePublic, setMePublic] = useState<CardTuple[]>([['W', 5]])
  const [oppPublic, setOppPublic] = useState<CardTuple[]>([['B', 4]])
  const [history, setHistory] = useState<OpponentAction[]>([
    { type: 'guess', target_color: 'B', target_num: 8, result: false }
  ])

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleCalculate = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const payload = {
        my_cards: myCards,
        public_cards: {
          me: mePublic,
          opponent: oppPublic
        },
        opponent_card_count: opponentCardCount,
        opponent_history: history
      }

      console.log("Sending payload:", payload)

      const response = await fetch('http://localhost:8000/api/turn', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        throw new Error(`请求错误: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const renderCard = (c: CardTuple, idx: number, type: 'my' | 'mepub' | 'opppub') => {
    const isBlack = c[0] === 'B'
    return (
      <div key={`${type}-${idx}`} className={`playing-card ${isBlack ? 'card-black' : 'card-white'}`}>
        {c[1] === '-' ? 'Joker' : c[1]}
      </div>
    )
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="glare"></div>
        <h1>达芬奇密码极客推演终端</h1>
        <p className="subtitle">基于不完全信息博弈的贝叶斯期望决策模型</p>
      </header>

      <main className="main-content">
        <div className="dashboard">
          
          <section className="glass-panel input-section">
            <h2>状态面板 (State Configuration)</h2>
            
            <div className="input-group">
              <label>我的手牌 (My Hand)</label>
              <div className="card-row">
                {myCards.map((c, i) => renderCard(c, i, 'my'))}
              </div>
              <p className="hint">示例数据: B2, W5, B7</p>
            </div>

            <div className="input-group">
              <label>场上明牌 - 我方</label>
              <div className="card-row">
                {mePublic.map((c, i) => renderCard(c, i, 'mepub'))}
              </div>
            </div>

            <div className="input-group">
              <label>场上明牌 - 敌方 (剩余暗牌: {opponentCardCount - oppPublic.length})</label>
              <div className="card-row">
                {oppPublic.map((c, i) => renderCard(c, i, 'opppub'))}
              </div>
            </div>

            <div className="input-group">
              <label>心理战日志 (Opponent History)</label>
              <div className="history-box">
                {history.map((h, i) => (
                  <div key={i} className="history-item">
                    <span className="badge">猜测记录</span>
                    敌方猜了 {h.target_color === 'B' ? '黑' : '白'} {h.target_num}，结果: {h.result ? '命中' : '失败'}
                  </div>
                ))}
              </div>
            </div>

            <button 
              className={`calc-btn ${loading ? 'loading' : ''}`} 
              onClick={handleCalculate}
              disabled={loading}
            >
              {loading ? '引擎推算中...' : '生成最高胜率决策'}
            </button>
            {error && <div className="error-msg">{error}</div>}
          </section>

          <section className="glass-panel output-section">
            <h2>决策控制台 (Decision Engine)</h2>
            
            {!result && !loading && (
              <div className="empty-state">
                <div className="radar"></div>
                <p>等待初始化环境数据...</p>
              </div>
            )}

            {result && result.best_move && (
              <div className="result-card success-card">
                <div className="result-header">
                  <h3>🎯 最高期望行动 (Best Move)</h3>
                </div>
                <div className="result-body">
                  <div className="stat-row">
                    <span className="stat-label">目标位置</span>
                    <span className="stat-value highlight">对手第 {result.best_move.target_index + 1} 张牌</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">建议猜测</span>
                    <span className="stat-value highlight">
                      {result.best_move.guess_card[0] === 'B' ? '黑色' : '白色'} 
                      {' '}
                      {result.best_move.guess_card[1] === '-' ? 'Joker' : result.best_move.guess_card[1]}
                    </span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">命中胜率 (贝叶斯后验)</span>
                    <span className="stat-value win-rate">{(result.best_move.win_probability * 100).toFixed(2)}%</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">数学期望 (EV)</span>
                    <span className="stat-value ev">+ {result.best_move.expected_value.toFixed(2)}</span>
                  </div>
                </div>
                
                <div className="info-footer">
                  <p>当前计算遍历了 <strong>{result.search_space_size}</strong> 个合法的平行世界假设树。</p>
                </div>
              </div>
            )}

            {result && !result.best_move && (
              <div className="result-card warning-card">
                <h3>⚠️ 停止警告</h3>
                <p>根据当前场面的风险评估，没有任何行动是正收益期望。引擎建议：停止继续猜牌，保留底牌信息。</p>
              </div>
            )}
            
          </section>

        </div>
      </main>
    </div>
  )
}

export default App
