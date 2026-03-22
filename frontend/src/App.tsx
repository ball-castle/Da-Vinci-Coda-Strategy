import { useState } from 'react'
import './App.css'

type CardValue = number | string
type CardTuple = [string, CardValue]

interface OpponentAction {
  type: string
  target_color: string
  target_num: CardValue
  result: boolean
  continued_turn?: boolean
}

interface CardSlotPayload {
  slot_index: number
  color?: string
  value?: CardValue
  is_revealed: boolean
  is_newly_drawn: boolean
}

interface PlayerStatePayload {
  player_id: string
  slots: CardSlotPayload[]
}

interface GuessActionPayload {
  guesser_id: string
  target_player_id: string
  target_slot_index?: number
  guessed_color?: string
  guessed_value?: CardValue
  result: boolean
  continued_turn?: boolean
  action_type: string
}

interface TurnRequestPayload {
  state: {
    self_player_id: string
    target_player_id: string
    players: PlayerStatePayload[]
    actions: GuessActionPayload[]
  }
}

interface Candidate {
  card: CardTuple
  probability: number
}

interface ProbabilityPosition {
  target_index: number
  target_slot_index: number
  target_scope: string
  candidates: Candidate[]
}

interface Move {
  target_index: number
  target_slot_index?: number
  guess_card: CardTuple
  win_probability: number
  expected_value: number
  information_gain: number
  target_scope: string
}

interface TurnResponse {
  best_move: Move | null
  top_moves: Move[]
  probability_matrix: ProbabilityPosition[]
  search_space_size: number
  opponent_hidden_count: number
  risk_factor: number
  effective_weight_sum?: number
  should_stop: boolean
  input_summary?: {
    self_player_id: string
    target_player_id: string
    player_count: number
    target_total_slots: number
    target_hidden_count: number
    action_count: number
  }
}

function App() {
  const [myCards] = useState<CardTuple[]>([['B', 2], ['W', 5], ['B', 7]])
  const [opponentCardCount] = useState<number>(3)
  const [mePublic] = useState<CardTuple[]>([['W', 5]])
  const [oppPublic] = useState<CardTuple[]>([['B', 4]])
  const [history] = useState<OpponentAction[]>([
    { type: 'guess', target_color: 'B', target_num: 8, result: false, continued_turn: false }
  ])

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<TurnResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleCalculate = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const payload = buildStructuredPayload(myCards, mePublic, oppPublic, opponentCardCount, history)

      console.log('Sending payload:', payload)

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

      const data: TurnResponse = await response.json()
      setResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : '未知错误'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const renderCard = (card: CardTuple, idx: number, type: 'my' | 'mepub' | 'opppub') => {
    const isBlack = card[0] === 'B'
    return (
      <div key={`${type}-${idx}`} className={`playing-card ${isBlack ? 'card-black' : 'card-white'}`}>
        {card[1] === '-' ? 'Joker' : card[1]}
      </div>
    )
  }

  const bestMove = result?.best_move
  const targetPositionLabel = bestMove
    ? `对手第 ${(bestMove.target_slot_index ?? bestMove.target_index) + 1} 张牌`
    : ''
  const topCandidates = result?.probability_matrix[0]?.candidates.slice(0, 3) ?? []

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
                {myCards.map((card, index) => renderCard(card, index, 'my'))}
              </div>
              <p className="hint">当前前端已改为发送结构化 state，不再依赖旧式散字段。</p>
            </div>

            <div className="input-group">
              <label>场上明牌 - 我方</label>
              <div className="card-row">
                {mePublic.map((card, index) => renderCard(card, index, 'mepub'))}
              </div>
            </div>

            <div className="input-group">
              <label>场上明牌 - 敌方 (剩余暗牌: {opponentCardCount - oppPublic.length})</label>
              <div className="card-row">
                {oppPublic.map((card, index) => renderCard(card, index, 'opppub'))}
              </div>
            </div>

            <div className="input-group">
              <label>心理战日志 (Opponent History)</label>
              <div className="history-box">
                {history.map((item, index) => (
                  <div key={index} className="history-item">
                    <span className="badge">猜测记录</span>
                    敌方猜了 {item.target_color === 'B' ? '黑' : '白'} {item.target_num}，
                    结果: {item.result ? '命中' : '失败'}
                    {item.continued_turn ? '，并继续了回合' : ''}
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

            {result && bestMove && (
              <div className="result-card success-card">
                <div className="result-header">
                  <h3>🎯 最高期望行动 (Best Move)</h3>
                </div>
                <div className="result-body">
                  <div className="stat-row">
                    <span className="stat-label">目标位置</span>
                    <span className="stat-value highlight">{targetPositionLabel}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">建议猜测</span>
                    <span className="stat-value highlight">
                      {bestMove.guess_card[0] === 'B' ? '黑色' : '白色'}{' '}
                      {bestMove.guess_card[1] === '-' ? 'Joker' : bestMove.guess_card[1]}
                    </span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">命中胜率 (贝叶斯后验)</span>
                    <span className="stat-value win-rate">{(bestMove.win_probability * 100).toFixed(2)}%</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">数学期望 (EV)</span>
                    <span className="stat-value ev">{bestMove.expected_value >= 0 ? '+' : ''}{bestMove.expected_value.toFixed(2)}</span>
                  </div>
                </div>

                <div className="info-footer">
                  <p>当前计算遍历了 <strong>{result.search_space_size}</strong> 个合法的平行世界假设树。</p>
                  <p>当前目标玩家共有 <strong>{result.input_summary?.target_total_slots ?? 0}</strong> 个槽位，其中暗牌 <strong>{result.opponent_hidden_count}</strong> 张。</p>
                  {topCandidates.length > 0 && (
                    <p>
                      首个暗位 Top3:
                      {' '}
                      {topCandidates
                        .map((candidate) => `${candidate.card[0]}${candidate.card[1]} ${(candidate.probability * 100).toFixed(1)}%`)
                        .join(' / ')}
                    </p>
                  )}
                </div>
              </div>
            )}

            {result && !bestMove && (
              <div className="result-card warning-card">
                <h3>⚠️ 停止警告</h3>
                <p>根据当前场面的风险评估，没有任何行动是正收益期望。引擎建议：停止继续猜牌，保留底牌信息。</p>
                {result.top_moves.length > 0 && (
                  <p>
                    当前最接近可出的行动：
                    {' '}
                    {result.top_moves[0].guess_card[0] === 'B' ? '黑' : '白'}
                    {result.top_moves[0].guess_card[1] === '-' ? 'Joker' : result.top_moves[0].guess_card[1]}
                    ，EV {result.top_moves[0].expected_value.toFixed(2)}
                  </p>
                )}
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}

function buildStructuredPayload(
  myCards: CardTuple[],
  mePublic: CardTuple[],
  oppPublic: CardTuple[],
  opponentCardCount: number,
  history: OpponentAction[],
): TurnRequestPayload {
  const mePublicKeySet = new Set(mePublic.map(cardKey))
  const mySlots: CardSlotPayload[] = myCards.map(([color, value], slotIndex) => ({
    slot_index: slotIndex,
    color,
    value,
    is_revealed: mePublicKeySet.has(cardKey([color, value])),
    is_newly_drawn: false,
  }))

  const opponentSlots: CardSlotPayload[] = oppPublic.map(([color, value], slotIndex) => ({
    slot_index: slotIndex,
    color,
    value,
    is_revealed: true,
    is_newly_drawn: false,
  }))

  const hiddenCount = Math.max(0, opponentCardCount - oppPublic.length)
  for (let offset = 0; offset < hiddenCount; offset += 1) {
    opponentSlots.push({
      slot_index: opponentSlots.length,
      is_revealed: false,
      is_newly_drawn: false,
    })
  }

  return {
    state: {
      self_player_id: 'me',
      target_player_id: 'opponent',
      players: [
        { player_id: 'me', slots: mySlots },
        { player_id: 'opponent', slots: opponentSlots },
      ],
      actions: history.map((action) => ({
        guesser_id: 'opponent',
        target_player_id: 'me',
        guessed_color: action.target_color,
        guessed_value: action.target_num,
        result: action.result,
        continued_turn: action.continued_turn,
        action_type: action.type,
      })),
    },
  }
}

function cardKey(card: CardTuple): string {
  return `${card[0]}:${String(card[1])}`
}

export default App
