# DaVinci POMDP AI - 后端服务

基于FastAPI的MCTS决策引擎后端服务

## 📋 目录

- [架构概览](#架构概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
- [API接口](#api接口)
- [MCTS算法](#mcts算法)
- [开发指南](#开发指南)
- [测试](#测试)
- [部署](#部署)

## 🏗️ 架构概览

后端采用**分层架构**设计：

```
┌─────────────────┐
│   API Layer     │  FastAPI路由和请求处理
├─────────────────┤
│  Business Logic │  游戏控制器和决策引擎
├─────────────────┤
│   Core Engine   │  MCTS算法实现
├─────────────────┤
│   Data Models   │  游戏状态和数据结构
└─────────────────┘
```

## 💻 技术栈

- **Web框架**: FastAPI 0.104.1
- **ASGI服务器**: Uvicorn 0.24.0
- **数据验证**: Pydantic 2.5.2
- **配置管理**: Pydantic Settings 2.1.0
- **核心算法**: 自研MCTS引擎
- **测试框架**: pytest (计划中)

## 📁 项目结构

```
backend/
├── app/
│   ├── main.py              # FastAPI应用入口
│   ├── config.py            # 配置管理
│   ├── exceptions.py        # 自定义异常
│   ├── middleware.py        # 错误处理中间件
│   │
│   ├── api/                 # API路由层
│   │   ├── endpoints.py     # API端点定义
│   │   └── __init__.py
│   │
│   ├── core/                # 核心业务逻辑
│   │   ├── engine.py        # 游戏控制器
│   │   ├── mcts.py          # MCTS算法实现（推测）
│   │   ├── state.py         # 游戏状态管理
│   │   ├── session.py       # 会话管理
│   │   └── __init__.py
│   │
│   └── backend/             # 后端辅助模块
│       └── ...
│
├── tests/                   # 测试文件
├── scripts/                 # 脚本工具
├── requirements.txt         # Python依赖
├── .env.example            # 环境变量模板
└── README.md               # 本文档
```

## 🔧 核心模块

### 1. 配置管理 (`config.py`)

使用Pydantic Settings进行类型安全的配置管理：

```python
from app.config import settings

# 访问配置
iterations = settings.mcts_iterations
port = settings.port
```

**主要配置项**：
- 服务器配置（host, port, reload）
- CORS设置
- MCTS算法参数
- 日志级别
- 语音服务配置（可选）

### 2. 异常处理 (`exceptions.py`)

自定义异常类型：
- `ValidationError`: 数据验证错误 (400)
- `GameStateError`: 游戏状态错误 (422)
- `SessionNotFoundError`: 会话未找到 (404)
- `MCTSComputationError`: MCTS计算错误 (500)
- `ConfigurationError`: 配置错误 (500)

### 3. 游戏状态管理 (`core/state.py`)

核心数据结构：
- `CardSlot`: 卡槽状态
- `PlayerState`: 玩家状态
- `GameState`: 完整游戏状态
- `GuessAction`: 猜测行动记录

### 4. MCTS引擎 (`core/engine.py`)

游戏控制器，负责：
- 状态验证和预处理
- 调用MCTS算法
- 结果格式化和返回

### 5. 会话管理 (`core/session.py`)

管理用户会话：
- 会话创建和检索
- 历史记录持久化
- 会话过期管理

## 🌐 API接口

### 基础端点

#### `GET /`
欢迎信息和API版本

**响应示例**：
```json
{
  "message": "Welcome to Da Vinci Coda API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

#### `GET /health`
健康检查端点

**响应示例**：
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 核心API

#### `POST /api/turn`
计算当前回合的最佳策略

**请求体**（结构化格式）：
```json
{
  "session_id": "optional-session-id",
  "state": {
    "self_player_id": "me",
    "target_player_id": "opponent",
    "players": [
      {
        "player_id": "me",
        "slots": [
          {
            "slot_index": 0,
            "color": "B",
            "value": 5,
            "is_revealed": true,
            "is_newly_drawn": false
          }
        ]
      }
    ],
    "actions": []
  }
}
```

**请求体**（旧版格式）：
```json
{
  "my_cards": [["B", 3], ["W", 7]],
  "public_cards": {
    "opponent": [["B", 5]]
  },
  "opponent_card_count": 4,
  "opponent_history": []
}
```

**响应示例**：
```json
{
  "session_id": "generated-or-provided-session-id",
  "action": "GUESS",
  "target_slot_index": 2,
  "confidence": 0.85,
  "reasoning": "高概率目标槽位分析",
  "opponent_hidden_count": 3,
  "input_summary": {
    "self_player_id": "me",
    "target_player_id": "opponent",
    "player_count": 2,
    "target_total_slots": 4,
    "target_hidden_count": 3,
    "action_count": 0
  }
}
```

**错误响应**：
```json
{
  "success": false,
  "error": {
    "type": "ValidationError",
    "message": "卡牌格式错误",
    "detail": {
      "expected_format": "[颜色, 数值]",
      "received": ["B"]
    }
  }
}
```

### API文档

FastAPI自动生成交互式文档：
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 🧮 MCTS算法

### 算法特点

本项目实现了混合蒙特卡洛树搜索算法，针对POMDP（部分可观测马尔可夫决策过程）进行优化。

**核心特性**：
1. **UCB1选择策略**: 平衡探索与利用
2. **概率推理**: 基于对手行为的信念状态更新
3. **启发式评估**: 考虑牌型分布和胜率
4. **并行模拟**: 提升搜索效率（计划中）

### 算法流程

```
1. Selection（选择）
   └─> 使用UCB1公式选择最优子节点

2. Expansion（扩展）
   └─> 添加新的可能行动节点

3. Simulation（模拟）
   └─> 随机模拟游戏直到结束

4. Backpropagation（回传）
   └─> 更新路径上所有节点的统计信息
```

### 可配置参数

在`.env`中配置：

```bash
MCTS_ITERATIONS=1000              # 迭代次数（越大越精确，但更慢）
MCTS_EXPLORATION_CONSTANT=1.414   # UCB1探索常数
MCTS_SIMULATION_DEPTH=50          # 模拟深度限制
MCTS_TIMEOUT_SECONDS=5.0          # 计算超时时间
```

## 👨‍💻 开发指南

### 环境设置

1. **克隆项目**：
```bash
git clone https://github.com/ball-castle/Da-Vinci-Coda-Strategy.git
cd Da-Vinci-Coda-Strategy/backend
```

2. **创建虚拟环境**：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**：
```bash
pip install -r requirements.txt
```

4. **配置环境变量**：
```bash
cp .env.example .env
# 编辑.env文件，填入实际配置
```

### 运行服务

**开发模式**（自动重载）：
```bash
python -m uvicorn app.main:app --reload --port 8000
```

**生产模式**：
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 代码规范

**推荐工具**：
```bash
pip install black flake8 mypy

# 代码格式化
black app/

# 代码检查
flake8 app/

# 类型检查
mypy app/
```

**代码风格**：
- 使用类型注解
- 函数和类添加docstring
- 遵循PEP 8规范

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_state.py

# 生成覆盖率报告
pytest --cov=app --cov-report=html
```

### 测试结构

```
tests/
├── test_api/           # API端点测试
├── test_core/          # 核心逻辑测试
│   ├── test_mcts.py   # MCTS算法测试
│   └── test_state.py  # 状态管理测试
└── conftest.py         # pytest配置和fixtures
```

## 🚀 部署

### Docker部署

```bash
# 构建镜像
docker build -t davinci-backend .

# 运行容器
docker run -p 8000:8000 \
  -e DEBUG=false \
  -e MCTS_ITERATIONS=2000 \
  davinci-backend
```

### 性能优化

1. **使用生产级ASGI服务器**：
   - Uvicorn with workers
   - Gunicorn + Uvicorn workers

2. **启用缓存**：
   - Redis缓存MCTS结果
   - 会话状态缓存

3. **并行计算**：
   - Worker Threads for MCTS
   - 异步I/O优化

## 📊 监控和日志

### 日志配置

在`.env`中设置：
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 日志格式

```
2026-04-01 12:00:00 - app.api.endpoints - INFO - 处理回合请求 - session_id: abc123
2026-04-01 12:00:05 - app.api.endpoints - INFO - 回合计算成功 - session_id: abc123, 决策: GUESS
```

### 健康检查

使用`/health`端点进行服务监控：
```bash
curl http://localhost:8000/health
```

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交Pull Request

## 📝 常见问题

### Q: MCTS计算超时怎么办？
A: 调整`MCTS_TIMEOUT_SECONDS`或减少`MCTS_ITERATIONS`

### Q: 如何提升计算性能？
A: 
- 减少迭代次数
- 使用更快的硬件
- 实现并行计算（计划中）

### Q: API返回422错误？
A: 检查请求数据格式，参考API文档确保数据结构正确

## 📄 许可证

MIT License - 详见根目录LICENSE文件

---

**维护者**: [@ball-castle](https://github.com/ball-castle)
