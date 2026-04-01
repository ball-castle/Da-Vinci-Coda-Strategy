# 配置管理指南

## 概述

项目采用环境变量进行配置管理，前后端分别使用不同的配置文件。

## 后端配置

### 配置文件位置
- 模板文件: `backend/.env.example`
- 实际配置: `backend/.env` (需要自行创建)

### 首次配置步骤

1. 复制模板文件:
```bash
cd backend
cp .env.example .env
```

2. 编辑 `.env` 文件，填入实际值

3. 配置项说明:

#### 应用配置
- `APP_NAME`: 应用名称
- `APP_VERSION`: 版本号
- `DEBUG`: 是否开启调试模式 (true/false)

#### 服务器配置
- `HOST`: 服务器监听地址 (默认: 127.0.0.1)
- `PORT`: 服务器端口 (默认: 8000)
- `RELOAD`: 是否启用热重载 (开发时建议true)

#### CORS配置
- `CORS_ORIGINS`: 允许的前端地址，多个地址用逗号分隔

#### MCTS算法参数
- `MCTS_ITERATIONS`: 搜索迭代次数 (默认: 1000)
- `MCTS_EXPLORATION_CONSTANT`: UCB1探索常数 (默认: 1.414)
- `MCTS_SIMULATION_DEPTH`: 模拟深度 (默认: 50)
- `MCTS_TIMEOUT_SECONDS`: 计算超时时间 (默认: 5.0秒)

#### 语音服务配置
- `VOICE_API_URL`: 语音API地址
- `VOICE_API_KEY`: API密钥
- `VOICE_API_SECRET`: API密钥

### 代码中使用配置

```python
from app.config import settings

# 获取配置值
print(settings.mcts_iterations)
print(settings.port)
```

## 前端配置

### 配置文件位置
- 模板文件: `frontend/.env.example`
- 开发配置: `frontend/.env` (需要自行创建)
- 生产配置: `frontend/.env.production`

### 首次配置步骤

1. 复制模板文件:
```bash
cd frontend
cp .env.example .env
```

2. 编辑 `.env` 文件

3. 配置项说明:

#### API配置
- `VITE_API_BASE_URL`: 后端API基础URL (默认: http://127.0.0.1:8000)
- `VITE_WS_URL`: WebSocket连接地址

#### 应用配置
- `VITE_APP_NAME`: 应用名称
- `VITE_DEV_MODE`: 开发模式开关

### 代码中使用配置

```typescript
// 获取环境变量
const apiUrl = import.meta.env.VITE_API_BASE_URL;
const wsUrl = import.meta.env.VITE_WS_URL;
```

## 安全注意事项

### ⚠️ 重要：不要提交敏感信息

1. `.env` 文件已添加到 `.gitignore`，不会被Git跟踪
2. 永远不要将包含真实密钥的 `.env` 文件提交到Git
3. 前端不要存储API密钥，所有需要密钥的API调用应通过后端代理

### 环境变量优先级

1. 系统环境变量（最高优先级）
2. `.env` 文件
3. 代码中的默认值（最低优先级）

## 不同环境的配置

### 开发环境
使用 `.env` 文件，启用调试和热重载

### 生产环境
- 后端: 使用系统环境变量或 `.env` 文件，关闭调试和热重载
- 前端: 使用 `.env.production` 文件

### 构建生产版本

```bash
# 前端
cd frontend
npm run build  # 自动使用 .env.production

# 后端
cd backend
# 设置环境变量后启动
DEBUG=false python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 常见问题

### Q: 修改 `.env` 后需要重启吗？
A: 是的，需要重启前后端服务使配置生效。

### Q: 为什么我的配置不生效？
A: 
1. 检查文件名是否正确（`.env`）
2. 检查环境变量名称是否正确（前端必须以 `VITE_` 开头）
3. 确认已重启服务
4. 检查是否有系统环境变量覆盖了配置

### Q: 如何在团队中共享配置？
A: 更新 `.env.example` 文件，添加新的配置项和说明，团队成员根据模板更新自己的 `.env` 文件。
