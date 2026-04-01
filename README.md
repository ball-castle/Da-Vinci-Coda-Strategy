# 🎴 DaVinci POMDP AI - 达芬奇密码AI决策助手

<div align="center">

**基于MCTS算法的达芬奇密码游戏AI决策支持系统**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-19.2+-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.9+-3178c6.svg)](https://www.typescriptlang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[功能特性](#✨-核心特性) •
[快速开始](#🚀-快速开始) •
[文档](#📚-文档) •
[贡献指南](#🤝-贡献指南) •
[许可证](#📄-许可证)

</div>

## 📖 项目简介

DaVinci POMDP AI 是一个针对达芬奇密码（Coda）卡牌游戏的智能决策支持系统。通过结合**蒙特卡洛树搜索（MCTS）**算法和**部分可观测马尔可夫决策过程（POMDP）**建模，为玩家提供实时的最优策略建议。

### ✨ 核心特性

- 🧠 **智能决策引擎**: 基于MCTS算法的混合决策树，支持深度探索和概率推理
- 🎯 **实时策略建议**: 自动分析当前局势，高亮显示最佳猜测目标
- 📊 **信息推理系统**: 利用游戏历史和对手行为进行概率推断
- 🎨 **现代化UI界面**: React + TypeScript + TailwindCSS构建的响应式界面
- ⚡ **高性能计算**: FastAPI后端，支持并发请求和实时状态同步
- 📱 **移动端支持**: 基于Capacitor的跨平台移动应用

## 🏗️ 技术栈

### 前端
- **框架**: React 19.2 + TypeScript 5.9
- **构建工具**: Vite 8.0
- **样式**: TailwindCSS 4.2
- **UI交互**: @dnd-kit (拖拽支持)
- **移动端**: Capacitor 8.3
- **代码规范**: ESLint + TypeScript ESLint

### 后端
- **框架**: FastAPI 0.104
- **服务器**: Uvicorn 0.24
- **数据验证**: Pydantic 2.5
- **核心算法**: 自研MCTS引擎
- **测试**: pytest

## 🚀 快速开始

### 前置要求

- **Node.js**: >= 18.0
- **Python**: >= 3.8
- **包管理器**: npm 或 yarn
- **Python包管理**: pip 或 conda

### 📦 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/ball-castle/Da-Vinci-Coda-Strategy.git
cd Da-Vinci-Coda-Strategy
```

#### 2. 安装前端依赖

```bash
cd frontend
npm install
```

#### 3. 安装后端依赖

```bash
cd ../backend
pip install -r requirements.txt
```

### 🎮 运行项目

项目采用前后端分离架构，需要同时启动两个服务：

#### 启动后端服务

```bash
# 在项目根目录下
cd backend
python -m uvicorn app.main:app --port 8000 --reload
```

看到以下提示说明后端启动成功：
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

#### 启动前端服务

新开一个终端：

```bash
# 在项目根目录下
cd frontend
npm run dev
```

看到以下提示说明前端启动成功：
```
VITE v8.0.1  ready in xxx ms

➜  Local:   http://localhost:5173/
```

在浏览器中打开 `http://localhost:5173` 即可使用。

## 📁 项目结构

```
DaVinci/
├── frontend/                 # 前端项目
│   ├── src/
│   │   ├── components/      # React组件
│   │   ├── hooks/           # 自定义Hooks
│   │   ├── types/           # TypeScript类型定义
│   │   └── utils/           # 工具函数
│   ├── public/              # 静态资源
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                 # 后端项目
│   ├── app/
│   │   ├── main.py          # FastAPI应用入口
│   │   ├── api/             # API路由
│   │   ├── core/            # 核心算法（MCTS）
│   │   ├── models/          # 数据模型
│   │   └── utils/           # 工具函数
│   ├── tests/               # 测试文件
│   └── requirements.txt
│
├── docs/                    # 文档
├── score/                   # 评分系统
└── README.md
```

## 🎯 使用指南

### 基本工作流程

1. **启动应用**: 按照上述步骤启动前后端服务
2. **记录游戏操作**: 在右上角的"ActionLogger/游戏记录器"面板中记录实际游戏中的操作
   - "我摸了一张黑牌"
   - "对手猜了某张牌"
   - 等等...
3. **自动状态同步**: 每次提交动作后，前端会自动进行状态同步（Auto-sync）
4. **获取AI建议**: 页面下方的"AI战术中枢"会自动拉取最新的MCTS探索结果
5. **执行策略**: AI会告诉你应该"继续猜（GUESS）"还是"停手（STOP）"，并用高亮标注最优目标
6. **撤销操作**: 如果操作有误，可以通过"Undo/撤销"按钮退回上一步

### API文档

后端服务启动后，可以访问以下地址查看自动生成的API文档：

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 🧪 开发指南

### 前端开发

```bash
cd frontend

# 开发模式（热重载）
npm run dev

# 代码检查
npm run lint

# 构建生产版本
npm run build

# 预览生产构建
npm run preview
```

### 后端开发

```bash
cd backend

# 运行测试
pytest

# 代码格式化
black app/

# 类型检查
mypy app/

# 启动开发服务器（自动重载）
python -m uvicorn app.main:app --reload
```

## 📚 文档

- [后端文档](backend/README.md) - API接口和MCTS算法说明
- [前端文档](frontend/README.md) - 组件和状态管理
- [配置指南](docs/CONFIGURATION.md) - 环境变量和配置说明
- [贡献指南](CONTRIBUTING.md) - 如何参与项目开发
- [更新日志](CHANGELOG.md) - 版本更新记录（计划中）

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请查看 [贡献指南](CONTRIBUTING.md) 了解详情。

### 提交规范

- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建/工具链更新

### 开发流程

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 📝 算法说明

### MCTS（蒙特卡洛树搜索）

本项目实现了混合MCTS算法，结合了：
- **UCB1选择策略**: 平衡探索与利用
- **概率推理**: 基于游戏历史的对手手牌推断
- **启发式评估**: 考虑牌型分布和胜率估计
- **并行模拟**: 提升搜索效率

### POMDP建模

- **状态空间**: 包括所有可能的手牌组合和公开信息
- **观测模型**: 基于对手行为更新信念状态
- **奖励函数**: 考虑胜率、风险和信息获取
- **策略优化**: 通过MCTS找到近似最优策略

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 维护者

- [@ball-castle](https://github.com/ball-castle)

## 🙏 致谢

- 感谢所有贡献者的付出
- 灵感来源于达芬奇密码桌游和AlphaGo的MCTS实现

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/ball-castle/Da-Vinci-Coda-Strategy/issues)
- 发送邮件至维护者

---

<div align="center">
Made with ❤️ by DaVinci POMDP AI Team
</div>
