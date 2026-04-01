# 贡献指南

感谢你考虑为 DaVinci POMDP AI 项目做出贡献！

## 📋 行为准则

- 尊重所有贡献者
- 接受建设性批评
- 专注于对项目最有利的事情
- 对其他社区成员表现出同理心

## 🚀 如何贡献

### 报告Bug

如果你发现了bug，请创建一个Issue并包含：
- 清晰的标题和描述
- 重现步骤
- 预期行为和实际行为
- 截图（如果适用）
- 你的环境信息（OS、Python版本、Node版本等）

### 提出新功能

在提出新功能之前，请：
1. 检查是否已有类似的Issue
2. 创建一个Issue描述该功能
3. 等待维护者的反馈

### 提交代码

1. **Fork项目**
   ```bash
   git clone https://github.com/your-username/Da-Vinci-Coda-Strategy.git
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **进行更改**
   - 遵循代码风格指南
   - 添加必要的测试
   - 更新相关文档

4. **提交更改**
   ```bash
   git commit -m 'feat: Add some amazing feature'
   ```
   
   提交信息格式：
   - `feat:` 新功能
   - `fix:` Bug修复
   - `docs:` 文档更新
   - `style:` 代码格式调整
   - `refactor:` 代码重构
   - `test:` 测试相关
   - `chore:` 构建/工具链更新

5. **推送到分支**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **创建Pull Request**

## 🎨 代码风格

### Python (后端)

- 使用 Black 进行代码格式化
- 使用 flake8 进行代码检查
- 使用 mypy 进行类型检查
- 最大行长度：120字符

```bash
cd backend
black app/
flake8 app/
mypy app/
```

### TypeScript/React (前端)

- 使用 ESLint 进行代码检查
- 遵循 TypeScript 严格模式
- 使用函数式组件和Hooks

```bash
cd frontend
npm run lint
npx tsc --noEmit
```

## 🧪 测试

### 后端测试

```bash
cd backend
pytest tests/ --cov=app
```

确保：
- 所有测试通过
- 代码覆盖率 > 70%
- 新功能包含测试

### 前端测试

```bash
cd frontend
npm run test
```

## 📝 文档

更新相关文档：
- README.md - 新功能或使用方法变更
- API文档 - 新的或修改的API端点
- 代码注释 - 复杂逻辑的说明
- CHANGELOG.md - 记录重要更改

## 🔍 Pull Request 检查清单

提交PR前，请确保：

- [ ] 代码遵循项目风格指南
- [ ] 已添加必要的测试且全部通过
- [ ] 已更新相关文档
- [ ] 提交信息清晰明确
- [ ] PR描述详细说明了更改内容
- [ ] 已在本地测试所有更改
- [ ] 没有合并冲突

## 🤝 开发流程

1. **Issue优先**: 在开始工作前先创建或认领Issue
2. **小步快跑**: 保持PR小而专注
3. **及时沟通**: 有问题随时在Issue或PR中讨论
4. **代码审查**: 耐心等待和响应代码审查意见
5. **持续改进**: 根据反馈不断完善代码

## 📚 资源

- [FastAPI文档](https://fastapi.tiangolo.com/)
- [React文档](https://react.dev/)
- [TypeScript文档](https://www.typescriptlang.org/docs/)
- [Python类型注解](https://docs.python.org/3/library/typing.html)

## ❓ 需要帮助？

如果你有任何问题：
- 查看现有的Issue和文档
- 在Issue中提问
- 联系项目维护者

---

再次感谢你的贡献！🎉
