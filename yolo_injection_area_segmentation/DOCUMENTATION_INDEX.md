# 📚 YOLO注射区域分割模型 - 文档索引

## 📋 文档概览

本项目提供了完整的文档体系，帮助你快速上手和深入使用YOLO注射区域分割模型。

## 📖 文档导航

### 🚀 快速开始
- **[README.md](./README.md)** - 项目主页和概览
- **[QUICK_START.md](./QUICK_START.md)** - 5分钟快速启动指南

### 📘 详细文档
- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - 完整的使用说明和配置文档
- **[ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)** - 环境配置详细说明

### ⚙️ 配置文件
- **[.env.example](./.env.example)** - 环境变量配置示例
- **[config.py](./config.py)** - 模型和应用配置
- **[label_studio_config.xml](./label_studio_config.xml)** - Label Studio标注界面配置

### 🧪 测试和工具
- **[test_predict_request.py](./test_predict_request.py)** - API测试脚本
- **[test_model.py](./test_model.py)** - 模型测试脚本
- **[inspect_model.py](./inspect_model.py)** - 模型检查工具

### 🚀 部署文件
- **[start_ml_backend.sh](./start_ml_backend.sh)** - 一键启动脚本
- **[Dockerfile](./Dockerfile)** - Docker容器配置
- **[docker-compose.yml](./docker-compose.yml)** - Docker Compose配置

## 📋 使用流程

### 第一次使用
1. 📖 阅读 [README.md](./README.md) 了解项目概况
2. 🚀 按照 [QUICK_START.md](./QUICK_START.md) 快速启动
3. ⚙️ 根据需要配置 [.env.example](./.env.example)

### 深入配置
1. 📘 查看 [USAGE_GUIDE.md](./USAGE_GUIDE.md) 了解详细配置
2. 🔧 修改 [config.py](./config.py) 调整模型参数
3. 🧪 运行测试脚本验证配置

### 生产部署
1. 🚀 使用 [start_ml_backend.sh](./start_ml_backend.sh) 启动服务
2. 🐳 或使用 [Dockerfile](./Dockerfile) 进行容器化部署
3. 📊 监控服务状态和性能

## 🔍 问题解决

### 常见问题查找顺序
1. **快速问题** → [QUICK_START.md](./QUICK_START.md) 故障排除部分
2. **配置问题** → [USAGE_GUIDE.md](./USAGE_GUIDE.md) 故障排除部分
3. **环境问题** → [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)
4. **测试验证** → 运行 [test_predict_request.py](./test_predict_request.py)

### 调试工具
- **模型检查**: `python inspect_model.py`
- **API测试**: `python test_predict_request.py`
- **模型测试**: `python test_model.py`
- **详细日志**: 设置 `LOG_LEVEL=DEBUG`

## 📊 文档特色

### 🎯 分层设计
- **入门级**: README + QUICK_START (5分钟上手)
- **进阶级**: USAGE_GUIDE (完整配置)
- **专家级**: 源码 + 配置文件 (深度定制)

### 🔧 实用工具
- **一键启动**: start_ml_backend.sh
- **自动测试**: test_*.py 脚本
- **配置模板**: .env.example
- **Docker支持**: Dockerfile + docker-compose.yml

### 📝 详细说明
- **步骤清晰**: 每个操作都有详细步骤
- **示例丰富**: 大量代码示例和配置示例
- **问题解答**: 常见问题和解决方案
- **性能优化**: 调优建议和最佳实践

## 🚀 快速导航

### 我想要...

#### 🏃‍♂️ 快速启动服务
→ [QUICK_START.md](./QUICK_START.md)

#### 🔧 详细配置系统
→ [USAGE_GUIDE.md](./USAGE_GUIDE.md)

#### 🧪 测试模型功能
→ [test_predict_request.py](./test_predict_request.py)

#### 🐳 Docker部署
→ [Dockerfile](./Dockerfile) + [docker-compose.yml](./docker-compose.yml)

#### 🔍 解决问题
→ 各文档的"故障排除"部分

#### ⚙️ 修改配置
→ [.env.example](./.env.example) + [config.py](./config.py)

## 📞 获取帮助

### 自助解决
1. 查看相关文档的故障排除部分
2. 运行测试脚本检查系统状态
3. 启用DEBUG日志查看详细信息

### 问题报告
提供以下信息：
- 使用的文档和步骤
- 错误日志 (LOG_LEVEL=DEBUG)
- 系统环境信息
- 配置文件内容

## 📈 文档更新

- **版本**: v1.0.0
- **更新日期**: 2025-05-22
- **维护状态**: 活跃维护
- **反馈渠道**: GitHub Issues

---

**💡 提示**: 建议按照文档顺序阅读，从README开始，然后根据需要深入特定文档。
