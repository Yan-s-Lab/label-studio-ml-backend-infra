# 🎯 YOLO注射区域分割模型 - Label Studio ML后端

基于YOLOv11的注射区域分割模型，专为Label Studio标注平台设计的机器学习后端服务。

## 📋 项目概述

这是一个完整的ML后端解决方案，提供：
- 🔮 **智能预测**: 基于YOLOv11的高精度注射区域分割
- 🔗 **无缝集成**: 与Label Studio完美集成
- 🚀 **即插即用**: 简单配置，快速启动
- 📊 **详细日志**: 完整的调试和监控信息
- 🛠️ **灵活配置**: 支持多种部署方式

## 🚀 快速开始

### 1️⃣ 一键启动
```bash
# 进入项目目录
cd yolo_injection_area_segmentation

# 配置环境 (可选)
cp .env.example .env
# 编辑 .env 文件，设置你的 Label Studio URL 和 API Key

# 启动服务
bash start_ml_backend.sh
```

### 2️⃣ 验证服务
```bash
# 健康检查
curl http://localhost:9090/health

# 预期响应
{"model_class":"YOLOInjectionAreaSegmentation","status":"UP"}
```

### 3️⃣ Label Studio集成
1. 在Label Studio中添加ML后端: `http://localhost:9090`
2. 配置标注界面（使用提供的XML配置）
3. 开始使用智能预测功能

## 📁 项目结构

```
yolo_injection_area_segmentation/
├── 📄 README.md                 # 项目说明（本文件）
├── 📄 QUICK_START.md            # 5分钟快速启动指南
├── 📄 USAGE_GUIDE.md            # 详细使用说明文档
├── 📄 .env.example              # 环境配置示例
├── 🚀 start_ml_backend.sh       # 一键启动脚本
├── 🐍 _wsgi.py                  # WSGI应用入口
├── 🧠 model.py                  # YOLO模型实现
├── ⚙️ config.py                 # 配置文件
├── 🧪 test_predict_request.py   # API测试脚本
├── 📦 requirements.txt          # Python依赖
└── 📁 train6/weights/best.pt    # 训练好的模型文件
```

## ⚙️ 配置说明

### 环境变量配置 (.env)
```bash
# Label Studio连接
LABEL_STUDIO_URL=http://192.168.1.124:8080/
LABEL_STUDIO_API_KEY=your_api_key_here

# 模型参数
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.7
IMAGE_SIZE=640
DEVICE=auto

# 服务配置
PORT=9090
HOST=0.0.0.0
LOG_LEVEL=INFO
```

### Label Studio标注配置
```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="injection_area_arm" background="#FFA39E"/>
  </BrushLabels>
</View>
```

## 🔧 功能特性

### ✅ 已实现功能
- [x] YOLOv11分割模型加载和推理
- [x] Label Studio API集成
- [x] 多种图像路径格式支持
- [x] 智能路径解析和fallback机制
- [x] 详细的日志记录和错误处理
- [x] 健康检查和状态监控
- [x] 环境配置文件支持
- [x] 一键启动脚本

### 🎯 支持的图像格式
- **本地绝对路径**: `/path/to/image.jpg`
- **Label Studio本地文件**: `/data/local-files/?d=folder/image.jpg`
- **HTTP/HTTPS URL**: `https://example.com/image.jpg`

### 📊 模型性能
- **精度**: 高精度注射区域检测
- **速度**: < 2秒/图像 (GPU)
- **内存**: < 4GB RAM
- **支持格式**: JPG, PNG, BMP, TIFF

## 🧪 测试验证

### 自动化测试
```bash
# 运行完整测试套件
python test_predict_request.py

# 测试模型加载
python -c "from model import YOLOInjectionAreaSegmentation; print('✅ 模型加载成功')"
```

### 手动API测试
```bash
# 健康检查
curl http://localhost:9090/health

# 预测测试
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks":[{"data":{"image":"/path/to/test/image.jpg"}}]}'
```

## 🔍 故障排除

### 常见问题及解决方案

| 问题 | 解决方案 |
|------|----------|
| 模型文件未找到 | 确保 `train6/weights/best.pt` 存在 |
| CUDA内存不足 | 设置 `DEVICE=cpu` |
| 端口被占用 | 更改 `PORT` 或终止占用进程 |
| API认证失败 | 更新 `LABEL_STUDIO_API_KEY` |

### 调试模式
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
bash start_ml_backend.sh
```

## 📚 文档导航

- 📖 **[快速启动指南](./QUICK_START.md)** - 5分钟快速上手
- 📘 **[详细使用说明](./USAGE_GUIDE.md)** - 完整的配置和使用文档
- ⚙️ **[环境配置示例](./.env.example)** - 所有可配置参数说明

## 🚀 部署选项

### 开发环境
```bash
# 直接启动
python _wsgi.py --port 9090 --host 0.0.0.0
```

### 生产环境
```bash
# 使用Gunicorn
gunicorn --bind 0.0.0.0:9090 --workers 4 _wsgi:app

# 使用uWSGI
uwsgi --http :9090 --wsgi-file _wsgi.py --processes 4
```

### Docker部署
```bash
# 构建镜像
docker build -t yolo-injection-segmentation .

# 运行容器
docker run -p 9090:9090 -e LABEL_STUDIO_URL="http://host:8080" yolo-injection-segmentation
```

## 📈 性能优化

### GPU加速
- 确保CUDA环境正确配置
- 设置 `DEVICE=cuda` 或 `DEVICE=auto`

### 内存优化
- 调整 `IMAGE_SIZE` 参数
- 设置合适的 `MAX_DETECTIONS`

### 并发处理
- 增加 `WORKERS` 数量
- 使用负载均衡器

## 🤝 技术支持

### 获取帮助
- 📖 查看详细文档: [USAGE_GUIDE.md](./USAGE_GUIDE.md)
- 🧪 运行测试脚本: `python test_predict_request.py`
- 🔍 启用调试日志: `LOG_LEVEL=DEBUG`

### 问题报告
提供以下信息以获得更好的支持：
- 错误日志 (设置 `LOG_LEVEL=DEBUG`)
- 系统环境信息
- 配置文件内容
- 复现步骤

## 📄 许可证

本项目基于 MIT 许可证开源。

---

**🎯 让AI标注更智能，让工作更高效！**

**版本**: v1.0.0 | **更新**: 2025-05-22 | **维护**: AI Annotation Studio Team