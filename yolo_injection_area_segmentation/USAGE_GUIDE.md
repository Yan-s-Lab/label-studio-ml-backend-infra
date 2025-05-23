# YOLO注射区域分割模型 - 启动使用说明

## 📋 目录
- [系统要求](#系统要求)
- [快速启动](#快速启动)
- [详细配置](#详细配置)
- [Label Studio集成](#label-studio集成)
- [测试验证](#测试验证)
- [故障排除](#故障排除)
- [API使用](#api使用)

## 🔧 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐，支持CUDA)
- **内存**: 至少8GB RAM
- **存储**: 至少2GB可用空间

### 软件要求
- **Python**: 3.8+
- **CUDA**: 11.0+ (如果使用GPU)
- **操作系统**: Linux/macOS/Windows

### 依赖包
```bash
ultralytics>=8.0.0
torch>=1.9.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
label-studio-ml>=1.0.0
```

## 🚀 快速启动

### 1. 环境准备
```bash
# 进入项目目录
cd yolo_injection_area_segmentation

# 安装依赖
pip install -r requirements.txt

# 检查模型文件
ls -la train6/weights/best.pt
```

### 2. 配置环境变量
```bash
# 设置Label Studio连接信息
export LABEL_STUDIO_URL="http://192.168.1.124:8080/"
export LABEL_STUDIO_API_KEY="your_api_key_here"

# 设置模型参数
export CONFIDENCE_THRESHOLD="0.25"
export IOU_THRESHOLD="0.7"
export IMAGE_SIZE="640"
export DEVICE="auto"
export LOG_LEVEL="INFO"
```

### 3. 启动ML后端
```bash
# 方法1: 使用启动脚本 (推荐)
bash start_ml_backend.sh

# 方法2: 直接启动
python _wsgi.py --port 9090 --host 0.0.0.0 --log-level INFO
```

### 4. 验证启动
```bash
# 健康检查
curl http://localhost:9090/health

# 预期响应
{"model_class":"YOLOInjectionAreaSegmentation","status":"UP"}
```

## ⚙️ 详细配置

### 配置文件说明 (`config.py`)
```python
# 模型配置
MODEL_PATH = "train6/weights/best.pt"  # 模型文件路径
MODEL_VERSION = "1.0.0"                # 模型版本
CONFIDENCE_THRESHOLD = 0.25             # 置信度阈值
IOU_THRESHOLD = 0.7                     # IoU阈值
IMAGE_SIZE = 640                        # 输入图像尺寸

# 类别映射
CLASS_MAPPING = {
    0: "arm_injection_area"             # 类别ID到名称的映射
}

# Label Studio配置
LABEL_STUDIO_TASK_DATA_KEY = "image"   # 任务数据中的图像键
LABEL_STUDIO_FROM_NAME = "label"       # 标注来源名称
LABEL_STUDIO_TO_NAME = "image"         # 标注目标名称
```

### 环境变量详解
| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LABEL_STUDIO_URL` | - | Label Studio服务器地址 |
| `LABEL_STUDIO_API_KEY` | - | Label Studio API密钥 |
| `CONFIDENCE_THRESHOLD` | 0.25 | 检测置信度阈值 |
| `IOU_THRESHOLD` | 0.7 | 非极大值抑制IoU阈值 |
| `IMAGE_SIZE` | 640 | 模型输入图像尺寸 |
| `DEVICE` | auto | 计算设备 (auto/cpu/cuda) |
| `MAX_DETECTIONS` | 300 | 最大检测数量 |
| `LOG_LEVEL` | INFO | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

## 🔗 Label Studio集成

### 1. 在Label Studio中添加ML后端
1. 登录Label Studio管理界面
2. 进入项目设置 → Machine Learning
3. 点击"Add Model"
4. 填写以下信息：
   - **URL**: `http://localhost:9090`
   - **Title**: `YOLO注射区域分割`
   - **Description**: `基于YOLOv11的注射区域分割模型`

### 2. 标注配置示例
```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="injection_area_arm" background="#FFA39E"/>
  </BrushLabels>
</View>
```

### 3. 使用预测功能
1. 在标注界面点击"Get Predictions"
2. 模型会自动分析图像并返回分割结果
3. 可以基于预测结果进行手动调整

## 🧪 测试验证

### 1. 运行测试脚本
```bash
# 运行完整测试
python test_predict_request.py

# 运行模型检查
python inspect_model.py

# 运行API测试
python test_api.py
```

### 2. 手动测试API
```bash
# 健康检查
curl -X GET http://localhost:9090/health

# 预测测试
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "data": {"image": "/path/to/your/image.jpg"}
    }],
    "label_config": "<View><Image name=\"image\" value=\"$image\"/></View>"
  }'
```

### 3. 预期输出格式
```json
{
  "results": [{
    "model_version": "1.0.0",
    "score": 0.726,
    "result": [{
      "from_name": "label",
      "to_name": "image",
      "type": "polygonlabels",
      "value": {
        "polygonlabels": ["arm_injection_area"],
        "points": [[x1,y1], [x2,y2], ...],
        "closed": true
      },
      "score": 0.726
    }]
  }]
}
```

## 🔍 故障排除

### 常见问题

#### 1. 模型文件未找到
```
Error: Model file not found at train6/weights/best.pt
```
**解决方案**: 确保模型文件存在于正确路径

#### 2. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 设置 `DEVICE=cpu`
- 减小 `IMAGE_SIZE`
- 减少 `MAX_DETECTIONS`

#### 3. Label Studio连接失败
```
401 Client Error: Unauthorized
```
**解决方案**:
- 检查 `LABEL_STUDIO_URL` 是否正确
- 更新 `LABEL_STUDIO_API_KEY`
- 确认Label Studio服务正在运行

#### 4. 端口被占用
```
Address already in use
```
**解决方案**:
```bash
# 查找占用端口的进程
lsof -i :9090

# 终止进程
kill -9 <PID>
```

### 日志分析
```bash
# 查看详细日志
export LOG_LEVEL="DEBUG"
python _wsgi.py --log-level DEBUG

# 常见日志信息
# ✅ 正常: "Model loaded successfully"
# ⚠️ 警告: "No image found in task"
# ❌ 错误: "Error loading model"
```

## 📡 API使用

### 端点说明

#### GET /health
检查服务状态
```bash
curl http://localhost:9090/health
```
**响应示例**:
```json
{
  "model_class": "YOLOInjectionAreaSegmentation",
  "status": "UP"
}
```

#### POST /predict
执行预测
```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "id": 1,
      "data": {"image": "/path/to/image.jpg"}
    }],
    "project": "3.1747795083",
    "label_config": "<View><Image name=\"image\" value=\"$image\"/><BrushLabels name=\"tag\" toName=\"image\"><Label value=\"injection_area_arm\" background=\"#FFA39E\"/></BrushLabels></View>",
    "params": {"context": null}
  }'
```

#### POST /webhook
处理训练事件
```bash
curl -X POST http://localhost:9090/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "START_TRAINING",
    "project": {"id": 3, "label_config": "..."}
  }'
```

### 支持的图像路径格式

1. **本地绝对路径**:
   ```json
   {"image": "/home/user/images/photo.jpg"}
   ```

2. **Label Studio本地文件**:
   ```json
   {"image": "/data/local-files/?d=folder/photo.jpg"}
   ```

3. **HTTP/HTTPS URL**:
   ```json
   {"image": "https://example.com/photo.jpg"}
   ```

### 性能优化建议

1. **GPU加速**: 确保CUDA环境正确配置
2. **批处理**: 一次处理多个图像
3. **图像预处理**: 优化图像尺寸和格式
4. **模型缓存**: 避免重复加载模型
5. **并发处理**: 使用多进程或异步处理

### 监控和日志

#### 日志级别设置
```bash
# 调试模式 - 详细日志
export LOG_LEVEL="DEBUG"

# 生产模式 - 基本日志
export LOG_LEVEL="INFO"

# 错误模式 - 仅错误日志
export LOG_LEVEL="ERROR"
```

#### 关键指标监控
- 模型加载时间
- 推理延迟
- 内存使用量
- GPU利用率
- 请求成功率

## 🔧 高级配置

### Docker部署
```dockerfile
# 使用提供的Dockerfile
docker build -t yolo-injection-segmentation .
docker run -p 9090:9090 -e LABEL_STUDIO_URL="http://host:8080" yolo-injection-segmentation
```

### 生产环境配置
```bash
# 使用uWSGI部署
pip install uwsgi
uwsgi --http :9090 --wsgi-file _wsgi.py --callable app --processes 4 --threads 2

# 使用Gunicorn部署
pip install gunicorn
gunicorn --bind 0.0.0.0:9090 --workers 4 _wsgi:app
```

### 负载均衡配置
```nginx
# Nginx配置示例
upstream ml_backend {
    server 127.0.0.1:9090;
    server 127.0.0.1:9091;
    server 127.0.0.1:9092;
}

server {
    listen 80;
    location / {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📞 技术支持

### 问题报告
如遇到问题，请提供以下信息：
- 错误日志 (设置LOG_LEVEL=DEBUG)
- 系统环境信息 (Python版本、CUDA版本等)
- 配置文件内容
- 复现步骤
- 输入图像示例

### 联系方式
- **项目仓库**: [GitHub链接]
- **技术文档**: [文档链接]
- **问题反馈**: [Issue链接]

### 更新日志
- **v1.0.0** (2025-05-22): 初始版本发布
  - 支持YOLO分割模型
  - Label Studio集成
  - 多种图像路径格式支持
  - 详细日志和错误处理

---

**版本**: v1.0.0
**更新日期**: 2025-05-22
**维护者**: AI Annotation Studio Team
