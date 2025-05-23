# 🚀 YOLO注射区域分割模型 - 快速启动指南

## ⚡ 5分钟快速启动

### 第一步：检查环境
```bash
# 检查Python版本 (需要3.8+)
python --version

# 检查CUDA (可选，用于GPU加速)
nvidia-smi

# 检查项目文件
ls -la train6/weights/best.pt
```

### 第二步：安装依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# 验证安装
python -c "import ultralytics, torch; print('✅ 依赖安装成功')"
```

### 第三步：配置环境
```bash
# 复制并编辑环境配置
cp .env.example .env

# 编辑配置文件
nano .env
```

**必需配置项**:
```bash
# Label Studio连接信息
LABEL_STUDIO_URL=http://192.168.1.124:8080/
LABEL_STUDIO_API_KEY=your_api_key_here

# 模型参数 (可选)
CONFIDENCE_THRESHOLD=0.25
LOG_LEVEL=INFO
```

### 第四步：启动服务
```bash
# 使用启动脚本 (推荐)
bash start_ml_backend.sh

# 或者直接启动
python _wsgi.py --port 9090 --host 0.0.0.0
```

### 第五步：验证服务
```bash
# 健康检查
curl http://localhost:9090/health

# 预期输出
{"model_class":"YOLOInjectionAreaSegmentation","status":"UP"}
```

## 🔧 Label Studio集成

### 1. 获取API密钥
1. 登录Label Studio: `http://192.168.1.124:8080`
2. 进入 **Account & Settings**
3. 复制 **Access Token**
4. 更新环境变量中的 `LABEL_STUDIO_API_KEY`

### 2. 添加ML后端
1. 进入项目设置
2. 点击 **Machine Learning** → **Add Model**
3. 填写信息：
   - **URL**: `http://localhost:9090`
   - **Title**: `YOLO注射区域分割`
4. 点击 **Validate and Save**

### 3. 配置标注界面
使用以下XML配置：
```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="injection_area_arm" background="#FFA39E"/>
  </BrushLabels>
</View>
```

## 🧪 快速测试

### 测试脚本
```bash
# 运行完整测试
python test_predict_request.py

# 测试单个图像
python -c "
from model import YOLOInjectionAreaSegmentation
model = YOLOInjectionAreaSegmentation()
print('✅ 模型加载成功')
"
```

### 手动测试API
```bash
# 测试预测接口
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "data": {"image": "/path/to/your/test/image.jpg"}
    }],
    "label_config": "<View><Image name=\"image\" value=\"$image\"/></View>"
  }'
```

## ⚠️ 常见问题

### 问题1: 模型文件未找到
```
FileNotFoundError: Model file not found at train6/weights/best.pt
```
**解决**: 确保模型文件在正确位置，或更新 `config.py` 中的路径

### 问题2: CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 设置环境变量 `DEVICE=cpu`

### 问题3: 端口被占用
```
Address already in use
```
**解决**: 
```bash
# 查找并终止占用进程
lsof -i :9090
kill -9 <PID>
```

### 问题4: Label Studio连接失败
```
401 Client Error: Unauthorized
```
**解决**: 
1. 检查 `LABEL_STUDIO_URL` 是否正确
2. 更新 `LABEL_STUDIO_API_KEY`
3. 确认Label Studio服务正在运行

## 📊 性能调优

### GPU加速设置
```bash
# 检查GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 强制使用GPU
export DEVICE=cuda

# 强制使用CPU
export DEVICE=cpu
```

### 内存优化
```bash
# 减少图像尺寸
export IMAGE_SIZE=416

# 减少最大检测数
export MAX_DETECTIONS=100

# 降低置信度阈值
export CONFIDENCE_THRESHOLD=0.1
```

## 🔍 调试模式

### 启用详细日志
```bash
# 设置调试级别
export LOG_LEVEL=DEBUG

# 重启服务
bash start_ml_backend.sh
```

### 查看日志输出
```bash
# 实时查看日志
tail -f /var/log/ml_backend.log

# 或者直接在终端查看
python _wsgi.py --log-level DEBUG
```

## 📱 监控面板

### 基本监控命令
```bash
# 检查服务状态
curl -s http://localhost:9090/health | jq

# 监控系统资源
htop

# 监控GPU使用
nvidia-smi -l 1
```

### 性能指标
- **启动时间**: 通常 < 30秒
- **推理延迟**: 通常 < 2秒/图像
- **内存使用**: 通常 < 4GB
- **GPU利用率**: 推理时 > 80%

## 🆘 紧急停止

```bash
# 停止服务
pkill -f "_wsgi.py"

# 或者使用Ctrl+C停止前台进程

# 清理端口
lsof -ti:9090 | xargs kill -9
```

## 📞 获取帮助

### 自助诊断
```bash
# 运行诊断脚本
python -c "
import sys, torch, ultralytics
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Ultralytics: {ultralytics.__version__}')
"
```

### 日志收集
```bash
# 收集系统信息
python _wsgi.py --check > system_info.txt 2>&1

# 收集错误日志
python _wsgi.py --log-level DEBUG > debug.log 2>&1
```

---

**🎯 目标**: 让你在5分钟内成功启动YOLO注射区域分割服务！

如果遇到问题，请参考详细的 [USAGE_GUIDE.md](./USAGE_GUIDE.md) 文档。
