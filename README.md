# GoGuard - Linux版本

这是WebGuard服务的Linux适配版本，用于与浏览器扩展通信，利用ONNX模型对图像内容进行审核。

## 环境准备

1. **安装Go**：确保已安装Go 1.25.2或更高版本。
2. **安装ONNX Runtime**：
   ```bash
   # 下载ONNX Runtime for Linux
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz
   tar -xzf onnxruntime-linux-x64-1.23.1.tgz
   sudo cp onnxruntime-linux-x64-1.23.1/lib/* /usr/local/lib/
   sudo ldconfig
   ```

## 构建

```bash
go build -o goguard src/main.go
```

## 运行

```bash
./goguard
```

服务将通过stdio与浏览器扩展通信。

## 配置

可以通过设置环境变量来指定ONNX Runtime库路径：
```bash
export ONNX_RUNTIME_LIB_PATH=/path/to/your/onnxruntime/lib/libonnxruntime.so
```