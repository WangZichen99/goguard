// main.go (Final Version using AdvancedSession API)
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	_ "golang.org/x/image/webp"
	_ "github.com/Kagami/go-avif"
	_ "github.com/klippa-app/go-libheif"
	"io"
	"log"
	"os"
	"runtime"
	"strings"

	"github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// ... 通信数据结构和函数保持不变 ...
type MessageEnvelope struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}
type ProcessImagePayload struct {
	ImageID   string `json:"imageId"`
	ImageData string `json:"imageData"`
}
type ImageResultPayload struct {
	ImageID    string `json:"imageId"`
	ShouldHide bool   `json:"shouldHide"`
}

func readMessage() ([]byte, error) {
	var length uint32
	err := binary.Read(os.Stdin, binary.LittleEndian, &length)
	if err != nil {
		return nil, err
	}
	message := make([]byte, length)
	_, err = io.ReadFull(os.Stdin, message)
	return message, err
}
func sendMessage(messageData interface{}) error {
	messageBytes, err := json.Marshal(messageData)
	if err != nil {
		return err
	}
	err = binary.Write(os.Stdout, binary.LittleEndian, uint32(len(messageBytes)))
	if err != nil {
		return err
	}
	_, err = os.Stdout.Write(messageBytes)
	return err
}

// -----------------------------------------------------------------------------
// 3. AI模型处理核心逻辑 (已使用 AdvancedSession 修复)
// -----------------------------------------------------------------------------

// --- 修复 1: 使用 AdvancedSession 来创建可重用的会话 ---
type ONNXModel struct {
	session      *onnxruntime_go.DynamicAdvancedSession
	inputShape   onnxruntime_go.Shape
	outputShape  onnxruntime_go.Shape
	inputWidth   int
	inputHeight  int
	channelCount int
	channelsLast bool
	inputNames   []string
	outputNames  []string
}

func NewONNXModel(modelPath string) (*ONNXModel, error) {
	options, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}

	inputInfos, outputInfos, err := onnxruntime_go.GetInputOutputInfoWithOptions(modelPath, options)
	if err != nil {
		_ = options.Destroy()
		return nil, fmt.Errorf("failed to read model io info: %w", err)
	}
	if len(inputInfos) == 0 {
		_ = options.Destroy()
		return nil, fmt.Errorf("model %s does not expose any inputs", modelPath)
	}
	if len(outputInfos) == 0 {
		_ = options.Destroy()
		return nil, fmt.Errorf("model %s does not expose any outputs", modelPath)
	}
	if len(inputInfos) != 1 {
		_ = options.Destroy()
		return nil, fmt.Errorf("model %s expects %d inputs, 当前代码仅支持 1 个输入", modelPath, len(inputInfos))
	}
	if len(outputInfos) != 1 {
		_ = options.Destroy()
		return nil, fmt.Errorf("model %s expects %d outputs, 当前代码仅支持 1 个输出", modelPath, len(outputInfos))
	}

	inputNames := make([]string, len(inputInfos))
	for i, info := range inputInfos {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(outputInfos))
	for i, info := range outputInfos {
		outputNames[i] = info.Name
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, options)
	if err != nil {
		_ = options.Destroy()
		return nil, fmt.Errorf("failed to create ONNX dynamic session: %w", err)
	}
	if err := options.Destroy(); err != nil {
		log.Printf("warning: failed to destroy session options: %v", err)
	}

	model := &ONNXModel{
		session:     session,
		inputShape:  append(onnxruntime_go.Shape{}, inputInfos[0].Dimensions...),
		outputShape: append(onnxruntime_go.Shape{}, outputInfos[0].Dimensions...),
		inputNames:  inputNames,
		outputNames: outputNames,
	}
	model.detectLayout()
	model.sanitizeOutputShape()

	log.Printf("ONNX DynamicAdvancedSession created: input=%v output=%v channelsLast=%v",
		model.inputShape, model.outputShape, model.channelsLast)
	return model, nil
}

func (m *ONNXModel) detectLayout() {
	shape := append(onnxruntime_go.Shape{}, m.inputShape...)
	if len(shape) >= 4 {
		if shape[3] == 3 || shape[3] <= 0 {
			if shape[0] <= 0 {
				shape[0] = 1
			}
			if shape[1] <= 0 {
				shape[1] = 224
			}
			if shape[2] <= 0 {
				shape[2] = 224
			}
			if shape[3] <= 0 {
				shape[3] = 3
			}
			m.inputHeight = int(shape[1])
			m.inputWidth = int(shape[2])
			m.channelCount = int(shape[3])
			m.channelsLast = true
			m.inputShape = shape
			return
		}
		if shape[1] == 3 || shape[1] <= 0 {
			if shape[0] <= 0 {
				shape[0] = 1
			}
			if shape[2] <= 0 {
				shape[2] = 224
			}
			if shape[3] <= 0 {
				shape[3] = 224
			}
			if shape[1] <= 0 {
				shape[1] = 3
			}
			m.inputHeight = int(shape[2])
			m.inputWidth = int(shape[3])
			m.channelCount = int(shape[1])
			m.channelsLast = false
			m.inputShape = shape
			return
		}
	}

	log.Printf("fallback to NHWC 1x224x224x3 for input shape %v", shape)
	m.inputHeight = 224
	m.inputWidth = 224
	m.channelCount = 3
	m.channelsLast = true
	m.inputShape = onnxruntime_go.NewShape(1, int64(m.inputHeight), int64(m.inputWidth), int64(m.channelCount))
}

func (m *ONNXModel) sanitizeOutputShape() {
	if len(m.outputShape) == 0 {
		return
	}
	clean := append(onnxruntime_go.Shape{}, m.outputShape...)
	for i := range clean {
		if clean[i] <= 0 {
			clean[i] = 1
		}
	}
	m.outputShape = clean
}

func (m *ONNXModel) Destroy() {
	m.session.Destroy()
}

func (m *ONNXModel) processImageWithONNX(imageDataB64 string) (bool, error) {
	// ... 图片解码、缩放、张量填充逻辑保持不变 ...
	coI := strings.Index(imageDataB64, ",")
	if coI != -1 {
		imageDataB64 = imageDataB64[coI+1:]
	}
	rawImage, err := base64.StdEncoding.DecodeString(imageDataB64)
	if err != nil {
		return false, fmt.Errorf("error decoding base64: %w", err)
	}

	img, _, err := image.Decode(bytes.NewReader(rawImage))
	if err != nil {
		return false, fmt.Errorf("error decoding image format: %w", err)
	}

	resizedImg := image.NewRGBA(image.Rect(0, 0, m.inputWidth, m.inputHeight))
	draw.BiLinear.Scale(resizedImg, resizedImg.Bounds(), img, img.Bounds(), draw.Over, nil)

	if m.channelCount <= 0 {
		return false, fmt.Errorf("invalid channel count: %d", m.channelCount)
	}
	inputTensor := make([]float32, m.inputHeight*m.inputWidth*m.channelCount)
	pixels := resizedImg.Pix
	channelArea := m.inputHeight * m.inputWidth
	for y := 0; y < m.inputHeight; y++ {
		for x := 0; x < m.inputWidth; x++ {
			srcOffset := (y*resizedImg.Stride + x*4)
			r := float32(pixels[srcOffset]) / 255.0
			g := float32(pixels[srcOffset+1]) / 255.0
			b := float32(pixels[srcOffset+2]) / 255.0
			if m.channelsLast {
				destOffset := (y*m.inputWidth + x) * m.channelCount
				if m.channelCount > 0 {
					inputTensor[destOffset] = r
				}
				if m.channelCount > 1 {
					inputTensor[destOffset+1] = g
				}
				if m.channelCount > 2 {
					inputTensor[destOffset+2] = b
				}
			} else {
				spatialIndex := y*m.inputWidth + x
				if m.channelCount > 0 {
					inputTensor[spatialIndex] = r
				}
				if m.channelCount > 1 {
					inputTensor[channelArea+spatialIndex] = g
				}
				if m.channelCount > 2 {
					inputTensor[2*channelArea+spatialIndex] = b
				}
			}
		}
	}

	// 创建输入和输出张量
	inputOrtValue, err := onnxruntime_go.NewTensor[float32](m.inputShape, inputTensor)
	if err != nil {
		return false, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputOrtValue.Destroy()

	outputOrtValue, err := onnxruntime_go.NewEmptyTensor[float32](m.outputShape)
	if err != nil {
		return false, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputOrtValue.Destroy()

	// --- 修复 2: 调用 AdvancedSession 的 Run 方法，它接收输入和输出参数 ---
	// 创建输入和输出的切片
	inputs := []onnxruntime_go.Value{inputOrtValue}
	outputs := []onnxruntime_go.Value{outputOrtValue}
	err = m.session.Run(inputs, outputs)
	if err != nil {
		return false, fmt.Errorf("failed to run model inference: %w", err)
	}

	// ... 解析结果逻辑保持不变 ...
	outputData := outputOrtValue.GetData()
	threshold := float32(0.3)
	isIllegal := outputData[1] > threshold || outputData[3] > threshold || outputData[4] > threshold
	log.Printf("Inference output: [%.4f, %.4f, %.4f, %.4f, %.4f], IsIllegal: %v",
		outputData[0], outputData[1], outputData[2], outputData[3], outputData[4], isIllegal)
	return isIllegal, nil
}

// -----------------------------------------------------------------------------
// 4. 主程序 (保持不变)
// -----------------------------------------------------------------------------
func main() {
	logFile, err := os.OpenFile("service.log", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		panic(err)
	}
	log.SetOutput(io.MultiWriter(os.Stderr, logFile))
	defer logFile.Close()
	log.Println("Go service starting...")
	
	// 跨平台ONNX Runtime库路径配置
	var onnxSharedLibraryPath string
	if runtime.GOOS == "windows" {
		onnxSharedLibraryPath = `E:\Programs\onnxruntime-win-x64-1.23.1\lib\onnxruntime.dll`
	} else {
		// Linux环境下的ONNX Runtime库路径
		onnxSharedLibraryPath = "/usr/local/lib/libonnxruntime.so"
	}
	
	// 检查环境变量是否设置了ONNX Runtime库路径
	if envPath := os.Getenv("ONNX_RUNTIME_LIB_PATH"); envPath != "" {
		onnxSharedLibraryPath = envPath
	}
	
	if _, err := os.Stat(onnxSharedLibraryPath); os.IsNotExist(err) {
		log.Fatalf("ONNX Runtime library not found at: %s", onnxSharedLibraryPath)
	}
	onnxruntime_go.SetSharedLibraryPath(onnxSharedLibraryPath)
	err = onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime environment: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()
	log.Println("ONNX Runtime environment initialized.")
	modelPath := "mobilenet_v2.onnx"
	model, err := NewONNXModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}
	defer model.Destroy()
	for {
		msgBytes, err := readMessage()
		if err != nil {
			if err == io.EOF {
				log.Println("Browser closed. Exiting.")
				break
			}
			log.Printf("Error reading message: %v", err)
			continue
		}
		var envelope MessageEnvelope
		if err := json.Unmarshal(msgBytes, &envelope); err != nil {
			log.Printf("Error unmarshaling envelope: %v", err)
			continue
		}
		if envelope.Type == "PROCESS_IMAGE" {
			var payload ProcessImagePayload
			if err := json.Unmarshal(envelope.Payload, &payload); err != nil {
				log.Printf("Error unmarshaling payload: %v", err)
				continue
			}
			log.Printf("Processing image: %s", payload.ImageID)
			shouldHide, err := model.processImageWithONNX(payload.ImageData)
			if err != nil {
				log.Printf("Error processing image %s with ONNX: %v", payload.ImageID, err)
				shouldHide = false
			}
			log.Printf("Decision for %s: shouldHide = %v", payload.ImageID, shouldHide)
			responsePayload := ImageResultPayload{ImageID: payload.ImageID, ShouldHide: shouldHide}
			responseEnvelope := map[string]interface{}{"type": "IMAGE_RESULT", "payload": responsePayload}
			if err := sendMessage(responseEnvelope); err != nil {
				log.Printf("Error sending response: %v", err)
			}
		}
	}
}
