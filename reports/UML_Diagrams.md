# Software Engineering UML Diagrams

## 1. System Architecture Diagram
```mermaid
graph TD
    UI[Streamlit app.py] --> Engine[Inference Engine inference.py]
    UI --> Metrics[Metrics Calculator metrics.py]
    UI --> Processor[Image Processor image_processing.py]
    
    Engine --> Factory[Model Factory models/loader.py]
    Factory --> M1[DehazeFormer]
    Factory --> M2[AOD-Net]
    Factory --> M3[Dark Channel Prior]
    
    UI --> Download[Download Manager download.py]
```

## 2. Process Flowchart
```mermaid
flowchart TD
    Start([Upload Hazy Image]) --> SelectModel[Select AI Model]
    SelectModel --> Preprocess[Preprocess Image & Pad]
    Preprocess --> ForwardPass[Execute Model Inference]
    ForwardPass --> Postprocess[Unpad & Apply Enhancements]
    Postprocess --> ComputeMetrics[Compute PSNR, SSIM, MSE]
    ComputeMetrics --> Display[Display Results & Download]
    Display --> End([End Process])
```

## 3. Use Case Diagram
```mermaid
usecaseDiagram
    actor User
    User --> (Upload Image)
    User --> (Select Dehazing Model)
    User --> (Run Dehazing Inference)
    User --> (Adjust CLAHE / Sharpen Sliders)
    User --> (Inspect Quality Metrics & Histograms)
    User --> (Download Enhanced Image & Report)
```

## 4. Sequence Diagram
```mermaid
sequenceDiagram
    participant User
    participant StreamlitUI as app.py
    participant Engine as inference.py
    participant Factory as models/loader.py
    participant Model as PyTorch Model
    
    User->>StreamlitUI: Upload Image & Click Dehaze
    StreamlitUI->>Engine: dehaze_image(img, model_name)
    Engine->>Factory: get_model(model_name)
    Factory-->>Engine: Return Model Instance
    Engine->>Model: Forward Pass(Tensor)
    Model-->>Engine: Output Tensor
    Engine-->>StreamlitUI: Return Enhanced Image & Telemetry
    StreamlitUI-->>User: Display Results & Metrics
```

## 5. Activity Diagram
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> ImageUploaded: User Uploads File
    ImageUploaded --> ModelLoading: User Clicks Run Dehazing
    ModelLoading --> ExecutingInference: Load Weights & Tensor
    ExecutingInference --> ComputingMetrics: Forward Pass Complete
    ComputingMetrics --> RenderDashboard: Calculate PSNR/SSIM
    RenderDashboard --> [*]: Export PNG/ZIP/Report
```

## 6. Class Diagram
```mermaid
classDiagram
    class DehazeInferenceEngine {
        +device: str
        +loaded_models: dict
        +load_model(model_name)
        +process_image(image_input)
        +process_batch(image_list)
    }
    class MetricsCalculator {
        +calculate_all_metrics(original_img, dehazed_img)
    }
    class ImageProcessor {
        +apply_clahe(img)
        +apply_unsharp_mask(img)
    }
    class DownloadManager {
        +save_image(image_np)
        +save_comparison(orig, dehazed)
        +create_zip(image_tuples)
    }
    DehazeInferenceEngine --> MetricsCalculator
    DehazeInferenceEngine --> ImageProcessor
```

## 7. Deployment Diagram
```mermaid
graph LR
    Client[Web Browser Client] -- HTTP / WebSocket --> Streamlit[Streamlit App Server app.py]
    Streamlit --> PyTorch[PyTorch Inference Runtime]
    PyTorch --> Hardware[Hardware Acceleration CUDA / CPU]
```
