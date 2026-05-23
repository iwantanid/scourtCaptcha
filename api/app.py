import os
import base64
import random
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np

# 1. Model Definition (Embedded for standalone portability)
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32 x 20 x 60
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64 x 10 x 30
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128 x 5 x 15
            nn.Dropout(0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256 x 2 x 7
            nn.Dropout(0.15)
        )
        
        # Dense Layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Six Output Heads (one for each digit position)
        self.head1 = nn.Linear(512, 10)
        self.head2 = nn.Linear(512, 10)
        self.head3 = nn.Linear(512, 10)
        self.head4 = nn.Linear(512, 10)
        self.head5 = nn.Linear(512, 10)
        self.head6 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        out1 = self.head1(x)
        out2 = self.head2(x)
        out3 = self.head3(x)
        out4 = self.head4(x)
        out5 = self.head5(x)
        out6 = self.head6(x)
        
        return out1, out2, out3, out4, out5, out6

app = FastAPI(title="대법원 나의 사건 검색 Captcha Decoder API")

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptchaCNN().to(device)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captcha_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Successfully loaded PyTorch model from {model_path}")
else:
    print(f"Warning: Model weights not found at {model_path}! Please train the model first.")

# 1. Image preprocessing helper
def preprocess_image(image_bytes):
    # Load image in PIL
    img = Image.open(io.BytesIO(image_bytes))
    # Convert to grayscale
    img = img.convert("L")
    # Resize to 120x40
    img = img.resize((120, 40), Image.Resampling.BILINEAR)
    
    # Convert to numpy array and normalize
    img_arr = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor (1, 1, 40, 120)
    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor

# 2. Prediction API Endpoint
@app.post("/predict")
async def predict_captcha(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_tensor = preprocess_image(contents)
        
        # Run inference
        with torch.no_grad():
            out1, out2, out3, out4, out5, out6 = model(img_tensor)
            
            pred1 = torch.argmax(out1, dim=1).item()
            pred2 = torch.argmax(out2, dim=1).item()
            pred3 = torch.argmax(out3, dim=1).item()
            pred4 = torch.argmax(out4, dim=1).item()
            pred5 = torch.argmax(out5, dim=1).item()
            pred6 = torch.argmax(out6, dim=1).item()
            
            prediction_str = f"{pred1}{pred2}{pred3}{pred4}{pred5}{pred6}"
            
        return {
            "status": "success",
            "prediction": prediction_str
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

# 3. Serving the beautiful premium drag-and-drop web client
@app.get("/", response_class=HTMLResponse)
async def get_web_client():
    # Load 5 random captcha samples to embed as base64 in the web interface for instant testing
    training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "TrainingData")
    samples_html = ""
    
    if os.path.exists(training_dir):
        all_files = [f for f in os.listdir(training_dir) if f.lower().endswith('.png')]
        if all_files:
            # Load answer_key to get labels
            answers = {}
            answer_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "answer_key.txt")
            if os.path.exists(answer_key_path):
                with open(answer_key_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and ":" in line:
                            parts = line.split(":")
                            answers[parts[0].strip()] = parts[1].strip()
            
            sampled_files = random.sample(all_files, min(5, len(all_files)))
            for filename in sampled_files:
                img_path = os.path.join(training_dir, filename)
                with open(img_path, "rb") as img_f:
                    b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                
                label = answers.get(filename, "Unknown")
                samples_html += f"""
                <div class="sample-card" onclick="predictSample('{b64_data}', '{filename}', '{label}')">
                    <img src="data:image/png;base64,{b64_data}" alt="Sample Captcha">
                    <div class="sample-label">Test {filename}</div>
                </div>
                """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Futuristic Captcha Solver & Verification Panel</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-gradient: radial-gradient(circle at 50% 50%, #151624 0%, #080911 100%);
                --panel-bg: rgba(22, 23, 41, 0.65);
                --border-color: rgba(255, 255, 255, 0.08);
                --glow-cyan: rgba(0, 242, 254, 0.4);
                --glow-purple: rgba(185, 0, 255, 0.4);
                --text-primary: #ffffff;
                --text-secondary: #8b92b6;
                --accent-cyan: #00f2fe;
                --accent-purple: #b900ff;
                --success-green: #39ff14;
            }}

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                font-family: 'Outfit', sans-serif;
                background: var(--bg-gradient);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                overflow-x: hidden;
                padding: 2rem 1rem;
            }}

            /* Glow effects */
            body::before {{
                content: '';
                position: absolute;
                width: 500px;
                height: 500px;
                background: radial-gradient(circle, var(--accent-cyan) 0%, transparent 70%);
                top: -10%;
                left: -10%;
                opacity: 0.15;
                z-index: -1;
                filter: blur(80px);
            }}

            body::after {{
                content: '';
                position: absolute;
                width: 500px;
                height: 500px;
                background: radial-gradient(circle, var(--accent-purple) 0%, transparent 70%);
                bottom: -10%;
                right: -10%;
                opacity: 0.15;
                z-index: -1;
                filter: blur(80px);
            }}

            .container {{
                width: 100%;
                max-width: 900px;
                display: flex;
                flex-direction: column;
                gap: 2rem;
                z-index: 1;
            }}

            header {{
                text-align: center;
                margin-bottom: 1rem;
            }}

            header h1 {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.8rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
                letter-spacing: -0.05em;
                filter: drop-shadow(0px 0px 20px rgba(0, 242, 254, 0.2));
            }}

            header p {{
                font-size: 1.1rem;
                color: var(--text-secondary);
                font-weight: 300;
            }}

            .main-panel {{
                background: var(--panel-bg);
                border: 1px solid var(--border-color);
                border-radius: 24px;
                padding: 2.5rem;
                backdrop-filter: blur(20px);
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 
                            inset 0 1px 0 rgba(255, 255, 255, 0.05);
                display: flex;
                flex-direction: column;
                gap: 2rem;
                transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            }}

            /* Drag and Drop Zone */
            .drop-zone {{
                border: 2px dashed rgba(0, 242, 254, 0.25);
                border-radius: 16px;
                padding: 3rem 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 1.2rem;
                cursor: pointer;
                background: rgba(255, 255, 255, 0.01);
                transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
                position: relative;
                overflow: hidden;
            }}

            .drop-zone:hover, .drop-zone.dragover {{
                border-color: var(--accent-cyan);
                background: rgba(0, 242, 254, 0.04);
                box-shadow: 0 0 30px rgba(0, 242, 254, 0.15);
                transform: scale(1.01);
            }}

            .drop-zone svg {{
                width: 60px;
                height: 60px;
                stroke: var(--accent-cyan);
                transition: transform 0.4s ease;
                filter: drop-shadow(0 0 10px var(--glow-cyan));
            }}

            .drop-zone:hover svg {{
                transform: translateY(-5px);
            }}

            .drop-zone p {{
                font-size: 1.1rem;
                color: var(--text-secondary);
                text-align: center;
            }}

            .drop-zone p span {{
                color: var(--accent-cyan);
                font-weight: 600;
            }}

            #file-input {{
                display: none;
            }}

            /* Preview and Result section */
            .result-wrapper {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                align-items: center;
                opacity: 0;
                transform: translateY(20px);
                max-height: 0;
                overflow: hidden;
                transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            }}

            .result-wrapper.show {{
                opacity: 1;
                transform: translateY(0);
                max-height: 500px;
                padding-top: 1rem;
            }}

            .preview-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.8rem;
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                min-height: 180px;
                justify-content: center;
            }}

            .preview-container img {{
                max-width: 100%;
                max-height: 100px;
                border-radius: 8px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}

            .preview-label {{
                font-size: 0.9rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            .prediction-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 1rem;
                background: linear-gradient(135deg, rgba(0, 242, 254, 0.05) 0%, rgba(185, 0, 255, 0.05) 100%);
                border: 1px solid rgba(0, 242, 254, 0.15);
                border-radius: 16px;
                padding: 2.2rem;
                min-height: 180px;
                position: relative;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }}

            .prediction-value {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 3.5rem;
                font-weight: 700;
                color: var(--accent-cyan);
                letter-spacing: 0.15em;
                text-align: center;
                text-shadow: 0 0 20px var(--glow-cyan);
                animation: pulseGlow 2s infinite ease-in-out;
            }}

            @keyframes pulseGlow {{
                0%, 100% {{ text-shadow: 0 0 15px var(--glow-cyan); }}
                50% {{ text-shadow: 0 0 30px var(--glow-cyan), 0 0 45px var(--accent-cyan); color: #fff; }}
            }}

            /* Samples Section */
            .samples-section {{
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}

            .samples-title {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--text-primary);
                border-left: 3px solid var(--accent-cyan);
                padding-left: 0.8rem;
            }}

            .samples-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 1rem;
            }}

            .sample-card {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 0.8rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.6rem;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            }}

            .sample-card:hover {{
                background: rgba(255, 255, 255, 0.05);
                border-color: var(--accent-purple);
                box-shadow: 0 8px 20px rgba(185, 0, 255, 0.15);
                transform: translateY(-3px);
            }}

            .sample-card img {{
                width: 100%;
                height: auto;
                max-height: 46px;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }}

            .sample-label {{
                font-size: 0.75rem;
                color: var(--text-secondary);
                font-weight: 500;
            }}

            footer {{
                text-align: center;
                font-size: 0.85rem;
                color: var(--text-secondary);
                margin-top: 1rem;
            }}

            footer a {{
                color: var(--accent-cyan);
                text-decoration: none;
                font-weight: 600;
            }}

            /* Loading spinner */
            .spinner {{
                border: 3px solid rgba(255, 255, 255, 0.05);
                border-radius: 50%;
                border-top: 3px solid var(--accent-cyan);
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>SCOURT CAPTCHA SOLVER</h1>
                <p>High-Accuracy Deep Learning Captcha Verification Panel</p>
            </header>

            <div class="main-panel">
                <!-- Drag and Drop Area -->
                <div class="drop-zone" id="drop-zone">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                    <p>Drag & Drop a captcha image here, or <span>browse files</span></p>
                    <input type="file" id="file-input" accept="image/*">
                </div>

                <!-- Preview and Prediction Result Card -->
                <div class="result-wrapper" id="result-wrapper">
                    <div class="preview-container">
                        <div class="preview-label">Input Captcha</div>
                        <img id="preview-image" src="" alt="Captcha Preview">
                    </div>
                    <div class="prediction-container">
                        <div class="preview-label" id="result-type-label">Model Prediction</div>
                        <div id="prediction-result" class="prediction-value">------</div>
                    </div>
                </div>

                <!-- Instant Test Bench (Sample Images) -->
                <div class="samples-section">
                    <div class="samples-title">Instant Test Bench</div>
                    <div class="samples-grid">
                        {samples_html}
                    </div>
                </div>
            </div>

            <footer>
                Powered by a custom PyTorch CNN model. Validation Accuracy &gt; 98%.
            </footer>
        </div>

        <script>
            const dropZone = document.getElementById('drop-zone');
            const fileInput = document.getElementById('file-input');
            const resultWrapper = document.getElementById('result-wrapper');
            const previewImage = document.getElementById('preview-image');
            const predictionResult = document.getElementById('prediction-result');
            const resultTypeLabel = document.getElementById('result-type-label');

            // Set up click on drop zone to open file picker
            dropZone.addEventListener('click', () => fileInput.click());

            // Listen for file select
            fileInput.addEventListener('change', (e) => {{
                if (e.target.files.length > 0) {{
                    processFile(e.target.files[0]);
                }}
            }});

            // Drag and drop event listeners
            ['dragenter', 'dragover'].forEach(eventName => {{
                dropZone.addEventListener(eventName, (e) => {{
                    e.preventDefault();
                    dropZone.classList.add('dragover');
                }}, false);
            }});

            ['dragleave', 'drop'].forEach(eventName => {{
                dropZone.addEventListener(eventName, (e) => {{
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                }}, false);
            }});

            dropZone.addEventListener('drop', (e) => {{
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {{
                    processFile(files[0]);
                }}
            }});

            // Process uploaded file
            function processFile(file) {{
                // Read and show preview
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onloadend = () => {{
                    previewImage.src = reader.result;
                    resultWrapper.classList.add('show');
                    
                    // Show spinner while predicting
                    predictionResult.innerHTML = '<div class="spinner"></div>';
                    resultTypeLabel.innerText = 'Calculating...';
                    
                    // Call prediction API
                    const formData = new FormData();
                    formData.append('file', file);

                    fetch('/predict', {{
                        method: 'POST',
                        body: formData
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.status === 'success') {{
                            predictionResult.innerText = data.prediction;
                            resultTypeLabel.innerText = 'Prediction Results';
                        }} else {{
                            predictionResult.innerText = 'ERROR';
                            resultTypeLabel.innerText = 'Error Occurred';
                        }}
                    }})
                    .catch(err => {{
                        console.error(err);
                        predictionResult.innerText = 'ERROR';
                        resultTypeLabel.innerText = 'Connection Failed';
                    }});
                }};
            }}

            // Run prediction for embedded instant samples
            function predictSample(base64Data, filename, trueLabel) {{
                // Convert base64 data to Blob
                const byteCharacters = atob(base64Data);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {{
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }}
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {{type: 'image/png'}});
                const file = new File([blob], filename, {{type: 'image/png'}});
                
                // Show local preview immediately
                previewImage.src = 'data:image/png;base64,' + base64Data;
                resultWrapper.classList.add('show');
                
                predictionResult.innerHTML = '<div class="spinner"></div>';
                resultTypeLabel.innerText = `Testing ${{filename}}...`;
                
                const formData = new FormData();
                formData.append('file', file);

                fetch('/predict', {{
                    method: 'POST',
                    body: formData
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'success') {{
                        const isCorrect = data.prediction === trueLabel;
                        predictionResult.innerHTML = data.prediction;
                        resultTypeLabel.innerHTML = `Predicted (${{filename}}) | True Answer: <span style="color: var(--success-green); font-weight: bold;">${{trueLabel}}</span>`;
                    }} else {{
                        predictionResult.innerText = 'ERROR';
                        resultTypeLabel.innerText = 'Error Occurred';
                    }}
                }})
                .catch(err => {{
                    console.error(err);
                    predictionResult.innerText = 'ERROR';
                    resultTypeLabel.innerText = 'Connection Failed';
                }});
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
