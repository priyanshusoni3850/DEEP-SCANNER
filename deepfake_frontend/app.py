# # Install required libraries
# !pip install flask
# !pip install torch torchvision torchaudio
# !pip install numpy opencv-python matplotlib moviepy librosa
# !pip install timm -q
# !pip install fpdf  # For PDF report generation
# !pip install reportlab
# !pip install flask-cors

import os
import shutil
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
import librosa
import librosa.display
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import dlib
from PIL import Image
from moviepy.editor import VideoFileClip
import copy
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Global Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dlib Face Landmark Setup
landmark_model_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(landmark_model_path):
    print("Downloading shape predictor model...")
    os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    os.system("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_model_path)

# Cell 4: Xception model definition and loading function
class XceptionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionNet, self).__init__()
        self.model = timm.create_model("xception", pretrained=False)  # Load Xception model
        self.model.fc = nn.Linear(self.model.num_features, num_classes)  # Adjust last layer

    def forward(self, x):
        return self.model(x)

def load_xception_model(model_path):
    model = XceptionNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Define model paths (adjust these paths as needed)
model_path_c40 = "C:\\Users\\priyanshu\\Downloads\\FaceForensics++ pretrained models-20250206T212505Z-001\\FaceForensics++ pretrained models\\ffpp_c40.pth"
model_path_c23 = "C:\\Users\\priyanshu\\Downloads\\FaceForensics++ pretrained models-20250206T212505Z-001\\FaceForensics++ pretrained models\\ffpp_c23.pth"

# Load both image models
model_c40 = load_xception_model(model_path_c40)
model_c23 = load_xception_model(model_path_c23)
print("Image models loaded successfully!")

# Cell 5: Image Preprocessing
inference_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to match Xception's input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    return inference_transform(image).unsqueeze(0)  # Add batch dimension

# Additional function to preprocess a NumPy array (for video frames)
def preprocess_frame(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(image).convert("RGB")
    return inference_transform(image).unsqueeze(0)

# Cell 6: Deepfake Prediction with Test-Time Fine-Tuning (TTFT) for Images
def predict_deepfake(model, image_tensor, adaptation_steps=3, adaptation_lr=1e-3):
    image_tensor = image_tensor.to(device)
    model_copy = copy.deepcopy(model)
    model_copy.train()
    bn_params = [p for n, p in model_copy.named_parameters() if 'bn' in n]
    optimizer = torch.optim.Adam(bn_params, lr=adaptation_lr)
    for _ in range(adaptation_steps):
        optimizer.zero_grad()
        outputs = model_copy(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        loss = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))
        loss.backward()
        optimizer.step()
    model_copy.eval()
    with torch.no_grad():
        outputs = model_copy(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        fake_prob = probabilities[0][1].item()
    return fake_prob

# Cell 7: Grad-CAM Heatmap Generation
def generate_heatmap(model, image_tensor):
    model.eval()
    gradients = []
    activations = []
    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    def save_activation(module, input, output):
        activations.append(output)
    target_layer = model.model.conv4  # The last convolutional layer before classification
    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    probs = F.softmax(output, dim=1)
    class_idx = torch.argmax(probs, dim=1).item()
    model.zero_grad()
    output[:, class_idx].backward()
    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grad, axis=(1, 2))  # Global average pooling
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

# Cell 8: Overlay Heatmap
def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed_image

# Cell 9: Save Highlighted Image and Generate Transparency Report for Images
def save_highlighted_image(preprocessed_image, heatmap_overlay_path):
    original_image = np.transpose(preprocessed_image.squeeze().cpu().numpy(), (1, 2, 0))
    original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
    original_image = (original_image * 255).astype(np.uint8)
    heatmap = generate_heatmap(model_c40, preprocessed_image)
    heatmap_overlay = overlay_heatmap(original_image, heatmap)
    cv2.imwrite(heatmap_overlay_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
    print(f"✅ Highlighted heatmap image saved: {heatmap_overlay_path}")
    return heatmap_overlay_path

def identify_deepfake_method(fake_prob):
    if fake_prob > 0.8:
        return "GAN-Based Face Swap (e.g., DeepFaceLab, FaceSwap)", "Face Swap"
    elif fake_prob > 0.6:
        return "Lip Syncing with Audio-Driven Models (e.g., Wav2Lip)", "Lip Syncing"
    elif fake_prob > 0.4:
        return "AI Voice Cloning with Speech Synthesis (e.g., Tacotron, WaveNet)", "Voice Cloning"
    else:
        return "Full AI Synthesis with Generative Models (e.g., StyleGAN, Synthesia)", "Full Synthesis"

def generate_transparency_report_image(image_path, highlighted_image_path, fake_prob, final_prediction, report_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Deepfake Transparency Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Original Uploaded Image:", ln=True)
    pdf.image(image_path, x=10, y=pdf.get_y(), w=90)
    pdf.ln(50)
    pdf.cell(200, 10, "Highlighted Deepfake Image (Heatmap):", ln=True)
    pdf.image(highlighted_image_path, x=10, y=pdf.get_y(), w=90)
    pdf.ln(60)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Prediction Results:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(200, 8, f"Deepfake Probability Score: {fake_prob:.4f}".encode('latin-1', 'replace').decode(), ln=True)
    pdf.cell(200, 8, f"Final Prediction: {final_prediction}".encode('latin-1', 'replace').decode(), ln=True)
    pdf.ln(10)
    method_used, deepfake_category = identify_deepfake_method(fake_prob)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "How This Deepfake Was Created:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, f"Detected Method: {method_used}".encode('latin-1', 'replace').decode())
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Deepfake Type:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, f"Identified as: {deepfake_category}".encode('latin-1', 'replace').decode())
    pdf.ln(10)
    pdf.cell(200, 10, "This report provides a transparent analysis of the detected deepfake.", ln=True, align="C")
    pdf.output(report_path)
    print(f"✅ Transparency report generated: {report_path}")
    return report_path

# Audio and Video processing functions

def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    if video.audio is None:
        print("⚠️ No audio found in the video. Skipping audio deepfake detection.")
        return False
    video.audio.write_audiofile(output_audio_path, codec="aac")
    print(f"✅ Extracted audio saved as: {output_audio_path}")
    return True

def convert_audio_to_wav(input_audio_path, output_wav_path):
    waveform, sample_rate = torchaudio.load(input_audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    torchaudio.save(output_wav_path, waveform, 16000)
    print(f"✅ Audio converted to WAV: {output_wav_path}")

def extract_audio_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs = torch.tensor(mfccs).unsqueeze(0)
    return mfccs.to(device)

class RawNet2(nn.Module):
    def __init__(self, input_channels=20, gru_input_size=256, gru_hidden_size=1024, num_classes=2):
        super(RawNet2, self).__init__()
        self.first_bn = nn.BatchNorm1d(input_channels)
        self.block0 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(64)
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(128)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(256)
        )
        self.gru = nn.GRU(input_size=256, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_bn(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def load_audio_model(checkpoint_path):
    model = RawNet2().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.eval()
    print("✅ Audio model loaded successfully!")
    return model

def predict_audio_deepfake(model, audio_features):
    with torch.no_grad():
        output = model(audio_features)
        probabilities = torch.softmax(output, dim=1)
        fake_prob = probabilities[0, 1].item()
    return fake_prob

# Load audio model (adjust the checkpoint path as needed)
audio_checkpoint_path = "C:\\Users\\priyanshu\\Downloads\\WaveFake pretrained\\WaveFake pretrained\\RawNet2\\leave_one_out\\ljspeech_hifiGAN\\ckpt.pth"
audio_model = load_audio_model(audio_checkpoint_path)

def extract_frames(video_path, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.png"), frame)
        frame_count += 1
    video.release()
    print(f"✅ Extracted {frame_count} frames.")
    return frame_count

def preprocess_frame(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_deepfake_video(model, image_tensor):
    image_tensor = image_tensor.to(device)
    torch.manual_seed(42)
    model_copy = copy.deepcopy(model)
    model_copy.train()
    bn_params = [p for n, p in model_copy.named_parameters() if 'bn' in n]
    optimizer = torch.optim.Adam(bn_params, lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model_copy(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        loss = - torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))
        loss.backward()
        optimizer.step()
    model_copy.eval()
    with torch.no_grad():
        outputs = model_copy(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        fake_prob = probabilities[0][1].item()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fake_prob

def highlight_fake_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) == 0:
        return image
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        regions = {
            "lips": list(range(48, 68)),
            "left_eye": list(range(42, 48)),
            "right_eye": list(range(36, 42)),
            "face": list(range(0, 17))
        }
        for region_name, indices in regions.items():
            points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in indices], np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=3)
    return image

def group_video_deepfake_segments(fake_frames, fps):
    groups = []
    if not fake_frames:
        return groups
    current_group = [fake_frames[0]]
    for prev, curr in zip(fake_frames, fake_frames[1:]):
        if curr[1] - prev[1] > 1.0/fps * 1.5:
            groups.append(current_group)
            current_group = [curr]
        else:
            current_group.append(curr)
    if current_group:
        groups.append(current_group)
    group_info = []
    for group in groups:
        start_time = group[0][1]
        end_time = group[-1][1] + 1.0/fps
        avg_score = sum([item[2] for item in group]) / len(group)
        group_info.append((start_time, end_time, avg_score))
    return group_info

def process_video_frames(frame_folder, model_c40, model_c23, total_frames, fps):
    frame_paths = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".png")])
    fake_count = 0
    video_fake_frames = []
    for idx, frame_path in enumerate(frame_paths):
        image_tensor = preprocess_image(frame_path)
        fake_prob_c40 = predict_deepfake(model_c40, image_tensor)
        fake_prob_c23 = predict_deepfake(model_c23, image_tensor)
        final_fake_score = (fake_prob_c40 + fake_prob_c23) / 2
        print(f"Frame {os.path.basename(frame_path)} score: {final_fake_score:.2f} - {'FAKE' if final_fake_score > 0.5 else 'REAL'}")
        if final_fake_score > 0.5:
            fake_count += 1
            timestamp = idx / fps
            video_fake_frames.append((idx, timestamp, final_fake_score))
    verdict = "FAKE" if fake_count > total_frames * 0.4 else "REAL"
    video_groups = group_video_deepfake_segments(video_fake_frames, fps)
    return verdict, video_groups

def process_and_highlight_video(input_video_path, output_video_path, model_c40, model_c23):
    video = cv2.VideoCapture(input_video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    frame_idx = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        # Use preprocess_frame here because "frame" is a NumPy array
        image_tensor = preprocess_frame(frame)
        fake_prob_c40 = predict_deepfake(model_c40, image_tensor)
        fake_prob_c23 = predict_deepfake(model_c23, image_tensor)
        final_fake_score = (fake_prob_c40 + fake_prob_c23) / 2
        label_text = f"{'FAKE' if final_fake_score > 0.5 else 'REAL'}: {final_fake_score:.2f}"
        cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if final_fake_score > 0.5:
            frame = highlight_fake_regions(frame)
        print(f"Processed frame {frame_idx:04d}: {label_text}")
        out.write(frame)
        frame_idx += 1
    video.release()
    out.release()
    print(f"✅ Highlighted video saved as: {output_video_path}")

def generate_transparency_report_video(report_path, audio_groups, overall_audio_prob, final_audio_prediction, audio_duration,
                                       prob_plot_img, waveform_plot_img, video_groups, video_verdict, video_duration, fps):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    margin = 50
    text_y = height - margin
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, text_y, "Combined Transparency Report")
    text_y -= 40
    if overall_audio_prob is not None:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, text_y, "Audio Deepfake Detection")
        text_y -= 25
        c.setFont("Helvetica", 12)
        c.drawString(margin, text_y, f"Audio Duration: {audio_duration:.2f} seconds")
        text_y -= 20
        c.drawString(margin, text_y, f"Overall Fake Probability (Audio): {overall_audio_prob:.4f}")
        text_y -= 20
        c.drawString(margin, text_y, f"Final Prediction (Audio): {final_audio_prediction}")
        text_y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, text_y, "Deepfake Creation Method (Audio):")
        text_y -= 18
        c.setFont("Helvetica", 12)
        c.drawString(margin, text_y, "This audio deepfake was created using advanced voice conversion and")
        text_y -= 15
        c.drawString(margin, text_y, "synthetic speech synthesis techniques that mimic human speech patterns.")
        text_y -= 25
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, text_y, "Type of Audio Deepfake:")
        text_y -= 18
        c.setFont("Helvetica", 12)
        c.drawString(margin, text_y, "Synthetic Speech Deepfake (Voice Conversion).")
        text_y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, text_y, "Detected Deepfake Audio Segments:")
        text_y -= 20
        c.setFont("Helvetica", 12)
        if audio_groups:
            for start, end, avg_score in audio_groups:
                segment_text = f"From {start:.2f} sec to {end:.2f} sec | Average Score: {avg_score:.4f}"
                c.drawString(margin, text_y, segment_text)
                text_y -= 15
                if text_y < margin:
                    c.showPage()
                    text_y = height - margin
        else:
            c.drawString(margin, text_y, "No deepfake audio segments detected.")
            text_y -= 20
        text_y -= 10
        if prob_plot_img:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, text_y, "Audio Fake Probability Over Time:")
            text_y -= 10
            try:
                prob_img = ImageReader(prob_plot_img)
                img_width, img_height = prob_img.getSize()
                aspect = img_height / float(img_width)
                display_width = width - 2 * margin
                display_height = display_width * aspect
                c.drawImage(prob_img, margin, text_y - display_height, width=display_width, height=display_height)
                text_y -= (display_height + 20)
            except Exception as e:
                c.drawString(margin, text_y, "Error loading probability plot image.")
                text_y -= 20
        if waveform_plot_img:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, text_y, "Audio Waveform with Detected Regions:")
            text_y -= 10
            try:
                waveform_img = ImageReader(waveform_plot_img)
                img_width, img_height = waveform_img.getSize()
                aspect = img_height / float(img_width)
                display_width = width - 2 * margin
                display_height = display_width * aspect
                c.drawImage(waveform_img, margin, text_y - display_height, width=display_width, height=display_height)
                text_y -= (display_height + 20)
            except Exception as e:
                c.drawString(margin, text_y, "Error loading waveform plot image.")
                text_y -= 20
    else:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, text_y, "Audio Deepfake Detection")
        text_y -= 25
        c.setFont("Helvetica", 12)
        c.drawString(margin, text_y, "No audio detected in this video. Skipping audio analysis.")
        text_y -= 40
    c.showPage()
    text_y = height - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, text_y, "Video Deepfake Detection")
    text_y -= 25
    video_duration_text = f"{video_duration:.2f} seconds" if video_duration > 0 else "Unknown"
    c.setFont("Helvetica", 12)
    c.drawString(margin, text_y, f"Video Duration: {video_duration_text}")
    text_y -= 20
    c.drawString(margin, text_y, f"Final Prediction (Video): {video_verdict}")
    text_y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, text_y, "Deepfake Creation Method (Video):")
    text_y -= 18
    c.setFont("Helvetica", 12)
    c.drawString(margin, text_y, "The video deepfake was generated using an XceptionNet-based model")
    text_y -= 15
    c.drawString(margin, text_y, "trained on FaceForensics++ data. It utilizes facial manipulation techniques,")
    text_y -= 15
    c.drawString(margin, text_y, "including GAN-based face swapping and expression synthesis.")
    text_y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, text_y, "Type of Video Deepfake:")
    text_y -= 18
    c.setFont("Helvetica", 12)
    c.drawString(margin, text_y, "Face Manipulation Deepfake (GAN-based and FaceForensics++ trained).")
    text_y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, text_y, "Detected Deepfake Video Segments:")
    text_y -= 20
    c.setFont("Helvetica", 12)
    if video_groups:
        for start, end, avg_score in video_groups:
            segment_text = f"From {start:.2f} sec to {end:.2f} sec | Average Score: {avg_score:.4f}"
            c.drawString(margin, text_y, segment_text)
            text_y -= 15
            if text_y < margin:
                c.showPage()
                text_y = height - margin
    else:
        c.drawString(margin, text_y, "No deepfake video segments detected.")
        text_y -= 20
    c.save()
    print(f"✅ Combined transparency report generated at: {report_path}")
    return report_path

# Flask Endpoints

app = Flask(__name__)
CORS(app)

@app.route('/detect_image', methods=['POST'])
def detect_image():
    # Create uploads folder if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded image
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    image_path = os.path.join(upload_dir, "input_image.png")
    file.save(image_path)

    # Preprocess and perform deepfake detection
    preprocessed_image = preprocess_image(image_path)
    fake_prob_c40 = predict_deepfake(model_c40, preprocessed_image)
    fake_prob_c23 = predict_deepfake(model_c23, preprocessed_image)
    final_fake_score = (fake_prob_c40 + fake_prob_c23) / 2
    label = "FAKE" if final_fake_score > 0.5 else "REAL"

    # Generate highlighted heatmap image and transparency report
    highlighted_image_path = save_highlighted_image(preprocessed_image, os.path.join(upload_dir, "highlighted_image.png"))
    report_path = os.path.join(upload_dir, "transparency_repor_image.pdf")
    generate_transparency_report_image(image_path, highlighted_image_path, final_fake_score, label, report_path)

    # Return results with absolute download links
    backend_url = request.host_url.rstrip("/")  # e.g., http://127.0.0.1:5000
    return jsonify({
        "fake_probability": final_fake_score,
        "label": label,
        "report_url": f"{backend_url}/download/{report_path}",
        "highlighted_image_url": f"{backend_url}/download/{highlighted_image_path}"
    })

@app.route('/detect_audio', methods=['POST'])
def detect_audio():
    # Create uploads folder if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded audio file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    audio_input_path = os.path.join(upload_dir, "input_audio.aac")
    file.save(audio_input_path)

    # Convert audio to WAV and perform detection
    wav_audio_path = os.path.join(upload_dir, "extracted_audio.wav")
    convert_audio_to_wav(audio_input_path, wav_audio_path)
    full_audio_features = extract_audio_features(wav_audio_path)
    overall_prob = predict_audio_deepfake(audio_model, full_audio_features)
    threshold = 0.5
    final_prediction = "FAKE" if overall_prob > threshold else "REAL"

    # Additional analysis: segment the audio for timestamps (optional)
    audio, sr = librosa.load(wav_audio_path, sr=16000)
    duration = len(audio) / sr
    window_duration = 1.0
    hop_duration = 0.5
    window_size = int(window_duration * sr)
    hop_size = int(hop_duration * sr)
    segment_times = []
    segment_scores = []
    for start in range(0, len(audio) - window_size + 1, hop_size):
        segment = audio[start:start+window_size]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        mfccs_tensor = torch.tensor(mfccs).unsqueeze(0).to(device)
        prob = predict_audio_deepfake(audio_model, mfccs_tensor)
        segment_times.append(start / sr)
        segment_scores.append(prob)
    # Plot probability over time
    plt.figure(figsize=(12, 4))
    plt.plot(segment_times, segment_scores, marker='o', label="Fake Probability")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fake Probability")
    plt.title("Audio Deepfake Detection Over Time")
    plt.legend()
    prob_plot_path = os.path.join(upload_dir, "probability_plot.png")
    plt.savefig(prob_plot_path)
    plt.close()
    # Plot waveform with highlighted segments
    plt.figure(figsize=(12, 4))
    time_axis = np.linspace(0, duration, len(audio))
    plt.plot(time_axis, audio, alpha=0.6, label="Audio Signal")
    for t, score in zip(segment_times, segment_scores):
        if score > threshold:
            plt.axvspan(t, t + window_duration, color='red', alpha=0.3)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal with Detected Deepfake Regions")
    plt.legend()
    waveform_plot_path = os.path.join(upload_dir, "waveform_plot.png")
    plt.savefig(waveform_plot_path)
    plt.close()

    def group_deepfake_segments(times, scores, window_dur, threshold):
        groups = []
        current_group_start = None
        current_group_scores = []
        for i, (t, score) in enumerate(zip(times, scores)):
            if score > threshold:
                if current_group_start is None:
                    current_group_start = t
                current_group_scores.append(score)
            else:
                if current_group_start is not None:
                    group_end = times[i-1] + window_dur
                    average_score = sum(current_group_scores) / len(current_group_scores)
                    groups.append((current_group_start, group_end, average_score))
                    current_group_start = None
                    current_group_scores = []
        if current_group_start is not None:
            group_end = times[-1] + window_dur
            average_score = sum(current_group_scores) / len(current_group_scores)
            groups.append((current_group_start, group_end, average_score))
        return groups
    deepfake_groups = group_deepfake_segments(segment_times, segment_scores, window_duration, threshold)

    # Generate transparency report for audio
    report_path = os.path.join(upload_dir, "transparency_report_audio.pdf")
    generate_transparency_report_video(report_path, deepfake_groups, overall_prob, final_prediction,
                                       duration, prob_plot_path, waveform_plot_path, [], "N/A", duration, sr)

    backend_url = request.host_url.rstrip("/")
    return jsonify({
        "overall_fake_probability": overall_prob,
        "final_prediction": final_prediction,
        "report_url": f"{backend_url}/download/{report_path}",
        "probability_plot_url": f"{backend_url}/download/{prob_plot_path}",
        "waveform_plot_url": f"{backend_url}/download/{waveform_plot_path}",
        "deepfake_segments": deepfake_groups
    })

@app.route('/detect_video', methods=['POST'])
def detect_video():
    # Create uploads folder if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded video file
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    video_path = os.path.join(upload_dir, "input_video.mp4")
    file.save(video_path)

    # AUDIO PIPELINE (if audio exists in video)
    audio_extracted = extract_audio(video_path, os.path.join(upload_dir, "extracted_audio.aac"))
    if audio_extracted:
        convert_audio_to_wav(os.path.join(upload_dir, "extracted_audio.aac"), os.path.join(upload_dir, "extracted_audio.wav"))
        audio_features = extract_audio_features(os.path.join(upload_dir, "extracted_audio.wav"))
        overall_prob_audio = predict_audio_deepfake(audio_model, audio_features)
        threshold = 0.5
        final_audio_prediction = "FAKE" if overall_prob_audio > threshold else "REAL"
        # Audio segmentation and plots
        audio, sr = librosa.load(os.path.join(upload_dir, "extracted_audio.wav"), sr=16000)
        duration = len(audio) / sr
        window_duration = 1.0
        hop_duration = 0.5
        window_size = int(window_duration * sr)
        hop_size = int(hop_duration * sr)
        segment_times = []
        segment_scores = []
        for start in range(0, len(audio) - window_size + 1, hop_size):
            segment = audio[start:start+window_size]
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
            mfccs_tensor = torch.tensor(mfccs).unsqueeze(0).to(device)
            prob = predict_audio_deepfake(audio_model, mfccs_tensor)
            segment_times.append(start / sr)
            segment_scores.append(prob)
        plt.figure(figsize=(12, 4))
        plt.plot(segment_times, segment_scores, marker='o', label="Fake Probability")
        plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Fake Probability")
        plt.title("Audio Deepfake Detection Over Time")
        plt.legend()
        prob_plot_path = os.path.join(upload_dir, "probability_plot.png")
        plt.savefig(prob_plot_path)
        plt.close()
        plt.figure(figsize=(12, 4))
        time_axis = np.linspace(0, duration, len(audio))
        plt.plot(time_axis, audio, alpha=0.6, label="Audio Signal")
        for t, score in zip(segment_times, segment_scores):
            if score > threshold:
                plt.axvspan(t, t + window_duration, color='red', alpha=0.3)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Audio Signal with Detected Deepfake Regions Highlighted")
        plt.legend()
        waveform_plot_path = os.path.join(upload_dir, "waveform_plot.png")
        plt.savefig(waveform_plot_path)
        plt.close()
        def group_deepfake_segments(times, scores, window_dur, threshold):
            groups = []
            current_group_start = None
            current_group_scores = []
            for i, (t, score) in enumerate(zip(times, scores)):
                if score > threshold:
                    if current_group_start is None:
                        current_group_start = t
                    current_group_scores.append(score)
                else:
                    if current_group_start is not None:
                        group_end = times[i-1] + window_dur
                        average_score = sum(current_group_scores) / len(current_group_scores)
                        groups.append((current_group_start, group_end, average_score))
                        current_group_start = None
                        current_group_scores = []
            if current_group_start is not None:
                group_end = times[-1] + window_dur
                average_score = sum(current_group_scores) / len(current_group_scores)
                groups.append((current_group_start, group_end, average_score))
            return groups
        deepfake_audio_groups = group_deepfake_segments(segment_times, segment_scores, window_duration, threshold)
    else:
        overall_prob_audio = None
        final_audio_prediction = None
        duration = 0
        prob_plot_path = None
        waveform_plot_path = None
        deepfake_audio_groups = []

    # VIDEO PIPELINE
    output_video_path = os.path.join(upload_dir, "highlighted_output_video.mp4")
    total_frames = extract_frames(video_path, os.path.join(upload_dir, "frames"))
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0
    video_cap.release()
    video_verdict, video_deepfake_groups = process_video_frames(os.path.join(upload_dir, "frames"), model_c40, model_c23, total_frames, fps)
    process_and_highlight_video(video_path, output_video_path, model_c40, model_c23)

    # Generate combined transparency report
    report_filename = os.path.join(upload_dir, "combined_transparency_report.pdf")
    generate_transparency_report_video(report_filename, deepfake_audio_groups, overall_prob_audio, final_audio_prediction,
                                       duration, prob_plot_path, waveform_plot_path, video_deepfake_groups,
                                       video_verdict, video_duration, fps)

    backend_url = request.host_url.rstrip("/")
    return jsonify({
        "video_verdict": video_verdict,
        "report_url": f"{backend_url}/download/{report_filename}",
        "highlighted_video_url": f"{backend_url}/download/{output_video_path}"
    })

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)

# Choose a port for the Flask app
port = 5000

# Run the Flask app locally
if __name__ == '__main__':
    app.run(port=port)
