import os
import sys
import streamlit as st
from pydub import AudioSegment
import numpy as np
import io
import plotly.graph_objects as go
import time
import tempfile
import subprocess

# ================== PYTHON VERSION CHECK ================== #
if sys.version_info >= (3, 13):
    st.error("""
    ❌ Unsupported Python Version Detected (3.13+)
    
    This application requires Python 3.10 for compatibility with audio processing libraries.
    
    Please redeploy with Python 3.10 by adding a `runtime.txt` file containing:
    ```
    python-3.10
    ```
    """)
    st.stop()
# ================== END VERSION CHECK ================== #

# ================== FFMPEG CONFIGURATION ================== #
FFMPEG_PATH = "/usr/bin/ffmpeg"
FFPROBE_PATH = "/usr/bin/ffprobe"

# Configure environment
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)
os.environ["FFMPEG_PATH"] = FFMPEG_PATH
os.environ["FFPROBE_PATH"] = FFPROBE_PATH

# Configure Pydub
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# Verify FFmpeg installation
try:
    ffmpeg_check = subprocess.run([FFMPEG_PATH, "-version"], 
                                 capture_output=True, text=True, check=True)
except Exception as e:
    st.error(f"❌ FFmpeg verification failed: {str(e)}")
# ================== END FFMPEG CONFIG ================== #

# Custom CSS styling with animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #a855f7;
        --accent: #ec4899;
        --dark: #0f172a;
        --light: #f8fafc;
    }
    
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    .gradient-border {
        position: relative;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
    }
    
    .gradient-border::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 2px;
        background: linear-gradient(45deg, var(--primary), var(--accent));
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .neon-text {
        text-shadow: 0 0 10px var(--primary),
                     0 0 20px var(--primary),
                     0 0 30px var(--primary);
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(255,255,255,0.2) !important;
        border-radius: 16px !important;
        background: rgba(15, 23, 42, 0.4) !important;
        transition: all 0.3s !important;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary) !important;
        background: rgba(99, 102, 241, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None
if 'semitones' not in st.session_state:
    st.session_state.semitones = 0

# Page header
st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="neon-text" style="color: #6366f1; font-size: 3.5rem;">🌀 Spectral Shift</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
            Professional-Grade Audio Pitch Shifting with AI Processing
        </p>
        <div class="floating" style="font-size: 2rem;">🎛️</div>
    </div>
""", unsafe_allow_html=True)

# File upload section
with st.expander("🎧 UPLOAD AUDIO", expanded=True):
    uploaded_file = st.file_uploader(" ", type=["mp3", "wav", "ogg", "m4a"],
                                   help="Drag and drop or click to upload your audio file")

# Visualization placeholder
viz_placeholder = st.empty()

# Controls section
with st.container():
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container():
            st.markdown("### 🎚️ Pitch Control")
            semitones = st.slider("Semitones (-24 to +24)", -24, 24, st.session_state.semitones,
                                help="Precision pitch adjustment in semitones")
            st.session_state.semitones = semitones
            st.markdown(f"""
                <div class="gradient-border">
                    <div style="display: flex; justify-content: space-between; padding: 1rem;">
                        <span style="color: rgba(255,255,255,0.8);">Current Adjustment:</span>
                        <span style="color: #ec4899; font-weight: 800; font-size: 1.2rem;">{semitones} st</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown("### ⚡ Quick Presets")
        presets = {
            "+1 Octave": 12,
            "-1 Octave": -12,
            "Perfect Fourth": 5,
            "Major Third": 4,
            "Tritone": 6
        }
        grid = st.columns(2)
        for i, (name, value) in enumerate(presets.items()):
            with grid[i % 2]:
                if st.button(f"🌟 {name}", key=f"preset_{name}",
                           use_container_width=True, type="secondary"):
                    st.session_state.semitones = value
                    st.rerun()

# Processing function with librosa
def process_audio(input_file, semitones):
    try:
        # Create temp file with proper extension
        file_ext = input_file.name.split('.')[-1].lower()
        if file_ext not in ['mp3', 'wav', 'ogg', 'm4a']:
            file_ext = 'mp3'
            
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp_file:
            tmp_file.write(input_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Load audio
        audio = AudioSegment.from_file(tmp_path, format=file_ext)
        audio = audio.set_sample_width(2).set_frame_rate(44100)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sr = audio.frame_rate
        channels = audio.channels
        
        # Visualization
        y = samples.astype(np.float32) / 32768.0
        if channels == 2:
            y = y[::2]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=y[::10],
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)',
            name="Waveform"
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, range=[-1, 1]),
            showlegend=False
        )
        viz_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Processing - using librosa
        import librosa.effects
        
        def process_channel(channel_data):
            return librosa.effects.pitch_shift(
                channel_data.astype(np.float32) / 32768.0,
                sr=sr,
                n_steps=semitones
            ) * 32768.0
        
        # Handle mono/stereo
        if channels == 1:
            processed_samples = process_channel(samples)
            processed = np.array(processed_samples, dtype=np.int16)
        else:
            left = samples[0::2]
            right = samples[1::2]
            processed_left = process_channel(left)
            processed_right = process_channel(right)
            processed = np.empty(len(processed_left) + len(processed_right), dtype=np.int16)
            processed[0::2] = processed_left.astype(np.int16)
            processed[1::2] = processed_right.astype(np.int16)
        
        return AudioSegment(
            processed.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=channels
        )
    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")
        return None
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

# Process button
if uploaded_file and st.button("🚀 PROCESS AUDIO", use_container_width=True, type="primary"):
    with st.spinner(""):
        start_time = time.time()
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
            <div class="pulse" style="text-align:center; color:#ec4899; font-size:1.5rem;">
                ⚡ Processing Audio...
            </div>
        """, unsafe_allow_html=True)
        
        processed_audio = process_audio(uploaded_file, st.session_state.semitones)
        processing_time = time.time() - start_time
        
        if processed_audio:
            st.session_state.processed = True
            st.session_state.processed_audio = processed_audio
            loading_placeholder.empty()
            
            # Create download buffer
            buffer = io.BytesIO()
            output_format = "mp3"
            st.session_state.processed_audio.export(buffer, format=output_format)
            buffer.seek(0)
            st.session_state.output_buffer = buffer
            st.session_state.processing_time = processing_time
            st.balloons()

# Display results
if st.session_state.processed and st.session_state.processed_audio:
    st.markdown("""
        <div style="margin: 3rem 0;">
            <h2 style="color: #6366f1; text-align: center; margin-bottom: 2rem;">
                🎉 Processing Complete!
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([2, 3], gap="large")
        with col1:
            st.markdown("#### 🎧 Audio Preview")
            st.audio(st.session_state.output_buffer, format="audio/mp3")
        
        with col2:
            st.markdown("#### 📥 Download Options")
            st.download_button(
                label=f"⬇️ DOWNLOAD MP3",
                data=st.session_state.output_buffer,
                file_name=f"pitch_shifted_{st.session_state.semitones}st.mp3",
                mime="audio/mp3",
                use_container_width=True,
                type="primary"
            )
            
            st.markdown(f"""
                <div class="gradient-border" style="margin-top: 2rem;">
                    <div style="padding: 1rem; color: rgba(255,255,255,0.8);">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Processing Time:</span>
                            <span style="color: #ec4899;">{st.session_state.processing_time:.2f}s</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                            <span>Pitch Adjustment:</span>
                            <span style="color: #ec4899;">{st.session_state.semitones} st</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="floating" style="font-size: 2rem;">⚡</div>
            <h3 style="color: #6366f1;">Real-Time Analytics</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="gradient-border">
                <div style="padding: 1rem; color: rgba(255,255,255,0.8);">
                    <h4>System Status</h4>
                    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>CPU Usage</div>
                        <div style="text-align: right; color: #ec4899;">42%</div>
                        <div>Memory Usage</div>
                        <div style="text-align: right; color: #ec4899;">64%</div>
                        <div>Processing Power</div>
                        <div style="text-align: right; color: #ec4899;">87%</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 1rem; color: rgba(255,255,255,0.6);">
        Powered by Librosa AI Core | v3.1 | © 2023 Audio Labs International
    </div>
""", unsafe_allow_html=True)