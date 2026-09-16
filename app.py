import io
import streamlit as st
import torch
import torchaudio
import soundfile as sf
import json

from src.models.AASIST import Model as AASISTModel

@st.cache_resource
def load_aasist_model(conf_path, inference_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(conf_path, "r") as f :
        conf = json.load(f)
    model = AASISTModel(conf["model_config"]) 
    model = model.to(device)
    model.load_state_dict(torch.load(inference_path, map_location=device)) 
    model.eval()
    return model, device

def preprocessing(audio):
    audio.seek(0)

    data, sr = sf.read(io.BytesIO(audio.read()))
    waveform = torch.from_numpy(data).float()

    #load the audio on the device
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    #put it in 16000Hz
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

    #length ~ 4s
    audio = waveform.squeeze(0)
    if len(audio) < 64600:
        audio = audio.repeat((64600 // len(audio)) + 1)[:64600]
    else:
        audio = audio[:64600]

    #add a dimension
    audio = audio.unsqueeze(0)

    return audio

st.title("Audio Deepfakes Detection")
CONFIG_PATH = "config/AASIST.conf"
INFERENCE_PATH = "checkpoints/best_aasist_model.pth"  

model, device = load_aasist_model(CONFIG_PATH, INFERENCE_PATH)
st.success(f"Modèle AASIST chargé avec succès sur : {device}")

if "choice" not in st.session_state:
    st.session_state.choice = None

choice = st.menu_button("Choose if you want to record directly your voice or upload an audio file", options=["Recorded voice", "uploaded audio file"])
if choice is not None:
    st.session_state.choice = choice
uploaded_file = None


if st.session_state.choice == "Recorded voice":
   uploaded_file = st.audio_input("Record your voice")
   if uploaded_file is not None:
        st.success("Voice recorded successfully!")
elif st.session_state.choice == "uploaded audio file":
    uploaded_file = st.file_uploader("UPLOAD AUDIO FILE", type=["flac","wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        st.success("File uploaded with success")
else: 
    st.info("Please select an option to proceed.")        


if uploaded_file is not None:
    if st.button("Analyser l'audio"):
        with st.spinner("Analyzing audio..."):
            # Preprocess and move tensor to device
            input_tensor = preprocessing(uploaded_file).to(device)

            # Model inference
            with torch.no_grad():
                _, outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            spoof_prob = probabilities[0][0].item()
            bonafide_prob = probabilities[0][1].item()

        # Render results
        st.divider()
        if bonafide_prob > spoof_prob:
            st.success(f"**Authentic Audio (Bonafide)** — Confidence: {bonafide_prob * 100:.2f}%")
        else:
            st.error(f"**Deepfake Audio (Spoof)** — Confidence: {spoof_prob * 100:.2f}%")
 






