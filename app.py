import io
import streamlit as st
import torch
import torchaudio
import soundfile as sf
import json

from src.models.AASIST import Model as AASISTModel


# --------------------------------------------------
# Load AASIST model
# --------------------------------------------------

@st.cache_resource
def load_aasist_model(conf_path, inference_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(conf_path, "r") as f:
        conf = json.load(f)

    model = AASISTModel(conf["model_config"])

    model = model.to(device)

    model.load_state_dict(
        torch.load(
            inference_path,
            map_location=device
        )
    )

    model.eval()

    return model, device


# --------------------------------------------------
# Audio preprocessing
# --------------------------------------------------

def preprocessing(audio):

    audio.seek(0)

    data, sr = sf.read(
        io.BytesIO(audio.read())
    )

    waveform = torch.from_numpy(data).float()

    # Convert audio to [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    # Convert stereo/multi-channel audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # Resample to 16000 Hz
    resampler = torchaudio.transforms.Resample(
        sr,
        16000
    )

    waveform = resampler(waveform)

    # Adjust length to approximately 4 seconds
    audio = waveform.squeeze(0)

    if len(audio) < 64600:

        audio = audio.repeat(
            (64600 // len(audio)) + 1
        )[:64600]

    else:

        audio = audio[:64600]

    # Add batch dimension
    audio = audio.unsqueeze(0)

    return audio


# --------------------------------------------------
# Load CSS
# --------------------------------------------------

with open("style.css") as f:
    css = f.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# --------------------------------------------------
# Page title
# --------------------------------------------------

with st.container(key="app-container"):

    st.title("Reel time Audio Deepfake Detection")

    st.markdown(
        '<p class="app-description">Detect whether an audio is authentic or deefake using an AASIST deepfake-audio Detection model.</p>',
        unsafe_allow_html=True
    )  



    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    CONFIG_PATH = "config/AASIST.conf"
    INFERENCE_PATH = "checkpoints/best_aasist_model.pth"

    model, device = load_aasist_model(
        CONFIG_PATH,
        INFERENCE_PATH
    )


    # --------------------------------------------------
    # Session state
    # --------------------------------------------------

    if "choice" not in st.session_state:
        st.session_state.choice = None

    if "results" not in st.session_state:
        st.session_state.results = []

    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    # --------------------------------------------------
    # Input Audio
    # --------------------------------------------------

    st.markdown(
        """
    <div class="section-title">
        <span class="bar">|</span> AUDIO SOURCE
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Alignement des deux boutons côte à côte
    col_mic, col_upload = st.columns(2)

    with col_mic:
        # Détermine si ce choix est actif pour appliquer du style ou du contexte
        if st.button("🎙️ MICROPHONE", use_container_width=True, key="btn_mic"):
            st.session_state.choice = "Recorded voice"

    with col_upload:
        if st.button(
            "📁 UPLOAD FILES", use_container_width=True, key="btn_upload"
        ):
            st.session_state.choice = "uploaded audio file"


    uploaded_file = []


    # --------------------------------------------------
    # Recorded voice
    # --------------------------------------------------

    if st.session_state.choice == "Recorded voice":

        recorded_file = st.audio_input(
            "Record your voice"
        )

        if recorded_file is not None:

            uploaded_file = [recorded_file]

            st.success(
                "Voice recorded successfully!"
            )


    # --------------------------------------------------
    # Uploaded audio files
    # --------------------------------------------------

    elif st.session_state.choice == "uploaded audio file":

        uploaded_file = st.file_uploader(
            "UPLOAD AUDIO FILE",
            type=["flac", "wav"],
            accept_multiple_files=True
        )

        if uploaded_file:

            st.success("Files uploaded with success")

            st.write("### Uploaded Audio Files")

            for audio in uploaded_file:

                st.write(f"**{audio.name}**")
                st.audio(audio)



    else:

        st.info(
            "Please select an option to proceed."
        )


    # --------------------------------------------------
    # Analysis
    # --------------------------------------------------

    if uploaded_file:

        if st.button(
            "Analyser l'audio",
            key="green"
        ):

            results = []

            with st.spinner(
                "Analyzing audio..."
            ):

                for audio in uploaded_file:

                    # Preprocess audio
                    input_tensor = preprocessing(
                        audio
                    ).to(device)

                    # Model inference
                    with torch.no_grad():

                        _, outputs = model(
                            input_tensor
                        )

                        probabilities = torch.softmax(
                            outputs,
                            dim=1
                        )

                    # Get probabilities
                    spoof_prob = probabilities[0][0].item()
                    bonafide_prob = probabilities[0][1].item()

                    # Determine prediction
                    if bonafide_prob > spoof_prob:

                        label = "Authentic"
                        confidence = bonafide_prob

                    else:

                        label = "Spoof"
                        confidence = spoof_prob

                    # Save result
                    results.append({
                        "file_name": audio.name,
                        "label": label,
                        "confidence": confidence,
                        "bonafide_prob": bonafide_prob,
                        "spoof_prob": spoof_prob
                    })

            # Save results
            st.session_state.results = results


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    results = st.session_state.results


    if results:

        total_count = len(results)

        authentic_count = sum(
            1
            for r in results
            if r["label"] == "Authentic"
        )

        spoof_count = sum(
            1
            for r in results
            if r["label"] == "Spoof"
        )


        # --------------------------------------------------
        # Analysis Results
        # --------------------------------------------------

        st.subheader("Analysis Results")


        for result in results:

            file_name = result["file_name"]
            label = result["label"]
            confidence = result["confidence"]

            st.write(f"### {file_name}")

            if label == "Authentic":

                st.success(
                    f"Authentic Audio (Bonafide) — "
                    f"Confidence: {confidence * 100:.2f}%"
                )

            else:

                st.error(
                    f"Deepfake Audio (Spoof) — "
                    f"Confidence: {confidence * 100:.2f}%"
                )


            # Probabilities

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Authentic",
                    f"{result['bonafide_prob'] * 100:.2f}%"
                )

                st.progress(
                    result["bonafide_prob"]
                )

            with col2:

                st.metric(
                    "Spoof",
                    f"{result['spoof_prob'] * 100:.2f}%"
                )

                st.progress(
                    result["spoof_prob"]
                )

            st.divider()


        # --------------------------------------------------
        # Batch Analytics Summary
        # --------------------------------------------------

        st.subheader("Batch Analytics Summary")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Analyzed",
                total_count
            )


        with col2:

            st.metric(
                "Authentic",
                authentic_count
            )


        with col3:

            st.metric(
                "Deepfake",
                spoof_count
            )