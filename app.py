import os
import uuid
import threading
import tempfile
import shutil
import traceback
import re
import numpy as np
import soundfile as sf
from scipy import signal
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, abort
from TTS.api import TTS
from trainer.io import get_user_data_dir
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token for IndicF5 access
HF_TOKEN = os.getenv("HF_TOKEN", "")
PREFERRED_TTS_MODEL = os.getenv("PREFERRED_TTS_MODEL", "indicf5")

print("[STARTUP] ===== APP RELOADING =====")

app = Flask(__name__)
app.secret_key = "bengali-tts-secret"

# Bengali TTS models: female and male voices (Coqui - backup)
MODEL_FEMALE = "tts_models/bn/custom/vits-female"
MODEL_MALE = "tts_models/bn/custom/vits-male"
VC_MODEL_NAME = os.environ.get("VC_MODEL_NAME", "voice_conversion_models/multilingual/vctk/freevc24")

# IndicF5 Model (Primary TTS)
INDICF5_MODEL = "ai4bharat/IndicF5"

# ── Directories ──────────────────────────────────────────────────
SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "speakers")
OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(SPEAKERS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR,  exist_ok=True)

# ── Text and Audio Processing Functions ───────────────────────────
def clean_bengali_text(text):
    """
    Clean Bengali text by removing punctuation marks.
    Handles both Bengali and English punctuation.
    """
    # Bengali punctuation marks
    bengali_punctuation = [
        '।',   # Danda
        '॥',   # Double danda
        '।।',  # Another variant
        ',', '.', '!', '?', ';', ':', '-', '"', "'", '"', '"', ''', '''
    ]
    
    cleaned = text
    for punct in bengali_punctuation:
        cleaned = cleaned.replace(punct, ' ')
    
    # Remove multiple spaces and strip
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def resample_audio(wav_path, target_sr=22050):
    """
    Resample audio file to target sample rate.
    Essential for matching speaker samples to TTS output rate (22050 Hz).
    """
    try:
        data, sr = sf.read(wav_path, dtype='float32')
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        if sr == target_sr:
            return  # Already at target rate
        
        # Calculate resampling ratio
        num_samples = int(len(data) * target_sr / sr)
        
        # Use scipy's resample for high-quality resampling
        data_resampled = signal.resample(data, num_samples)
        
        # Write back at new sample rate
        sf.write(wav_path, data_resampled, target_sr, subtype='PCM_16')
        print(f"[Audio Processing] Resampled {wav_path} from {sr}Hz to {target_sr}Hz")
        
    except Exception as e:
        print(f"[Audio Processing] Warning: Could not resample audio: {e}")


def normalize_audio(wav_path, target_db=-20):
    """
    Normalize audio to prevent clipping and improve voice conversion quality.
    Useful before voice conversion to ensure consistent levels.
    """
    try:
        data, sr = sf.read(wav_path, dtype='float32')
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(data**2))
        
        if current_rms < 1e-5:
            return  # Silent file, skip normalization
        
        # Calculate target amplitude
        target_linear = 10 ** (target_db / 20.0)
        
        # Scale audio
        data = data * (target_linear / current_rms)
        
        # Soft clip to prevent distortion
        data = np.clip(data, -0.98, 0.98)
        
        sf.write(wav_path, data, sr, subtype='PCM_16')
        print(f"[Audio Processing] Normalized {wav_path} to {target_db}dB")
        
    except Exception as e:
        print(f"[Audio Processing] Warning: Could not normalize audio: {e}")


def trim_audio_artifacts(wav_path, threshold_db=-40, min_duration=0.5):
    """
    Trim silence and artifacts from the end of audio file.
    Removes trailing noise/artifacts that cause strange sounds.
    """
    try:
        # Read audio file
        data, sr = sf.read(wav_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Calculate RMS energy for each frame
        frame_length = int(sr * 0.02)  # 20ms frames
        hop_length = frame_length // 2
        
        rms_energy = np.array([
            np.sqrt(np.mean(data[i:i+frame_length]**2))
            for i in range(0, len(data) - frame_length, hop_length)
        ])
        
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold_db / 20.0)
        
        # Find the last frame above threshold
        above_threshold = np.where(rms_energy > threshold_linear)[0]
        
        if len(above_threshold) == 0:
            # All silence, keep at least min_duration
            keep_samples = int(sr * min_duration)
            data = data[:keep_samples]
        else:
            # Find end of last non-silent frame
            last_above_idx = above_threshold[-1]
            end_sample = min((last_above_idx + 1) * hop_length + frame_length, len(data))
            # Add small buffer (0.2 seconds) after the last voice
            buffer_samples = int(sr * 0.2)
            end_sample = min(end_sample + buffer_samples, len(data))
            data = data[:end_sample]
        
        # Write back the trimmed audio
        sf.write(wav_path, data, sr, subtype='PCM_16')
        print(f"[Audio Processing] Trimmed artifacts from: {wav_path}")
        
    except Exception as e:
        print(f"[Audio Processing] Warning: Could not trim audio: {e}")
        # If trimming fails, continue without it (don't break synthesis)
        pass


def generate_reference_text_from_audio(audio_path):
    """
    Auto-generate reference text from audio using Whisper speech-to-text.
    Used when registering custom speakers - transcribes their sample audio.
    """
    try:
        import whisper
        print(f"[STT] Transcribing reference audio: {audio_path}")
        
        # Load Whisper model for transcription
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language="bn")
        
        ref_text = result["text"].strip()
        print(f"[STT] Transcribed text: {ref_text}")
        
        return ref_text if ref_text else "আমি বাংলা বলি"  # Fallback text
        
    except Exception as e:
        print(f"[STT] Warning: Could not transcribe audio: {e}")
        # Return fallback Bengali text if transcription fails
        return "আমি বাংলা বলি"


# ── Lazy-load IndicF5 Model (Primary TTS) ──────────────────────────
_indicf5_model = None
_indicf5_lock = threading.Lock()


def get_indicf5_model():
    """Load and cache IndicF5 model from Hugging Face."""
    global _indicf5_model
    if _indicf5_model is None:
        with _indicf5_lock:
            if _indicf5_model is None:
                try:
                    print("[IndicF5] Loading model with HF token...")
                    from transformers import AutoModel
                    
                    _indicf5_model = AutoModel.from_pretrained(
                        INDICF5_MODEL,
                        trust_remote_code=True,
                        token=HF_TOKEN if HF_TOKEN else None
                    )
                    print("[IndicF5] Model loaded successfully.")
                except Exception as e:
                    print(f"[IndicF5] Failed to load: {e}")
                    _indicf5_model = None
                    raise
    
    return _indicf5_model


def synthesize_with_indicf5(text, ref_audio_path=None, ref_text=None):
    """
    Generate speech using IndicF5 TTS model.
    
    Args:
        text: Bengali text to synthesize
        ref_audio_path: Optional reference speaker audio path
        ref_text: Optional text of reference audio (auto-generated if not provided)
    
    Returns:
        Audio numpy array at 24kHz, or None if synthesis fails
    """
    try:
        model = get_indicf5_model()
        if model is None:
            return None
        
        # If reference speaker provided, auto-generate reference text if needed
        if ref_audio_path and not ref_text:
            ref_text = generate_reference_text_from_audio(ref_audio_path)
        
        # Set defaults for reference audio/text (uses first available example)
        if not ref_audio_path:
            # Use a generic reference if available
            ref_audio_path = None
            ref_text = "আমি বাংলা বলি"
        
        print(f"[IndicF5] Synthesizing: '{text[:50]}...' with ref_text: '{ref_text}'")
        
        audio = model(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text
        )
        
        # Ensure audio is float32
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        
        print("[IndicF5] Synthesis successful")
        return audio
        
    except Exception as e:
        print(f"[IndicF5] Synthesis failed: {e}")
        traceback.print_exc()
        return None

# ── Lazy-load Coqui TTS models (female and male) ────────────────
# Model load/download can take a while on first run.
_tts_model_female = None
_tts_model_male = None
_tts_lock = threading.Lock()
_vc_model = None
_vc_lock = threading.Lock()


def get_tts_model(gender="female"):
    """Load and cache the TTS model (female or male) on first synthesis request."""
    global _tts_model_female, _tts_model_male
    
    if gender == "male":
        if _tts_model_male is None:
            with _tts_lock:
                if _tts_model_male is None:
                    print("Loading male TTS model ...")
                    _tts_model_male = TTS(MODEL_MALE)
                    print("Male TTS model ready.")
        return _tts_model_male
    else:  # Default to female
        if _tts_model_female is None:
            with _tts_lock:
                if _tts_model_female is None:
                    print("Loading female TTS model ...")
                    _tts_model_female = TTS(MODEL_FEMALE)
                    print("Female TTS model ready.")
        return _tts_model_female


def get_vc_model():
    """Load and cache voice-conversion model for speaker adaptation."""
    global _vc_model
    if _vc_model is None:
        with _vc_lock:
            if _vc_model is None:
                print("Loading voice-conversion model ...")
                _vc_model = TTS(VC_MODEL_NAME)
                print("Voice-conversion model ready.")
    return _vc_model


# ── Helpers ───────────────────────────────────────────────────────
def list_speakers():
    """Return list of speaker names who have a sample.wav on disk."""
    speakers = []
    if os.path.isdir(SPEAKERS_DIR):
        for name in sorted(os.listdir(SPEAKERS_DIR)):
            wav_path = os.path.join(SPEAKERS_DIR, name, "sample.wav")
            if os.path.isfile(wav_path):
                speakers.append(name)
    return speakers


def speaker_wav_path(speaker_name):
    """Build the canonical sample path for a speaker name."""
    return os.path.join(SPEAKERS_DIR, speaker_name, "sample.wav")


def get_wavlm_checkpoint_path():
    """Return path of the WavLM checkpoint used by FreeVC."""
    return os.path.join(get_user_data_dir("tts"), "wavlm", "WavLM-Large.pt")


def reset_vc_model_cache():
    """Clear in-memory VC model so it reloads after cache repair."""
    global _vc_model
    with _vc_lock:
        _vc_model = None


def is_vc_checkpoint_corruption_error(error):
    """Detect the known torch/miniz corruption error for WavLM checkpoints."""
    message = str(error).lower()
    return (
        "pytorchstreamreader failed reading zip archive" in message
        or "failed finding central directory" in message
        or "checkpoint file is corrupted" in message
    )


def clear_wavlm_checkpoint_cache():
    """Remove cached WavLM checkpoint files so Coqui can re-download them."""
    wavlm_path = get_wavlm_checkpoint_path()
    wavlm_dir = os.path.dirname(wavlm_path)

    if os.path.isfile(wavlm_path):
        os.remove(wavlm_path)

    if os.path.isdir(wavlm_dir):
        for item in os.listdir(wavlm_dir):
            item_path = os.path.join(wavlm_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

    reset_vc_model_cache()


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    generated_audio = request.args.get("audio", "").strip()
    generated_audio_url = None

    if generated_audio:
        safe_name = os.path.basename(generated_audio)
        audio_path = os.path.join(OUTPUTS_DIR, safe_name)
        if safe_name == generated_audio and safe_name.lower().endswith(".wav") and os.path.isfile(audio_path):
            generated_audio_url = url_for("output_audio", filename=safe_name)

    return render_template(
        "index.html",
        speakers=list_speakers(),
        generated_audio_url=generated_audio_url,
    )


@app.route("/outputs/<path:filename>")
def output_audio(filename):
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".wav"):
        abort(404)

    output_path = os.path.join(OUTPUTS_DIR, safe_name)
    if not os.path.isfile(output_path):
        abort(404)

    return send_from_directory(OUTPUTS_DIR, safe_name, mimetype="audio/wav", as_attachment=False)


@app.route("/templates/<filename>")
def serve_template_assets(filename):
    """Serve images and other assets from the templates folder."""
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
        abort(404)
    
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_path = os.path.join(template_dir, safe_name)
    if not os.path.isfile(template_path):
        abort(404)
    
    return send_from_directory(template_dir, safe_name)


# ── 1. Phoneme Script page ────────────────────────────────────────
@app.route("/phoneme-script")
def phoneme_script():
    """
    Displays the phonemically rich Bengali reading script that every
    speaker must record to capture all Bengali phonemes.
    """
    # A curated phoneme-coverage script for Bengali.
    # Covers all primary vowels (স্বরবর্ণ), consonants (ব্যঞ্জনবর্ণ),
    # conjuncts (যুক্তাক্ষর), and common diphthongs.
    script_sentences = [
        "অআইঈউঊঋএঐওঔ — এগুলো বাংলা ভাষার মূল স্বরবর্ণ।",
        "আমি বাংলায় কথা বলি এবং বাংলা ভাষাকে ভালোবাসি।",
        "আকাশ নীল, মাঠ সবুজ, নদী শান্ত এবং পাখি গান করে।",
        "ক খ গ ঘ ঙ — এগুলো কণ্ঠ্য ব্যঞ্জনবর্ণ।",
        "চ ছ জ ঝ ঞ — এগুলো তালব্য বর্ণ।",
        "ট ঠ ড ঢ ণ — এগুলো মূর্ধন্য বর্ণ।",
        "ত থ দ ধ ন — এগুলো দন্ত্য বর্ণ।",
        "প ফ ব ভ ম — এগুলো ওষ্ঠ্য বর্ণ।",
        "য র ল ব শ ষ স হ ড় ঢ় য় ৎ — বিশেষ ব্যঞ্জনবর্ণ।",
        "বিদ্যালয়ে ছাত্ররা প্রতিদিন পড়াশোনা করে।",
        "স্বাস্থ্য, শিক্ষা এবং পরিবেশ রক্ষা করা আমাদের দায়িত্ব।",
        "তুমি কি কখনো পাহাড়ে গিয়েছ? সেখানে ঠান্ডা বাতাস বয়।",
        "আজকের সূর্যোদয় অত্যন্ত সুন্দর ছিল।",
        "ক্ষমা, জ্ঞান, শ্রম — এই তিনটি গুণ মানুষকে মহান করে।",
        "রাত্রিবেলা তারার আলোয় মন ভরে যায়।",
    ]
    return render_template("phoneme_script.html", sentences=script_sentences)


# ── 2. Register / Upload a Speaker Sample ────────────────────────
@app.route("/register-speaker", methods=["GET", "POST"])
def register_speaker():
    if request.method == "POST":
        speaker_name = request.form.get("speaker_name", "").strip()
        audio_file   = request.files.get("audio_file")

        if not speaker_name:
            flash("Please enter a speaker name.", "error")
            return redirect(url_for("register_speaker"))

        if not audio_file or audio_file.filename == "":
            flash("Please upload a WAV recording.", "error")
            return redirect(url_for("register_speaker"))

        # Save the sample
        speaker_dir = os.path.join(SPEAKERS_DIR, speaker_name)
        os.makedirs(speaker_dir, exist_ok=True)
        save_path = os.path.join(speaker_dir, "sample.wav")
        audio_file.save(save_path)

        flash(f"Speaker '{speaker_name}' registered successfully!", "success")
        return redirect(url_for("index"))

    return render_template("register_speaker.html")


@app.route("/manage-speakers")
def manage_speakers():
    speakers = []
    for name in list_speakers():
        wav_path = speaker_wav_path(name)
        speakers.append(
            {
                "name": name,
                "size_kb": round(os.path.getsize(wav_path) / 1024, 1),
            }
        )
    wavlm_path = get_wavlm_checkpoint_path()
    wavlm_exists = os.path.isfile(wavlm_path)
    wavlm_size_mb = round(os.path.getsize(wavlm_path) / (1024 * 1024), 1) if wavlm_exists else None
    return render_template(
        "manage_speakers.html",
        speakers=speakers,
        wavlm_path=wavlm_path,
        wavlm_exists=wavlm_exists,
        wavlm_size_mb=wavlm_size_mb,
    )


@app.route("/manage-speakers/replace", methods=["POST"])
def replace_speaker_sample():
    speaker_name = request.form.get("speaker_name", "").strip()
    audio_file = request.files.get("audio_file")

    if not speaker_name:
        flash("Missing speaker name.", "error")
        return redirect(url_for("manage_speakers"))

    if not audio_file or audio_file.filename == "":
        flash("Please choose a WAV file to replace the sample.", "error")
        return redirect(url_for("manage_speakers"))

    speaker_dir = os.path.join(SPEAKERS_DIR, speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)
    audio_file.save(speaker_wav_path(speaker_name))

    flash(f"Speaker '{speaker_name}' sample replaced.", "success")
    return redirect(url_for("manage_speakers"))


@app.route("/manage-speakers/delete", methods=["POST"])
def delete_speaker():
    speaker_name = request.form.get("speaker_name", "").strip()
    if not speaker_name:
        flash("Missing speaker name.", "error")
        return redirect(url_for("manage_speakers"))

    speaker_dir = os.path.join(SPEAKERS_DIR, speaker_name)
    if os.path.isdir(speaker_dir):
        shutil.rmtree(speaker_dir)
        flash(f"Speaker '{speaker_name}' deleted.", "success")
    else:
        flash(f"Speaker '{speaker_name}' not found.", "error")
    return redirect(url_for("manage_speakers"))


@app.route("/manage-speakers/repair-vc-cache", methods=["POST"])
def repair_vc_cache():
    try:
        clear_wavlm_checkpoint_cache()
        flash("Voice-conversion cache cleared. It will re-download on next speaker-adapted synthesis.", "success")
    except Exception as e:
        flash(f"Failed to clear VC cache: {e}", "error")

    return redirect(url_for("manage_speakers"))


# ── 3. Synthesise Speech ──────────────────────────────────────────
@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    Generate Bengali speech using IndicF5 (primary) with Coqui fallback.
    Supports custom speaker voice cloning via reference audio.
    """
    # Get form inputs
    bengali_text  = request.form.get("text", "").strip()
    speaker_name  = request.form.get("speaker", "").strip()

    if not bengali_text:
        flash("Please enter Bengali text.", "error")
        return redirect(url_for("index"))

    # Clean Bengali text: remove punctuation marks
    bengali_text = clean_bengali_text(bengali_text)
    
    if not bengali_text:
        flash("Please enter valid Bengali text.", "error")
        return redirect(url_for("index"))

    # Generate unique output filename
    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path     = os.path.join(OUTPUTS_DIR, output_filename)
    
    # Determine speaker type and reference audio
    speaker_wav = None
    ref_text = None
    
    # Handle default voices (female/male) vs custom speakers
    if speaker_name and speaker_name not in ("female", "male"):
        # Custom speaker
        speaker_wav = os.path.join(SPEAKERS_DIR, speaker_name, "sample.wav")
        if not os.path.isfile(speaker_wav):
            flash(f"Speaker sample for '{speaker_name}' not found.", "error")
            return redirect(url_for("index"))
        print(f"[synthesize] Using custom speaker: {speaker_name}")
        
        # Auto-generate reference text from speaker sample
        ref_text = generate_reference_text_from_audio(speaker_wav)
    else:
        print(f"[synthesize] Using default voice or IndicF5 auto")

    try:
        audio_data = None
        synthesis_method = None
        
        # ===== ATTEMPT 1: Try IndicF5 (Primary) =====
        if PREFERRED_TTS_MODEL == "indicf5":
            print("[synthesize] Attempting IndicF5 synthesis...")
            try:
                audio_data = synthesize_with_indicf5(
                    text=bengali_text,
                    ref_audio_path=speaker_wav,
                    ref_text=ref_text
                )
                if audio_data is not None:
                    synthesis_method = "IndicF5"
                    print("[synthesize] ✓ IndicF5 synthesis successful")
            except Exception as e:
                print(f"[synthesize] ✗ IndicF5 failed: {e}")
                audio_data = None
        
        # ===== ATTEMPT 2: Fallback to Coqui if IndicF5 failed =====
        if audio_data is None:
            print("[synthesize] Falling back to Coqui TTS...")
            try:
                audio_data, sr = _synthesize_with_coqui(
                    text=bengali_text,
                    speaker_wav=speaker_wav
                )
                if audio_data is not None:
                    synthesis_method = "Coqui (fallback)"
                    print("[synthesize] ✓ Coqui synthesis successful")
            except Exception as e:
                print(f"[synthesize] ✗ Coqui also failed: {e}")
                audio_data = None
        
        # If both failed, error out
        if audio_data is None:
            raise Exception("Both IndicF5 and Coqui synthesis failed. Check logs.")
        
        # ===== Save audio to file =====
        # Ensure audio is proper format
        if isinstance(audio_data, tuple):
            audio_array, sr = audio_data
        else:
            audio_array = audio_data
            sr = 24000  # IndicF5 output rate
        
        # Ensure float32
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize and trim
        temp_audio_path = output_path
        sf.write(temp_audio_path, audio_array, sr, subtype='PCM_16')
        
        normalize_audio(temp_audio_path, target_db=-20)
        trim_audio_artifacts(temp_audio_path)
        
        print(f"[synthesize] Audio saved to {output_path} via {synthesis_method}")
        
    except Exception as e:
        traceback.print_exc()
        flash(f"Synthesis failed: {e}", "error")
        return redirect(url_for("index"))

    flash(f"Speech generated successfully ({synthesis_method}). You can play it below.", "success")
    return redirect(url_for("index", audio=output_filename))


def _synthesize_with_coqui(text, speaker_wav=None):
    """
    Fallback synthesis using Coqui TTS (multilingual or gender-based).
    Returns (audio_array, sample_rate) tuple.
    """
    try:
        gender = "female"  # default
        
        if speaker_wav:
            # Use voice conversion for custom speakers
            tts = get_tts_model(gender="female")
            
            # Pre-process speaker sample to match TTS output rate (22050 Hz)
            speaker_sr_target = 22050
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=OUTPUTS_DIR) as tmp:
                base_wav_path = tmp.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=OUTPUTS_DIR) as tmp:
                resampled_speaker_wav = tmp.name
            
            try:
                # Check speaker sample rate and resample if needed
                speaker_data, actual_sr = sf.read(speaker_wav, dtype='float32')
                if actual_sr != speaker_sr_target:
                    print(f"[Coqui] Resampling speaker from {actual_sr}Hz to {speaker_sr_target}Hz...")
                    num_samples = int(len(speaker_data) * speaker_sr_target / actual_sr)
                    speaker_data_resampled = signal.resample(speaker_data, num_samples)
                    sf.write(resampled_speaker_wav, speaker_data_resampled, speaker_sr_target, subtype='PCM_16')
                    speaker_wav_to_use = resampled_speaker_wav
                else:
                    shutil.copy2(speaker_wav, resampled_speaker_wav)
                    speaker_wav_to_use = resampled_speaker_wav
                
                # Step 1: Base synthesis
                tts.tts_to_file(
                    text=text,
                    file_path=base_wav_path,
                    split_sentences=True,
                )
                
                normalize_audio(base_wav_path, target_db=-20)
                
                # Step 2: Voice conversion
                try:
                    vc = get_vc_model()
                except Exception as vc_load_error:
                    if not is_vc_checkpoint_corruption_error(vc_load_error):
                        raise
                    print("[Coqui] VC checkpoint corrupted; clearing cache...")
                    clear_wavlm_checkpoint_cache()
                    vc = get_vc_model()
                
                normalize_audio(speaker_wav_to_use, target_db=-20)
                
                output_path = os.path.join(OUTPUTS_DIR, f"coqui_temp_{uuid.uuid4().hex}.wav")
                vc.voice_conversion_to_file(
                    source_wav=base_wav_path,
                    target_wav=speaker_wav_to_use,
                    file_path=output_path,
                )
                
                # Read final audio
                audio_data, sr = sf.read(output_path, dtype='float32')
                
                # Clean up
                if os.path.exists(base_wav_path):
                    os.remove(base_wav_path)
                if os.path.exists(resampled_speaker_wav):
                    os.remove(resampled_speaker_wav)
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                return audio_data, sr
                
            except Exception as e:
                if os.path.exists(base_wav_path):
                    os.remove(base_wav_path)
                if os.path.exists(resampled_speaker_wav):
                    os.remove(resampled_speaker_wav)
                raise e
        
        else:
            # Default voices: just use TTS directly
            tts = get_tts_model(gender=gender)
            output_path = os.path.join(OUTPUTS_DIR, f"coqui_temp_{uuid.uuid4().hex}.wav")
            
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                split_sentences=True,
            )
            
            audio_data, sr = sf.read(output_path, dtype='float32')
            if os.path.exists(output_path):
                os.remove(output_path)
            
            return audio_data, sr
    
    except Exception as e:
        print(f"[Coqui] Synthesis error: {e}")
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)
