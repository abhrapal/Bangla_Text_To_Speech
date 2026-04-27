import os
import uuid
import threading
import tempfile
import shutil
import traceback
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, abort
from TTS.api import TTS
from trainer.io import get_user_data_dir

print("[STARTUP] ===== APP RELOADING =====")

app = Flask(__name__)
app.secret_key = "bengali-tts-secret"

# Bengali TTS models: female and male voices
MODEL_FEMALE = "tts_models/bn/custom/vits-female"
MODEL_MALE = "tts_models/bn/custom/vits-male"
VC_MODEL_NAME = os.environ.get("VC_MODEL_NAME", "voice_conversion_models/multilingual/vctk/freevc24")

# ── Directories ──────────────────────────────────────────────────
SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "speakers")
OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(SPEAKERS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR,  exist_ok=True)

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
    # Get form inputs
    bengali_text  = request.form.get("text", "").strip()
    speaker_name  = request.form.get("speaker", "").strip()

    if not bengali_text:
        flash("Please enter Bengali text.", "error")
        return redirect(url_for("index"))

    # Generate unique output filename
    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path     = os.path.join(OUTPUTS_DIR, output_filename)
    
    # Determine gender and check for custom speaker
    gender = "female"  # default
    speaker_wav = None
    
    # Handle default voices (female/male) vs custom speakers
    if speaker_name in ("female", "male"):
        # Default voice selection
        gender = speaker_name
        print(f"[DEBUG] Selected gender voice: {gender}")
    elif speaker_name:
        # Custom speaker
        speaker_wav = os.path.join(SPEAKERS_DIR, speaker_name, "sample.wav")
        if not os.path.isfile(speaker_wav):
            flash(f"Speaker sample for '{speaker_name}' not found.", "error")
            return redirect(url_for("index"))
        print(f"[DEBUG] Using custom speaker: {speaker_name}")
    else:
        print(f"[DEBUG] Using default voice: female")

    try:
        tts = get_tts_model(gender=gender)
        print(f"[synthesize] gender={gender}  speaker_wav={speaker_wav!r}  is_multi_lingual={tts.is_multi_lingual}")

        if speaker_wav:
            # Two-step: (1) generate base Bengali audio, (2) apply FreeVC voice conversion.
            # Do NOT pass language to either call since the Bengali model is mono-lingual.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=OUTPUTS_DIR) as tmp:
                base_wav_path = tmp.name
            
            try:
                # Step 1: Base Bengali synthesis (no language parameter for mono-lingual model)
                tts.tts_to_file(
                    text=bengali_text,
                    file_path=base_wav_path,
                    split_sentences=True,
                )
                
                # Step 2: Apply voice conversion to match speaker timbre
                # Note: VC model loading can fail with corrupted checkpoint; retry after clearing cache
                try:
                    vc = get_vc_model()
                except Exception as vc_load_error:
                    if not is_vc_checkpoint_corruption_error(vc_load_error):
                        raise
                    print("[synthesize] VC checkpoint corrupted on load; clearing and retrying...")
                    clear_wavlm_checkpoint_cache()
                    vc = get_vc_model()
                
                vc.voice_conversion_to_file(
                    source_wav=base_wav_path,
                    target_wav=speaker_wav,
                    file_path=output_path,
                )
            finally:
                if os.path.exists(base_wav_path):
                    os.remove(base_wav_path)
        else:
            # No speaker selected: plain Bengali synthesis (mono-lingual, no language parameter)
            tts.tts_to_file(
                text=bengali_text,
                file_path=output_path,
                split_sentences=True,
            )
    except Exception as e:
        traceback.print_exc()
        flash(f"Synthesis failed: {e}", "error")
        return redirect(url_for("index"))

    flash("Speech generated successfully. You can play it below.", "success")
    return redirect(url_for("index", audio=output_filename))


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)
