# Bangla Text To Speech

An application to generate Bangla speech from your voice.

**Built by [O Kolkata Radio](https://www.okolkataradio.com)**

## 🎯 Overview

Bangla Text To Speech is a professional web application that converts Bengali text into natural-sounding speech using advanced text-to-speech synthesis and voice conversion technology. Users can generate speech using default male/female voices or register custom speaker voices for personalized audio output.

## ✨ Features

- **Default Voice Options**
  - 👩 Female Voice - High-quality default female voice synthesis
  - 👨 Male Voice - High-quality default male voice synthesis

- **Custom Speaker Support**
  - Register new speaker voices using the phoneme coverage script
  - Voice-adapted speech synthesis for registered speakers
  - Automatic voice conversion using FreeVC technology

- **Professional Interface**
  - Clean, modern web-based UI
  - Real-time speech generation
  - Audio playback and download capability
  - Responsive design for all devices

- **Speaker Management**
  - Register and manage custom speaker voices
  - Voice model cache management
  - Speaker sample replacement and deletion

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Text-to-Speech**: Coqui TTS v0.27.5
  - Bengali TTS Models: `tts_models/bn/custom/vits-female` and `vits-male`
  - Voice Conversion: FreeVC with WavLM checkpoint
- **Audio Processing**: PyTorch, torchaudio, torchcodec
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Language Support**: Bengali (Bangla)

## 📋 System Requirements

- **Python**: 3.10+
- **RAM**: Minimum 4GB (recommended 8GB+ for smooth model loading)
- **Storage**: 3GB+ for TTS and voice conversion models
- **OS**: Windows, macOS, or Linux

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/bangla-text-to-speech.git
cd "Bengali Text To Speech"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

**Activate Virtual Environment:**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5050`

## 📁 Project Structure

```
Bengali Text To Speech/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── outputs/                  # Generated audio files
├── speakers/                 # Registered speaker samples
│   └── Abhra/
│       └── sample.wav
├── static/
│   └── style.css            # Application styling
└── templates/
    ├── index.html           # Main page (speech generation)
    ├── manage_speakers.html # Speaker management
    ├── register_speaker.html # Speaker registration
    ├── phoneme_script.html   # Phoneme reading script
    └── Logo.png             # Application logo
```

## 💻 Usage

### Generate Speech
1. Navigate to the home page
2. Enter Bengali text in the text area
3. Select a voice:
   - Default Female Voice
   - Default Male Voice
   - Custom Speaker (if registered)
4. Click "Generate Speech"
5. Download or play the generated audio

### Register a Custom Speaker
1. Open **"📖 Phoneme Reading Script"** to view the complete script
2. Record a clear WAV audio file (30–90 seconds)
   - Format: 16-bit PCM, mono
   - Sample rate: 22050 Hz or 24000 Hz
   - Environment: Quiet room, no background noise
3. Go to **"Register Speaker"** page
4. Enter a speaker name and upload the WAV file
5. Click "Register Speaker"
6. New speaker will appear in voice selection dropdown

### Manage Speakers
- View registered speakers and their sample sizes
- Replace speaker samples
- Delete speakers
- Repair voice conversion model cache if needed

## 🎤 Recording Tips

For best results when registering speaker voices:
- **Environment**: Use a quiet room with minimal background noise
- **Equipment**: Use a good quality microphone
- **Speaking**: Speak clearly at a natural, steady pace
- **Volume**: Keep consistent volume throughout the recording
- **Avoid**: Shouting, whispering, background music, or clipping
- **Duration**: Record between 30–90 seconds (minimum 6 seconds)

## 🔧 Configuration

### Audio Output Settings
Audio files are saved to the `outputs/` directory with unique filenames (UUID format).

### Speaker Storage
Custom speaker samples are stored in `speakers/<speaker_name>/sample.wav` format.

### Model Caching
- TTS models are lazy-loaded on first use
- Voice conversion model cache location: `~/.local/tts/wavlm/WavLM-Large.pt`
- Cache can be cleared from the "Manage Speakers" page

## 🐛 Troubleshooting

### Issue: "Speaker sample not found" error
- **Solution**: Ensure the WAV file is properly uploaded and the speaker was successfully registered.

### Issue: Slow first synthesis
- **Solution**: First use downloads and caches the TTS model (~917 MB). Subsequent requests are much faster.

### Issue: Voice conversion fails
- **Solution**: Go to "Manage Speakers" and click "Repair Voice Conversion Cache" to re-download the model.

### Issue: Flask doesn't reload code changes
- **Solution**: 
  - Clear `__pycache__` directories
  - Use `python -B app.py` to prevent bytecode caching
  - Restart Flask completely

## 📝 API Endpoints

- `GET /` - Home page with speech generation
- `POST /synthesize` - Generate speech from text
- `GET /manage-speakers` - Speaker management page
- `GET /register-speaker` - Speaker registration page
- `POST /register-speaker` - Submit new speaker
- `POST /delete-speaker` - Delete registered speaker
- `POST /replace-sample` - Replace speaker sample
- `POST /clear-vc-cache` - Clear voice conversion cache
- `GET /phoneme-script` - Phoneme reading script
- `GET /outputs/<filename>` - Download generated audio

## 📄 Dependencies

Key Python packages:
- `Flask` - Web framework
- `TTS` - Text-to-Speech synthesis (Coqui TTS v0.27.5)
- `torch` - Deep learning framework
- `torchaudio` - Audio processing
- `transformers` - Transformer models (>=4.57,<5.0)

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Coqui TTS** - For providing state-of-the-art text-to-speech models
- **FreeVC** - For voice conversion technology
- **O Kolkata Radio** - For supporting this project
- **Community Contributors** - For feedback and improvements

## 📞 Support

For issues, questions, or suggestions:
- Visit: [O Kolkata Radio](https://www.okolkataradio.com)
- Create an issue on GitHub
- Contact: support@okolkataradio.com

## 🚀 Future Enhancements

- [ ] Support for multiple Indian languages
- [ ] Advanced audio editing features
- [ ] Batch processing for multiple texts
- [ ] API for programmatic access
- [ ] Cloud deployment option
- [ ] Real-time streaming speech
- [ ] Emotion/tone control
- [ ] Speech speed and pitch adjustment

---

**Version**: 1.0.0  
**Last Updated**: April 27, 2026  
**Status**: Production Ready
