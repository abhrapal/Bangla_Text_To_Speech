from TTS.api import TTS

tts = TTS('tts_models/bn/custom/vits-female')
print(f'Sample rate: {tts.synthesizer.output_sample_rate}')
print(f'Model: {tts.model_name}')
