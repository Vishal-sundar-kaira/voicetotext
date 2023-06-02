# for managing aurdio files
import librosa
import torch # For working with pytorch
# Wav2Vec2ForCTC use for actual transcription task and Wav2Vec2CTCTokenizer use for converting to token that model can understand
from transformers import Wav2Vec2Tokenizer,Wav2Vec2ForCTC;
import streamlit as st

# so now lets download pretrained models tokenizer for getting token and model for training audio
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(audio_path):
    # loading an audio file
    # here sr is frequency hertz we are giving it by our choice(sampling rate)
    # inside speech there will be file and inside there will be rate or frequency i.e here 16000
    speech, rate = librosa.load(audio_path,sr=16000)
    #so librosa help us to convert audio file in waveform.

    # here we are giving out speech or audio file to pytorch through tokenizer to convert into token
    input_values = tokenizer(speech, return_tensors = 'pt').input_values
    # return_tensorts="pt" confirms that It will give use tensor in pytorch format.

    # store logits (non-normalized prediction)
    logits=model(input_values).logits

    # store predicted id's(this is the main ids now we will pass this for decode and get our transcribe)
    predicted_ids=torch.argmax(logits,dim=-1)

    #decode audio to generate text by passing ids in tokenizer
    transcription=tokenizer.decode(predicted_ids[0])
    return transcription

def main():
    st.title("Voice to Text")

    # File uploader
    uploaded_file = st.file_uploader("upload an audio file", type=["wav"])

    if uploaded_file is not None:
        # Transcribe the uploaded audio file
        transcription = transcribe_audio(uploaded_file)
        st.text("Transcription:")
        st.write(transcription)

if __name__ == "__main__":
    main()



