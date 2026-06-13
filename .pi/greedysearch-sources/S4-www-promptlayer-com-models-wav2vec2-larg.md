---
url: https://www.promptlayer.com/models/wav2vec2-large-xlsr-basque/
title: wav2vec2-large-xlsr-basque
source: http
status: 200
chars: 1916
---

Property

Value

License

Apache-2.0

Author

cahya

Test WER

12.44%

Base Model

wav2vec2-large-xlsr-53

## What is wav2vec2-large-xlsr-basque?

wav2vec2-large-xlsr-basque is a specialized speech recognition model fine-tuned specifically for the Basque language. Built upon Facebook's wav2vec2-large-xlsr-53 architecture, this model demonstrates impressive performance in automatic speech recognition (ASR) tasks, achieving a Word Error Rate (WER) of 12.44% on the Common Voice Basque test dataset.

## Implementation Details

The model operates on 16kHz audio input and utilizes the powerful Wav2Vec2 architecture combined with CTC (Connectionist Temporal Classification) for speech recognition. It's implemented using PyTorch and the Transformers library, making it easily accessible for deployment in production environments.

-   Built on the wav2vec2-large-xlsr-53 foundation model
-   Requires 16kHz audio input sampling rate
-   Implements CTC-based speech recognition
-   Fine-tuned on the Basque Common Voice dataset

## Core Capabilities

-   Direct speech-to-text transcription without language model
-   Batch processing support for multiple audio files
-   Robust performance on Basque speech recognition
-   Compatible with standard audio processing libraries like torchaudio

## Frequently Asked Questions

### Q: What makes this model unique?

This model is specifically optimized for Basque language speech recognition, leveraging the powerful XLSR-53 architecture while achieving a competitive 12.44% WER on the test set. It's one of the few models specifically trained for Basque ASR.

### Q: What are the recommended use cases?

The model is ideal for Basque speech transcription tasks, including audio content indexing, subtitle generation, and voice command systems. It's particularly suitable for applications requiring real-time transcription without the need for a separate language model.