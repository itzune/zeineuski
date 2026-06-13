---
url: https://www.gladia.io/blog/best-open-source-speech-to-text-models
title: Best open-source speech-to-text models in 2026
source: http
status: 200
chars: 7912
---

Open-source speech-to-text (STT) models — also called automatic speech recognition (ASR) models — convert spoken audio into written text. Modern ASR systems typically use encoder-decoder transformer architectures: the encoder extracts acoustic features from raw audio, and the decoder generates text sequences from those features.

Open-source models give developers full control over deployment, fine-tuning, and data privacy. But that flexibility comes with trade-offs: infrastructure costs, optimization work, and production features like speaker diarization, PII redaction, and real-time streaming that most open-source models don't include out of the box.

The [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) is the standard benchmark for comparing these models, ranking them by Word Error Rate (WER) and real-time factor (RTFx) across diverse datasets. Worth noting: most leaderboard datasets are read-speech or broadcast audio — not the messy, multi-speaker conversational audio that most production voice applications actually process. We'll flag this gap for each model where it's relevant.

## Quick comparison: Top open-source STT models (2026)

Model

Developer

Parameters

Avg WER (%)

Languages

Best For

NVIDIA Canary-Qwen 2.5B

NVIDIA

2.5B

5.63

25

Highest benchmark accuracy

IBM Granite Speech 3.3 8B

IBM

~9B

5.85

English + 7 translation

Accuracy + speech translation

Qwen3-ASR 1.7B

Alibaba / Qwen

1.7B

Competitive with top commercial APIs

52

Multilingual breadth

Whisper Large V3

OpenAI

1.55B

7.4

99+

Multilingual coverage

Whisper Large V3 Turbo

OpenAI

809M

7.75

99+

Speed vs. accuracy balance

NVIDIA Parakeet TDT 1.1B

NVIDIA

1.1B

~8.0

English (+ multilingual variants)

Maximum throughput

Moonshine

Useful Sensors

27M–331M

Comparable to Whisper Large V3

English (expanding)

Edge / on-device

SpeechBrain

Open community

Varies (toolkit)

Varies

Multi

Research / custom pipelines

## 1\. OpenAI Whisper

**The most widely adopted open-source ASR model:** Whisper remains the default starting point for most developers. Trained on over 5 million hours of multilingual audio data (up from 680,000 hours in the original release), Whisper uses an end-to-end encoder-decoder transformer architecture that handles transcription, translation, language identification, and timestamp prediction in a single model.

**What's new:** Whisper Large V3 expanded training data by 635% compared to V2, achieving a [10–20% error reduction across languages](https://openai.com/research/whisper-v3) according to OpenAI's published results. Whisper Large V3 Turbo prunes the decoder from 32 layers to 4, cutting parameters from 1.55B to 809M — the result is approximately 216x real-time processing speed with only a marginal WER increase (7.75% vs 7.4%). Distil-Whisper compresses Large V3 further to 756M parameters with WER within 1% of the full model and 5–6x faster inference. OpenAI also released GPT-4o-based transcription models in early 2025 that outperform all Whisper versions on benchmarks — but these are commercial API-only, not open source.

**Strengths:**

-   Broadest language support (99+ languages)
-   Strong accuracy across accents, noise, and technical vocabulary
-   Massive ecosystem: Hugging Face integrations, community fine-tunes, deployment tools
-   MIT license

**Limitations:**

-   Well-documented hallucination issues on silent or low-quality audio segments — a significant problem for long-form or noisy recordings
-   No built-in speaker diarization; requires bolting on a separate model such as [pyannote.audio](https://github.com/pyannote/pyannote-audio)
-   On real conversational audio (crosstalk, overlapping speakers, accents), WER degrades meaningfully beyond what leaderboard numbers suggest
-   Requires significant GPU resources at full size (Large V3 needs ~10GB VRAM)
-   Not optimized for real-time streaming out of the box

**Who should use it:** Developers who need broad multilingual coverage and can tolerate batch processing latency. If you need English-only with maximum speed, Parakeet or Distil-Whisper are better choices. If conversational audio quality is the priority, test carefully against your actual data before committing.

## 2\. NVIDIA Canary-Qwen 2.5B

**Currently the #1 model on the Hugging Face Open ASR Leaderboard:** Released in June 2025, Canary-Qwen 2.5B uses a Speech-Augmented Language Model (SALM) architecture that pairs a FastConformer encoder optimized for speech recognition with an unmodified Qwen3-1.7B LLM decoder. It tops the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) with a 5.63% average WER.

**Key specs:**

-   Architecture: FastConformer encoder + Qwen3-1.7B LLM decoder
-   Training data: 234,000 hours of English audio
-   LibriSpeech WER: 1.6% (clean) / 3.1% (other)
-   Noise tolerance: 2.41% WER at 10 dB SNR
-   License: CC-BY-4.0
-   Supported languages: 25 (via the Canary-1b-v2 family)

**Strengths:**

-   Highest accuracy among open-source models on standard benchmarks
-   Strong noise robustness as measured on benchmark datasets
-   Inference up to 10x faster than similarly accurate models
-   Requires NVIDIA NeMo framework — well-maintained and production-grade

**Limitations:**

-   Higher VRAM requirements than smaller models
-   Language coverage (25) is narrower than Whisper (99+) or Qwen3-ASR (52)
-   Training data is primarily English read-speech and broadcast audio; performance on multi-speaker conversational audio has less published evidence than benchmark numbers imply
-   CC-BY-4.0 license requires attribution

**Who should use it:**Teams prioritizing transcription accuracy above all else, especially for English and European languages. Strong choice when you have GPU infrastructure and primarily process clean or semi-clean audio. Test against your own conversational data before assuming leaderboard WER transfers.

[...content trimmed...]

**Is Kaldi still relevant in 2026?**

Kaldi remains in production at some large organizations, and [Kaldi 2.0 (k2)](https://github.com/k2-fsa/k2) continues development combining neural end-to-end models with WFST decoding. For new projects, modern end-to-end models offer better accuracy with significantly less setup complexity. Kaldi is best suited for teams with existing infrastructure or very specific domain adaptation requirements.

**Which open-source STT model is best for real-time transcription?**

NVIDIA Parakeet TDT 1.1B offers the highest throughput at RTFx above 2,000. For edge and on-device real-time processing, Moonshine v2 with its Ergodic Streaming Encoder is designed specifically for latency-critical applications. Whisper Large V3 Turbo provides strong real-time performance at 216x real-time speed with broader language coverage.

**Can open-source STT models do speaker diarization?**

Most open-source ASR models — including Whisper, Canary-Qwen, and Parakeet — do not include built-in speaker diarization. Integrating diarization requires adding a separate model (such as [pyannote.audio](https://github.com/pyannote/pyannote-audio)), aligning speaker segments with transcript output, and handling edge cases like speaker overlap. SpeechBrain provides a framework for building this pipeline. For production use cases where diarization quality matters (contact center, meeting intelligence), the integration complexity is worth factoring into your build-vs-buy calculation.

**What is the best open-source model for non-English languages?**

For sheer language count, Whisper Large V3 supports 99+ languages. Qwen3-ASR covers 52 languages and dialects with competitive accuracy. For fine-tuning on low-resource languages with limited labeled data, [Wav2Vec 2.0 / XLS-R](https://arxiv.org/abs/2111.09296) (128 languages) remains the strongest foundation model for custom fine-tuning.