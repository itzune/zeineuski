---
url: https://huggingface.co/HiTZ/whisper-large-v3-eu
title: HiTZ/whisper-large-v3-eu · Hugging Face
source: http
status: 200
chars: 6738
---

## [](#whisper-large-v3-basque)Whisper Large-V3 Basque

## [](#model-summary)Model summary

**Whisper Large-V3 Basque** is an automatic speech recognition (ASR) model for **Basque (eu)** speech. It is fine-tuned from \[openai/whisper-large-v3\] on the **Basque portion of Mozilla Common Voice 13.0**, achieving a **Word Error Rate (WER) of 10.62%** on the Common Voice evaluation split.

This model offers state-of-the-art transcription quality for Basque speech, delivering improved accuracy and robustness over previous large Whisper variants while remaining suitable for offline and batch processing.

* * *

## [](#model-description)Model description

-   **Architecture:** Transformer-based encoder–decoder (Whisper)
-   **Base model:** openai/whisper-large-v3
-   **Language:** Basque (eu)
-   **Task:** Automatic Speech Recognition (ASR)
-   **Output:** Text transcription in Basque
-   **Decoding:** Autoregressive sequence-to-sequence decoding

Leveraging Whisper’s multilingual pretraining, this large-v3 model is fine-tuned on Basque speech data to provide highly accurate transcription for a low-resource language, suitable for research, media, and archival use cases.

* * *

## [](#intended-use)Intended use

### [](#primary-use-cases)Primary use cases

-   High-quality transcription of Basque audio recordings
-   Offline or batch ASR pipelines
-   Research and development in Basque ASR
-   Media, educational, and archival transcription tasks

### [](#intended-users)Intended users

-   Researchers working on Basque or low-resource ASR
-   Developers building Basque speech applications
-   Academic and institutional users

### [](#out-of-scope-use)Out-of-scope use

-   Real-time or low-latency ASR without optimization
-   Speech translation tasks
-   Safety-critical applications without validation

* * *

## [](#limitations-and-known-issues)Limitations and known issues

-   Performance may degrade on:
    -   Noisy or low-quality recordings
    -   Conversational or spontaneous speech
    -   Accents underrepresented in Common Voice
-   While highly accurate, transcription errors may still occur under challenging acoustic conditions
-   Dataset biases from Common Voice may be reflected in outputs

Users are encouraged to evaluate the model on their own data before deployment.

* * *

## [](#training-and-evaluation-data)Training and evaluation data

### [](#training-data)Training data

-   **Dataset:** Mozilla Common Voice 13.0 (Basque subset)
-   **Data type:** Crowd-sourced, read speech
-   **Preprocessing:**
    -   Audio resampled to 16 kHz
    -   Text normalized using Whisper tokenizer
    -   Filtering of invalid or problematic samples

### [](#evaluation-data)Evaluation data

-   **Dataset:** Mozilla Common Voice 13.0 (Basque evaluation split)
-   **Metric:** Word Error Rate (WER)

* * *

## [](#evaluation-results)Evaluation results

Metric

Value

WER (eval)

**10.62%**

These results indicate state-of-the-art transcription performance for Basque ASR using a large-v3 Whisper model.

* * *

## [](#training-procedure)Training procedure

### [](#training-hyperparameters)Training hyperparameters

-   Learning rate: 1e-5
-   Optimizer: Adam (β1=0.9, β2=0.999, ε=1e-8)
-   LR scheduler: Linear
-   Warmup steps: 500
-   Training steps: 20,000
-   Train batch size: 32
-   Gradient accumulation steps: 2
-   Total effective batch size: 64
-   Evaluation batch size: 16
-   Seed: 42
-   Mixed precision training: Native AMP

### [](#training-results-summary)Training results (summary)

Training Loss

Epoch

Step

Validation Loss

WER

0.0326

4.85

1000

0.2300

13.3278

0.004

9.71

2000

0.2723

12.2038

0.0058

14.56

3000

0.2771

12.4246

0.003

19.42

4000

0.2838

12.2119

0.003

24.27

5000

0.2740

11.7704

0.0014

29.13

6000

0.2936

11.5436

0.0015

33.98

7000

0.2911

11.5193

0.0012

38.83

8000

0.2939

11.3674

0.0009

43.69

9000

0.3039

11.4140

0.0002

48.54

10000

0.3063

10.9624

0.0009

53.4

11000

0.3014

11.3350

0.0011

58.25

12000

0.3052

11.0474

0.0001

63.11

13000

0.3204

10.8692

0.0

67.96

14000

0.3413

10.7092

0.0

72.82

15000

0.3524

10.6647

0.0

77.67

16000

0.3607

10.6566

0.0

82.52

17000

0.3675

10.6120

0.0

87.38

18000

0.3737

10.6140

0.0

92.23

19000

0.3782

10.6181

0.0

97.09

20000

0.3803

10.6201

* * *

## [](#framework-versions)Framework versions

-   Transformers 4.37.2
-   PyTorch 2.2.0+cu121
-   Datasets 2.16.1
-   Tokenizers 0.15.1

* * *

## [](#how-to-use)How to use

```
from transformers import pipeline

hf_model = "HiTZ/whisper-large-v3-eu"  # replace with actual repo ID
device = 0  # set to -1 for CPU

pipe = pipeline(
    task="automatic-speech-recognition",
    model=hf_model,
    device=device
)

result = pipe("audio.wav")
print(result["text"])
```

* * *

## [](#ethical-considerations-and-risks)Ethical considerations and risks

-   This model transcribes speech and may process personal data.
-   Users should ensure compliance with applicable data protection laws (e.g., GDPR).
-   The model should not be used for surveillance or non-consensual audio processing.

* * *

## [](#citation)Citation

If you use this model in your research, please cite:

```
@misc{dezuazo2025whisperlmimprovingasrmodels,
  title={Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages},
  author={Xabier de Zuazo and Eva Navas and Ibon Saratxaga and Inma Hernáez Rioja},
  year={2025},
  eprint={2503.23542},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

Please, check the related paper preprint in [arXiv:2503.23542](https://arxiv.org/abs/2503.23542) for more details.

* * *

## [](#license)License

This model is available under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0). You are free to use, modify, and distribute this model as long as you credit the original creators.

* * *

## [](#contact-and-attribution)Contact and attribution

-   Fine-tuning and evaluation: HiTZ/Aholab - Basque Center for Language Technology
-   Base model: OpenAI Whisper
-   Dataset: Mozilla Common Voice

For questions or issues, please open an issue in the model repository.

## [](#funding)Funding

This project with reference 2022/TL22/00215335 has been parcially funded by the Ministerio de Transformación Digital and by the Plan de Recuperación, Transformación y Resiliencia – Funded by the European Union – NextGenerationEU [ILENIA](https://proyectoilenia.es/) and by the project [IkerGaitu](https://www.hitz.eus/iker-gaitu/) funded by the Basque Government. This model was trained at [Hyperion](https://scc.dipc.org/docs/systems/hyperion/overview/), one of the high-performance computing (HPC) systems hosted by the DIPC Supercomputing Center.