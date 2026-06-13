---
url: https://huggingface.co/xezpeleta/whisper-large-v3-eu
title: xezpeleta/whisper-large-v3-eu · Hugging Face
source: http
status: 200
chars: 5414
---

## [](#whisper-large-v3-basque)Whisper Large v3 Basque

This model is a fine-tuned version of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) specifically for Basque (eu) language Automatic Speech Recognition (ASR). It was trained on the [asierhv/composite\_corpus\_eu\_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1) dataset, which is a composite corpus designed to improve Basque ASR performance.

**Key improvements and results compared to the base model:**

-   **Significant WER reduction:** The fine-tuned model achieves a Word Error Rate (WER) of 6.5443 on the validation set of the `asierhv/composite_corpus_eu_v2.1` dataset, demonstrating a substantial improvement in accuracy for Basque speech.
-   **Exceptional performance on Common Voice:** When evaluated on the Mozilla Common Voice 18.0 dataset, the model achieved a WER of 4.84. This showcases the model's outstanding ability to generalize to diverse Basque speech datasets, and highlights the high accuracy achievable with the large-v3 model.

## [](#model-description)Model description

This model leverages the `whisper-large-v3` architecture, the most powerful variant of the Whisper models, known for its exceptional accuracy in multilingual speech recognition. By fine-tuning this model on a dedicated Basque speech corpus, it achieves state-of-the-art performance in Basque ASR. The `whisper-large-v3` model offers the highest capacity and therefore the highest accuracy, but requires significantly more computational resources.

## [](#intended-uses--limitations)Intended uses & limitations

**Intended uses:**

-   Ultra-high-accuracy automatic transcription of Basque speech for critical applications.
-   Development of cutting-edge Basque speech-based applications demanding the highest possible precision.
-   Research in Basque speech processing requiring the most accurate transcriptions.
-   Professional transcription services and applications where accuracy is paramount and computational resources are available.
-   Use in scenarios where the highest possible accuracy is required, and the computational cost is justifiable.

**Limitations:**

-   Performance is still influenced by audio quality, with challenges arising from background noise and poor recording conditions.
-   Accuracy may be affected by highly dialectal or informal Basque speech, although the large model mitigates this to a great degree.
-   Despite its high performance, the model may still produce errors, particularly with complex linguistic structures or rare words.
-   The large-v3 model demands substantial computational resources, making it less suitable for real-time or resource-constrained applications.

## [](#training-and-evaluation-data)Training and evaluation data

-   **Training dataset:** [asierhv/composite\_corpus\_eu\_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1). This dataset is a comprehensive and meticulously curated collection of Basque speech data, designed to maximize the performance of Basque ASR systems.
-   **Evaluation Dataset:** The `test` split of `asierhv/composite_corpus_eu_v2.1`.

## [](#training-procedure)Training procedure

### [](#training-hyperparameters)Training hyperparameters

The following hyperparameters were used during training:

-   learning\_rate: 4.375e-06
-   train\_batch\_size: 16
-   eval\_batch\_size: 8
-   seed: 42
-   optimizer: Use OptimizerNames.ADAMW\_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer\_args=No additional optimizer arguments
-   lr\_scheduler\_type: linear
-   lr\_scheduler\_warmup\_steps: 1000
-   training\_steps: 20000
-   mixed\_precision\_training: Native AMP

### [](#training-results)Training results

Training Loss

Epoch

Step

Validation Loss

Wer

0.2854

0.025

500

0.4194

25.8898

0.1425

0.05

1000

0.3923

20.5071

0.2199

0.075

1500

0.3291

17.4785

0.2343

0.1

2000

0.2861

14.1314

0.1391

0.125

2500

0.2906

13.3134

0.0853

0.15

3000

0.2688

12.0457

0.0866

0.175

3500

0.2575

11.4712

0.1311

0.2

4000

0.2472

12.4828

0.1338

0.225

4500

0.2437

10.9904

0.0748

0.25

5000

0.2557

10.7094

0.0821

0.275

5500

0.2597

10.2473

0.0988

0.3

6000

0.2407

9.4480

0.0824

0.325

6500

0.2425

9.2232

0.0678

0.35

7000

0.2301

9.1358

0.1124

0.375

7500

0.2559

9.3231

0.1122

0.4

8000

0.2240

8.5238

0.0477

0.425

8500

0.2379

8.3177

0.0638

0.45

9000

0.2354

8.9484

0.0735

0.475

9500

0.2231

8.3989

0.0548

0.5

10000

0.2330

8.5737

0.0557

0.525

10500

0.2133

8.3614

0.0626

0.55

11000

0.2084

8.2865

0.0472

0.575

11500

0.2331

8.0742

0.0636

0.6

12000

0.2118

7.9618

0.0466

0.625

12500

0.2126

7.4685

0.0604

0.65

13000

0.2160

7.6558

0.0544

0.675

13500

0.2187

7.9993

0.07

0.7

14000

0.2117

7.4372

0.0534

0.725

14500

0.1381

7.0438

0.046

0.75

15000

0.1496

7.0813

0.066

0.775

15500

0.1525

7.0001

0.0632

0.8

16000

0.1408

6.6817

0.0437

0.825

16500

0.1475

6.5942

0.0478

0.85

17000

0.1573

6.7941

0.0418

0.875

17500

0.1565

6.6504

0.0382

0.9

18000

0.1559

6.5630

0.0658

0.925

18500

0.1452

6.5630

0.0531

0.95

19000

0.1576

6.6629

0.0416

0.975

19500

0.1550

6.5443

0.0435

1.0

20000

0.1549

6.5443

### [](#framework-versions)Framework versions

-   Transformers 4.49.0.dev0
-   Pytorch 2.6.0+cu124
-   Datasets 3.3.1.dev0
-   Tokenizers 0.21.0