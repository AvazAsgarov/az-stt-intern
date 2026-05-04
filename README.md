# Azerbaijani ASR System (Mozilla Common Voice)

Bu layihə Azerbaijani ASR (Automatic Speech Recognition) sisteminin qurulması və Mozilla Common Voice dataseti üzərində fine-tuning edilməsini əhatə edir. Layihə "AI Engineer Intern" texniki tapşırığı çərçivəsində hazırlanıb.

## Layihə Strukturu
- `part_a/`: ASR Baza Pipeline (Whisper-Medium) kodu.
- `part_b/`: Fine-Tuning cəhdi (Whisper-Small) kodu.
- `results/`: İnferens nəticələri, WER/CER cədvəlləri və qrafiklər.
- `requirements.txt`: Lazımi Python kitabxanaları.

## İstifadə Olunan Model və Parametrlər
- **Baza Model:** `openai/whisper-medium`
- **Fine-Tuning Modeli:** `openai/whisper-small`
- **Dataset:** Mozilla Common Voice 17.0 (Azerbaijani split)
- **Parametrlər:**
  - Sampling Rate: 16000 Hz
  - Batch Size: 8
  - Learning Rate: 1e-5
  - Max Steps: 100 (demo məqsədilə)

## WER/CER Nəticələri

### Baza Model (Part A)
| Metric | Value |
| --- | --- |
| **Corpus WER** | ~45.2% |
| **Corpus CER** | ~12.8% |

### Fine-Tuning Müqayisəsi (Part B)
| Model | WER (%) | CER (%) |
| --- | --- | --- |
| Base (Whisper-Small) | 58.4% | 18.2% |
| Fine-Tuned (Small) | **42.1%** | **11.5%** |

*(Qeyd: Nəticələr kiçik dataset (200 train, 50 val) üzərində qısa müddətli training-ə əsaslanır.)*

## Quraşdırılma və İşə Salma

1. Kitabxanaları quraşdırın:
```bash
pip install -r requirements.txt
```

2. Baza pipeline-ı işə salın:
```bash
python part_a/asr_pipeline.py
```

3. Fine-tuning prosesini başladın:
```bash
python part_b/asr_finetuning.py
```

## Müəllif
Avaz Asgarov
