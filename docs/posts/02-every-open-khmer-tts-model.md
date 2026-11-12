# Every open Khmer voice, measured, and where Gemini goes silent

*10 November 2026 · Part 2 of the Samleng-TTS write-ups · Previous: [How good is open Khmer TTS in 2026?](01-how-good-is-open-khmer-tts.md) · [Project README](../../README.md) (latest results)*

In the [first post](01-how-good-is-open-khmer-tts.md) I compared two models on 100 sentences. Since then the benchmark has grown to 320 sentences, and I ran every open Khmer text-to-speech model I could find through it, plus Google's Gemini: version 2.5, the voice my tutoring app uses today, and the newer 3.8.

## The benchmark

**KhmerTTS-Bench v1** has 320 sentences, each one checked by me as a native speaker:

- **Pure Khmer (80)**, **long (40)**: everyday sentences, questions, requests, longer multi-clause ones.
- **Numbers (60)**: prices, dates, times, phone numbers, fractions, football scores.
- **Mixed (70)**: Khmer with English words, the way people talk: "សូម upload រូបថតរបស់អ្នកនៅលើ website។"
- **Names (50)**: people, provinces, landmarks, foreign countries.
- **English (20)**: short tutor-style sentences.

One check I hadn't planned: some open models are trained on public Khmer speech datasets, so I compared every sentence against those datasets' text. One draft sentence was nearly identical to a training sentence of one of the models, which would have handed it an unfair advantage. I replaced it.

## How I measured

As before: a Khmer speech recognizer transcribes each clip, and I count the share of characters it gets wrong (character error rate, CER). English sentences are transcribed with Whisper and scored by word error rate instead. I report the median, because the recognizers occasionally get stuck repeating themselves, and a single runaway transcript would distort an average.

## Results

| | Pure Khmer | Names | All Khmer | English (WER) | Speed | License |
|---|---|---|---|---|---|---|
| Gemini 3.8 TTS | 0.0% | 2.0% | 2.2% | 0% | 0.84 (API) | commercial |
| VoxCPM2 (2.3B) | 5.9% | 3.6% | 6.4% | 0% | 1.86 (GPU) | Apache-2.0 |
| **sengtha (small VITS)** | **7.8%** | **7.8%** | 17.2% | ✗ | **0.16 (CPU)** | CC-BY-SA |
| Gemini 2.5 TTS | 13.0% | 10.0% | 15.4% | 0% | 1.09 (API) | commercial |
| OmniVoice | 10.6% | 7.8% | 17.3% | 0% | 1.43 (GPU) | non-commercial |
| MMS fine-tune (kto) | 22.2% | 26.0% | 31.5% | ✗ | 0.08 | non-commercial |
| MMS (Meta) | 23.1% | 30.0% | 35.3% | ✗ | 0.08 | non-commercial |
| Zio, MMS fine-tune (krorng) | 64–70% | 59–70% | ~70% | ✗ | 0.08–0.17 | CC-BY-SA / non-commercial |

Median CER; lower is better. Speed is generation time ÷ audio length, below 1 is faster than real time. "All Khmer" includes numbers written as digits; with my number-spelling frontend those errors mostly disappear.

## Four things I learned

**1. The newest Gemini is in a league of its own.**
Gemini 3.8 got a median of zero errors on pure Khmer, long sentences and numbers written as digits, and read every one of the 320 sentences. On intelligibility it beats every open model, including the 2.3B-parameter VoxCPM2. One quirk: my app's style prefix ("Say in a warm, friendly voice:") keeps Gemini 2.5 from answering the text, but Gemini 3.8 reads it out loud. It gets the bare text instead.

**2. A small open model still beats the Gemini my app uses.**
sengtha, a small community model that runs six times faster than real time on a laptop CPU, made fewer errors on pure Khmer than Gemini 2.5 (7.8% vs 13.0%). Its weak spots are numbers, which a text frontend fixes, and English words, which it mangles: it read "meeting" as "អ៊ីទីង". It won't match Gemini 3.8 on accuracy. What it offers instead: it runs offline on a laptop, costs nothing per sentence, and its weights are open.

**3. Gemini 2.5 goes silent on code-switching.**
Gemini 2.5's text-to-speech is built on a language model, and sometimes it treats the text as something to *answer* rather than read. Even with the same three retries my app uses, 15 of 320 sentences came back as no audio at all. One more retry recovered 7, but 8 mixed sentences, about 11% of the mixed category, failed all four attempts. They're everyday sentences like "WiFi នៅហាងកាហ្វេនេះលឿនណាស់។". In a tutoring app, that's a silent turn. Gemini 3.8 fixed this: no silent sentences.

**4. Spelling English in Khmer script helps, but isn't enough.**
Khmer speakers have conventional spellings for English loanwords (meeting → មីធីង). Rewriting the English words that way before synthesis raised my blind rating of sengtha on mixed sentences from 1.76 to 2.56 out of 5: better on 18 of 25 sentences, worse on none. But VoxCPM2 still scored 4.16. The rest of the gap is in the voice itself.

## What I'm doing about it

The plan from the first post is now running. VoxCPM2 can imitate sengtha's voice from an 8-second sample (voice similarity 0.96–0.98, as close as sengtha's own recordings). Overnight it read 1,500 Khmer and Khmer–English sentences in that voice. After automatic checks for voice, misread words and speed, 1,396 clips (1.2 hours) remain, and sengtha is now training on them on a Mac's GPU. That's distillation: the large model teaches the small one.

To keep myself honest, half of the benchmark's English words never appear in the training data. If the result only improves on words it has seen, that's memorizing, not learning.

I also built an English-to-Khmer-script converter from the CMU Pronouncing Dictionary. The hard part is that a Khmer vowel sounds different depending on which of two consonant series comes before it, so every syllable has to pick its consonant to fit its vowel. On approved spellings it never saw, it matches exactly 21% of the time and is 76% similar on average. That's good enough as a fallback for unknown words, not a replacement for a human-checked dictionary.

## Limitations

- One rater (me). Human scores are only comparable within one listening test.
- CER is not valid for the mixed category: the recognizer writes English inconsistently.
- Gemini 2.5's 8 sentences without audio are scored as silence, which counts against it in CER.
- CER measures whether the words are right, not whether the voice sounds native. That needs human ratings, which come next.

Code, sentences and results: [github.com/tangheng05/samleng-tts](https://github.com/tangheng05/samleng-tts). The benchmark is on Hugging Face: [hengthegoat/khmertts-bench-v1](https://huggingface.co/datasets/hengthegoat/khmertts-bench-v1).
