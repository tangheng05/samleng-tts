# How good is open Khmer text-to-speech in 2026?

*6 November 2026 · Part 1 of the Samleng-TTS write-ups · Next: [Every open Khmer voice, measured](02-every-open-khmer-tts-model.md) · [Project README](../../README.md)*

Khmer is spoken by about 17 million people, yet there is no shared way to measure how well a machine reads it aloud. So I built a small benchmark, ran two open models through it, and rated the hardest sentences myself as a native speaker. Here's what I found.

## The test

100 sentences in four groups, all checked by a native speaker:

- **Pure Khmer (40):** everyday sentences, questions, place names, a proverb.
- **Long (10):** sentences with several clauses.
- **Numbers (25):** prices in riel and dollars, dates, times, phone numbers, percentages, temperatures.
- **Mixed (25):** Khmer with English words, the way people actually talk: "ខ្ញុំភ្លេច password របស់ account ខ្ញុំ។"

Two open models:

- **MMS-TTS Khmer** (Meta, 36M parameters): small and fast, runs on a laptop CPU.
- **VoxCPM2** (OpenBMB, 2.3B parameters): a large multilingual model.

## How I measured

1. **Intelligibility:** I transcribe each clip with a Khmer speech recognizer ([Qwen3-ASR-0.6B-Khmer](https://huggingface.co/seanghay/Qwen3-ASR-0.6B-Khmer)) and count how many characters differ from the original sentence (character error rate, CER).
2. **Human ratings:** for the mixed sentences, I listened blind (shuffled, with the voice hidden) and scored each clip from 1 (bad) to 5 (sounds native).
3. **Speed:** generation time divided by audio length. Below 1 means faster than real time.

## Results

| | MMS (36M) | VoxCPM2 (2.3B) |
|---|---|---|
| Pure Khmer, median CER | 23.1% | 2.7% |
| Numbers, median CER (with text frontend) | 26.9% | 0.0% |
| Mixed, human score (1–5) | 1.56 | 3.72 |
| Speed (real-time factor, median) | 0.08 | 2.1 |

The large model is far better, and VoxCPM2 won on all 25 mixed sentences. But it is also far slower: it takes about twice as long to speak a sentence as the sentence lasts, while MMS is more than ten times faster than real time. **Nothing open today is both good and fast for Khmer.** That gap is what this project is about.

## Three things I learned

**1. Numbers are a text problem, not a voice problem.**
Given "$12.50", VoxCPM2 garbled it. Spelling numbers out as Khmer words *before* synthesis (ដប់ពីរដុល្លារហាសិបសេន) brought its median error on number sentences from 8% to 0%, and cut the small model's in half. I built this on [Tha](https://github.com/seanghay/tha), an open Khmer normalizer, and fixed a few cases it missed: "25,000" was read as "25 comma 000", "38°C" wasn't read at all, and "14:00" came out as "fourteen zero" instead of ម៉ោងពីររសៀល.

**2. Automatic scores break on code-switching.**
On mixed sentences, the speech recognizer writes English words inconsistently, sometimes in Latin letters ("Facebook") and sometimes in Khmer script (អ៊ីមែល). So the error rate says almost nothing: for VoxCPM2 it didn't predict my scores at all (correlation −0.06). I also tried UTMOSv2, a popular model that predicts how natural speech sounds. It couldn't tell the two systems apart, even though I rated them more than two points apart. It was trained on English, Chinese and Japanese, and published studies find the same problem in other low-resource languages. **For Khmer code-switching, you still need native ears.**

**3. Big models aren't consistent.**
I rated some clips twice without knowing it. For the same audio, my scores matched most of the time. But VoxCPM2 speaking the same sentence twice produced clips I scored up to two points apart. Its quality varies from take to take, which matters if you want to use it to teach a smaller model.

## Limitations

- One rater (me), and only 25 rated sentences. The confidence intervals are in the repo, but this needs more native listeners.
- Human scores are only comparable within one listening test.
- The CER judge is itself a model and sometimes mishears.

## What's next

Use the large model as a teacher to train a small model that runs in real time on ordinary hardware, and grow the benchmark to about 300 sentences. Everything (code, sentences, results) is open at [github.com/tangheng05/samleng-tts](https://github.com/tangheng05/samleng-tts).
