"""Sentiment analysis, English->German translation, extractive QA."""
import hf_env  # loads HF_TOKEN from .env only
import os, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering)

dev = "cuda" if torch.cuda.is_available() else "cpu"
load = lambda cls, name: (AutoTokenizer.from_pretrained(name), cls.from_pretrained(name).to(dev).eval())

SENT = "sentiment_model" if os.path.isdir("sentiment_model") else "distilbert-base-uncased-finetuned-sst-2-english"
s_tok, s_model = load(AutoModelForSequenceClassification, SENT)
t_tok, t_model = load(AutoModelForSeq2SeqLM, "Helsinki-NLP/opus-mt-en-de")
q_tok, q_model = load(AutoModelForQuestionAnswering, "distilbert-base-cased-distilled-squad")

@torch.no_grad()
def sentiment(text):
    x = s_tok(text, return_tensors="pt", truncation=True).to(dev)
    p = s_model(**x).logits.softmax(-1)[0]
    i = int(p.argmax()); return s_model.config.id2label[i], float(p[i])

@torch.no_grad()
def translate(text):
    x = t_tok(text, return_tensors="pt").to(dev)
    return t_tok.decode(t_model.generate(**x, num_beams=4, max_new_tokens=128)[0], skip_special_tokens=True)

@torch.no_grad()
def answer(question, context):
    x = q_tok(question, context, return_tensors="pt", truncation=True).to(dev)
    o = q_model(**x)
    s, e = int(o.start_logits.argmax()), int(o.end_logits.argmax())
    return q_tok.decode(x["input_ids"][0][s:e + 1], skip_special_tokens=True)

def run_samples():
    print("== Sentiment (model:", SENT, ")")
    for t in ["A wonderful, moving film.", "Dull plot and terrible acting.", "It was fine, I guess."]:
        l, c = sentiment(t); print(f"{l:9} {c:.3f}  {t}")
    print("\n== Translation EN->DE")
    for t in ["Deep learning models can understand natural language.", "Where is the nearest train station?"]:
        print(t, "->", translate(t))
    print("\n== Question Answering")
    ctx = ("The Transformer architecture was introduced in 2017 in the paper 'Attention Is All You Need' "
           "by researchers at Google. It relies on self-attention instead of recurrence, which allows "
           "much more parallelisation during training.")
    for q in ["When was the Transformer introduced?", "What does it rely on instead of recurrence?"]:
        print(f"Q: {q}\nA: {answer(q, ctx)}")


HELP = """Commands:
  sentiment <text>      classify text as POSITIVE / NEGATIVE
  translate <text>      English -> German
  context <passage>     set the passage for question answering
  ask <question>        answer a question from the current passage
  help | quit"""

def demo():
    import readline  # noqa: F401  (line editing)
    print(HELP); context = None
    while True:
        try: line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt): break
        cmd, _, arg = line.partition(" "); cmd = cmd.lower(); arg = arg.strip()
        if cmd in ("quit", "exit", "q"): break
        elif cmd == "help": print(HELP)
        elif not arg: print("Missing text. Type 'help'.")
        elif cmd == "sentiment":
            l, c = sentiment(arg); print(f"{l} ({c:.1%})")
        elif cmd == "translate": print(translate(arg))
        elif cmd == "context": context = arg; print("Passage set. Now use: ask <question>")
        elif cmd == "ask":
            print(answer(arg, context) if context else "Set a passage first: context <passage>")
        else: print("Unknown command. Type 'help'.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="interactive query mode")
    demo() if ap.parse_args().demo else run_samples()
