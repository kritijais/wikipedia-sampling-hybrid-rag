from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Generator using FLAN-T5
class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # Generate answer given query and contexts
    def generate(self, query, contexts):
        prompt = "Answer the question using the context.\n\n"
        for c in contexts:
            prompt += c + "\n"
        prompt += f"\nQuestion: {query}\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
