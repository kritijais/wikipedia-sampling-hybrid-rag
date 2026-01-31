from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Generator using FLAN-T5
class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    # Generate answer given query and contexts
    def generate(self, query, contexts, max_length=300) -> str:
        prompt = f"""You are an expert Q&A assistant. Your task is to provide accurate, 
comprehensive answers based ONLY on the provided context.

INSTRUCTIONS:
1. Read the context carefully and identify relevant information
2. Answer the question directly and concisely
3. Use ONLY information from the context
4. Do NOT use external knowledge or make up information
5. If the answer is not in the context, say "I cannot answer this from the provided context"
6. Be factually accurate and avoid hallucination
7. Provide a complete answer with supporting details when available
8. Structure your answer clearly

IMPORTANT: 
- Never hallucinate or invent information
- Stay faithful to the context
- If unsure, say you don't know

CONTEXT:
"""
        for i, ctx in enumerate(contexts[:10], 1):  # Limit to 10 contexts
            # Truncate to avoid token limit
            ctx_text = ctx[:300] if len(ctx) > 300 else ctx
            prompt += f"\n[Source {i}]\n{ctx_text}\n"
        
        prompt += f"\n\nQUESTION: {query}\n\nANSWER: "

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Larger input window
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,           # Longer output (was 200)
                    min_length=20,                   # Force meaningful output
                    num_beams=4,                     # Better beam search
                    length_penalty=2.0,              # Encourage longer, more complete answers
                    early_stopping=True,
                    temperature=0.7,                 # More deterministic (not random)
                    top_p=0.9,
                    do_sample=False,                 # Greedy decoding for consistency
                    no_repeat_ngram_size=2           # Avoid repetition
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            answer = answer.replace("<pad>", "").strip()

            if answer.startswith("Context:"):
                answer = answer.replace("Context:", "").strip()
            
            return answer

        except Exception as e:
            print(f"Error in generation: {e}")
            return "I was unable to generate an answer due to a technical error."

        # for c in contexts:
        #     prompt += c + "\n"
        # prompt += f"\nQuestion: {query}\nAnswer:"

        # inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        # outputs = self.model.generate(**inputs, max_length=200)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
