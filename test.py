from transformers import pipeline

pipe = pipeline("text2text-generation", model="declare-lab/flan-alpaca-large")

question = "Explain flight price prediction project"
result = pipe(question, max_length=100)[0]['generated_text']
print(result)