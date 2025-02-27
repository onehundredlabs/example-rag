from langchain_ollama import OllamaEmbeddings
from langchain.evaluation import load_evaluator

embeddings = OllamaEmbeddings(model="deepseek-r1:14b")

evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
words = ("apple", "iphone")

result = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])

print(f"Comparing ({words[0]}, {words[1]}): {result}")
