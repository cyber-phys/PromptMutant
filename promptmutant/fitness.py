import os
import openai
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from datasets import load_dataset
import torch
import logging
import random
from .llm import openai_chat, openai_instruct

def bert_encode(texts):
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def cosine_similarity_score(prompt, training_set, llm):
    seed = random.randint(0, 1000000)
    shuffled_set = training_set.shuffle(seed=seed)
    question_set = shuffled_set["question"][:5]
    answer_set = shuffled_set["answer"][:5]

    total_similarity = 0
    for i, question in enumerate(question_set):
        response = llm(prompt + "\n" + question)
        print("\033[33m" + response + "\033[0m")
        print("\033[32m" + answer_set[i] + "\033[0m")
        response_embedding = bert_encode([response])
        answer_embedding = bert_encode([answer_set[i]])
        similarity = cosine_similarity(response_embedding, answer_embedding)
        total_similarity += similarity[0][0]
        print(similarity)
        
    average_similarity = total_similarity / len(question_set)
    return average_similarity

if __name__ == "__main__":
    prompt = "Think out-loud while you answer the question"
    gsm8k_dataset = load_dataset("gsm8k", "main")
    cosine_similarity_score(prompt, gsm8k_dataset["train"])