import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from transformers import BertModel, BertTokenizer
from fitness import cosine_similarity_score, bert_encode
from datasets import load_dataset
import random

# PromptDNA = Component("PromptDNA")

# @PromptDNA.init_state
# def init_state():
#     openai.api_key = os.getenv("OPENAI_API_KEY")

#     prompteng_system_message = {
#         "role": "system",
#         "content": "You are a prompt engineer, and your goal is to generate the best prompt template for a large language model to perform some task.",
#     }

#     return {
#         "prompteng_system_message": prompteng_system_message,
#         "thinking_styles": ["Let's think step by step", "How would a teacher solve this?", "Approach this systematically", "Take a deep breath"],
#         "task_prompts": [],
#         "genotypes": [],
#         "number_of_generations": 5
#     }

def prompt_similarity_filer(prompt_population):
    # Convert prompts to BERT embeddings
    prompt_embeddings = [bert_encode(prompt) for prompt in prompt_population]

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(prompt_embeddings)

    # Create a mask for entries that are more than 0.95 similar
    mask = similarity_matrix > 0.95

    # Apply mask to similarity matrix and get indices of prompts to remove
    indices_to_remove = np.where(mask.any(axis=0))

    # Remove similar prompts
    filtered_prompt_population = [prompt for i, prompt in enumerate(prompt_population) if i not in indices_to_remove]
    filtered_prompt_embeddings = [embedding for i, embedding in enumerate(prompt_embeddings) if i not in indices_to_remove]

    return filtered_prompt_population, filtered_prompt_embeddings

class PromptMutant:
    def __init__(self):
        self.thinking_styles = ["Let's think step by step", "How would a teacher solve this?", "Approach this systematically", "Take a deep breath"]
        self.mutation_prompt = [""]
        self.task_prompts = []
        self.genotypes=[]
        self.number_of_generations = 5
        self.population = [] ## (prompt, mutation, score)
        self.training_dataset = load_dataset("gsm8k", "main")["train"]

    def initialization(self, problem_description, number_of_prompts):
        for i in range(number_of_prompts):
            thinking_style = random.choice(self.thinking_styles)
            mutation_prompt = random.choice(self.mutation_prompt)
            prompt = thinking_style + " " + mutation_prompt + " " + "\nINSTRUCTION: " + problem_description + "\nINSTRUCTION MUTANT = "
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion.choices[0].message["content"]
            score = cosine_similarity_score(response, self.training_dataset)
            self.population.append((response, mutation_prompt, score))


    #TODO: test this !!!
    def zero_order_prompt_generation(problem_description):
        prompt  = "A list of 100 hinits:\n" + problem_description
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message
    
    #TODO: test this !!!
    def first_order_prompt_generation(task_prompt):
        mutation_prompt = "Say that instruction again in another way. DON’T use any of the words in the original instruction there’s a good chap."
        prompt = mutation_prompt + "INSTRUCTION: " + task_prompt
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "INSTRUCTION MUTANT: "}
            ]
        )
        return completion.choices[0].message
    
    #TODO: test this !!!
    def eda_prompt_mutation(prompt_population):
        filtered_prompt_population, filtered_prompt_embeddings = prompt_similarity_filer(prompt_population)
        prompt = "Continue this list with new task-prompts:\n" + "\n".join(filtered_prompt_population)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message
        
    # def eda_rank_and_index_mutation(prompt_population):
    #     filtered_prompt_population, filtered_prompt_embeddings = prompt_similarity_filer(prompt_population)

    #     # Convert prompts to BERT embeddings
    #     prompt_embeddings = [bert_encode(prompt) for prompt in prompt_population]

    #     # Compute cosine similarity matrix
    #     similarity_matrix = cosine_similarity(prompt_embeddings)


    #     return completion.choices[0].message


    # def initialize_promptbreeder(self, prompt, problem_description):
    #     for style in self.thinking_styles:
    #         self.task_prompts.append(f"{style}: {problem_description}")

    # def mutate_prompts(task_prompts, database_path):
    #     return random.choice(self.mutation_prompts)
    #     # Use the database to evaluate and mutate the prompts
    #     # Return evolved prompts
    #     return evolved_prompts

    # def hyper_mutate(self):
    #     for i, mutation_prompt in enumerate(self.mutation_prompts):
    #         self.mutation_prompts[i] = self.mutate_prompt(mutation_prompt)

    # def evaluate_fitness(prompt, database_path):
    #     # Evaluate the fitness of a prompt using the provided database
    #     # Return a fitness score
    #     return score
    
    #     def mutate_prompt(self, prompt):
    #     mutation = random.choice(self.mutation_prompts)
    #     if mutation == "Make it fun":
    #         return f"Fun version: {prompt}"
    #     elif mutation == "Add a twist":
    #         return f"Twisted: {prompt} but with a twist!"
    #     elif mutation == "Simplify":
    #         return f"Simplified: {prompt.split(':')[1]}"

if __name__ == "__main__":
    problem_description = "Solve the math word problem, giving your answer as an arabic numeral"
    number_of_prompts = 5
    prompt_mutant = PromptMutant()
    # Call the initialization method
    prompt_mutant.initialization(problem_description, number_of_prompts)

    # Print the population
    for prompt in prompt_mutant.population:
        print(prompt)

