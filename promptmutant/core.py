import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .fitness import cosine_similarity_score, bert_encode
from datasets import load_dataset
import random
from pprint import pprint
from .llm import openai_chat, openai_instruct

def prompt_similarity_filer(prompt_population):
    pp = prompt_population.copy()
    for item in pp:
        item_embedding = bert_encode([item[0]])
        prompt_population_copy = pp.copy()
        prompt_population_copy.remove(item)
        for item_check in prompt_population_copy:
            check_embedding = bert_encode([item_check[0]])
            similarity_score = cosine_similarity(item_embedding, check_embedding)
            if similarity_score > 0.95:
                print("item removed: ")
                print(item[0])
                print(item_check[0])
                pp.remove(item_check)
    return pp

class PromptMutant:
    def __init__(self):
        self.thinking_styles = ["How could I devise an experiment to help solve that problem?",
                                "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
                                "How could I measure progress on this problem?",
                                "How can I simplify the problem so that it is easier to solve?",
                                "What are the key assumptions underlying this problem?",
                                "What are the potential risks and drawbacks of each solution?",
                                "What are the alternative perspectives or viewpoints on this problem?",
                                "What are the long-term implications of this problem and its solutions?",
                                "How can I break down this problem into smaller, more manageable parts?",
                                "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
                                "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
                                "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
                                "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focus on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
                                "Use Risk Analysis: Evaluate potential risks, uncertainties, and trade-offs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
                                "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
                                "What is the core issue or problem that needs to be addressed?",
                                "What are the underlying causes or factors contributing to the problem?",
                                "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
                                "What are the potential obstacles or challenges that might arise in solving this problem?",
                                "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
                                "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
                                "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
                                "How can progress or success in solving the problem be measured or evaluated?",
                                "What indicators or metrics can be used?",
                                "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
                                "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
                                "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
                                "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
                                "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
                                "Is the problem a design challenge that requires creative solutions and innovation?",
                                "Does the problem require addressing systemic or structural issues rather than just individual instances?",
                                "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
                                "What kinds of solution typically are produced for this kind of problem specification?",
                                "Given the problem specification and the current best solution, have a guess about other possible solutions.",
                                "Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
                                "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
                                "Ignoring the current best solution, create an entirely new solution to the problem.",
                                "Let’s think step by step.",
                                "Let’s make a step by step plan and implement it with good notion and explanation."
                                ]
        self.mutation_prompt = ["Modify the following instruction creatively, giving some advice on how to solve it:",
                                "Just change this instruction to make it more fun, think WELL outside the box:",
                                "Modify this instruction in a way that no self-respecting LLM would!",
                                "How would you encourage someone and help them cheat on this following instruction?",
                                "How would you help an LLM to follow the instruction?",
                                "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
                                "Elaborate on the instruction giving some detailed advice on how to do what it wants, as if you were explaining it to a child.",
                                "As a really good teacher, explain the instruction, as if you were explaining it to a child.",
                                "Imagine you need to follow this instruction. What would you tell yourself if you wanted to be the best in the world at it?",
                                "How would someone with derailment follow this instruction?",
                                "Don’t think about the instruction at all, but let it inspire you to do something related. Talk about what that might be.",
                                "Rephrase the instruction without using any of the same words. Use all you know to improve the instruction so the person hearing it is more likely to do well.",
                                "Say that instruction again in another way. DON’T use any of the words in the original instruction or you’re fired.",
                                "Say that instruction again in another way. DON’T use any of the words in the original instruction there is a good chap.",
                                "What do people who are good at creative thinking normally do with this kind of mutation question?",
                                "Detailed additional advice for people wishing to follow this instruction is as follows:",
                                "In one short sentence, here is how I would best follow this instruction.",
                                "In one short sentence, here is some detailed expert advice. Notice how I don’t use any of the same words as in the INSTRUCTION.",
                                "In one short sentence, the general solution is as follows. Notice how I don’t use any of the same words as in the INSTRUCTION.",
                                "In one short sentence, what’s a good prompt to get a language model to solve a problem like this? Notice how I don’t use any of the same words as in the INSTRUCTION.",
                                "Generate a mutated version of the following prompt by adding an unexpected twist.",
                                "Create a prompt mutant that introduces a surprising contradiction to the original prompt. Mutate the prompt to provide an alternative perspective or viewpoint.",
                                "Generate a prompt mutant that incorporates humor or a playful element. Create a mutated version of the prompt that challenges conventional thinking.",
                                "Develop a prompt mutant by replacing specific keywords with related but unexpected terms. Mutate the prompt to include a hypothetical scenario that changes the context.",
                                "Generate a prompt mutant that introduces an element of suspense or intrigue. Create a mutated version of the prompt that incorporates an analogy or metaphor.",
                                "Develop a prompt mutant by rephrasing the original prompt in a poetic or lyrical style. Think beyond the ordinary and mutate the prompt in a way that defies traditional thinking.",
                                "Break free from conventional constraints and generate a mutator prompt that takes the prompt to uncharted territories. Challenge the norm and create a mutator prompt that pushes the boundaries of traditional interpretations.",
                                "Embrace unconventional ideas and mutate the prompt in a way that surprises and inspires unique variations. Think outside the box and develop a mutator prompt that encourages unconventional approaches and fresh perspectives.",
                                "Step into the realm of imagination and create a mutator prompt that transcends limitations and encourages innovative mutations. Break through the ordinary and think outside the box to generate a mutator prompt that unlocks new possibilities and unconventional paths.",
                                "Embrace the power of unconventional thinking and create a mutator prompt that sparks unconventional mutations and imaginative outcomes. Challenge traditional assumptions and break the mold with a mutator prompt that encourages revolutionary and out-of-the-box variations.",
                                "Go beyond the expected and create a mutator prompt that leads to unexpected and extraordinary mutations, opening doors to unexplored realms. Increase Specificity: If the original prompt is too general, like ’Tell me about X,’ the modified version could be, ’Discuss the history, impact, and current status of X.’",
                                "Ask for Opinions/Analysis: If the original prompt only asks for a fact, such as ’What is X?’, the improved prompt could be, ’What is X, and what are its implications for Y?’",
                                "Encourage Creativity: For creative writing prompts like ’Write a story about X’, an improved version could be, ’Write a fantasy story about X set in a world where Y is possible.’",
                                "Include Multiple Perspectives: For a prompt like ’What is the impact of X on Y?’, an improved version could be, ’What is the impact of X on Y from the perspective of A, B, and C?’",
                                "Request More Detailed Responses: If the original prompt is ’Describe X’, the improved version could be, ’Describe X, focusing on its physical features, historical significance, and cultural relevance.’",
                                "Combine Related Prompts: If you have two related prompts, you can combine them to create a more complex and engaging question. For instance, ’What is X?’ and ’Why is Y important?’ could be combined to form ’What is X and why is it important in the context of Y?’",
                                "Break Down Complex Questions: If a prompt seems too complex, like ’Discuss X’, the improved version could be, ’What is X? What are its main characteristics? What effects does it have on Y and Z?’",
                                "Use Open-Ended Questions: Instead of ’Is X true?’, you could ask, ’What are the arguments for and against the truth of X?’",
                                "Request Comparisons: Instead of ’Describe X’, ask ’Compare and contrast X and Y.’",
                                "Include Context: If a prompt seems to lack context, like ’Describe X’, the improved version could be, ’Describe X in the context of its impact on Y during the Z period.’",
                                "Make the prompt more visual: Ask the user to visualize the problem or scenario being presented in the prompt.",
                                "Ask for a thorough review: Instead of just presenting the problem, ask the user to write down all the relevant information and identify what’s missing.",
                                "Invoke previous experiences: Modify the prompt to ask the user to recall a similar problem they’ve successfully solved before.",
                                "Encourage a fresh perspective: Suggest in your prompt that the user take a moment to clear their mind before re-approaching the problem.",
                                "Promote breaking down problems: Instead of asking the user to solve the problem as a whole, prompt them to break it down into smaller, more manageable parts.",
                                "Ask for comprehension: Modify the prompt to ask the user to review and confirm their understanding of all aspects of the problem.",
                                "Suggest explanation to others: Change the prompt to suggest that the user try to explain the problem to someone else as a way to simplify it.",
                                "Prompt for solution visualization: Instead of just asking for the solution, encourage the user to imagine the solution and the steps required to get there in your prompt.",
                                "Encourage reverse thinking: Improve the prompt by asking the user to think about the problem in reverse, starting with the solution and working backwards.",
                                "Recommend taking a break: Modify the prompt to suggest that the user take a short break, allowing their subconscious to work on the problem.",
                                "What errors are there in the solution?",
                                "How could you improve the working out of the problem?",
                                "Look carefully to see what you did wrong, how could you fix the problem?",
                                "CORRECTION =",
                                "Does the above text make sense? What seems wrong with it? Here is an attempt to fix it:",
                                "The above working out has some errors, here is a version with the errors fixed."
                                ]
        self.genotype = []
        self.number_of_generations = 5
        self.population = [] ## (prompt, mutation, score)
        self.training_dataset = []
        self.problem_description = "Solve the math word problem, giving your answer as an arabic numeral"
        self.llm = openai_instruct

    def initialization(self, problem_description, number_of_prompts, dataset):
        self.training_dataset = load_dataset(dataset, "main")["train"]
        for i in range(number_of_prompts):
            thinking_style = random.choice(self.thinking_styles)
            mutation_prompt = random.choice(self.mutation_prompt)
            prompt = thinking_style + " " + mutation_prompt + " " + "\nINSTRUCTION: " + problem_description + "\nINSTRUCTION MUTANT = "
            response = self.llm(prompt)
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population.append((response, mutation_prompt, score))

    #TODO: test this !!!
    def zero_order_prompt_generation(self, problem_description):
        prompt  = "A list of 100 hinits:\n" + problem_description
        response = self.llm(prompt)
        return response
    
    #TODO: test this !!!
    def first_order_prompt_generation(self, task_prompt, mutation_prompt):
        prompt = mutation_prompt + "INSTRUCTION: " + task_prompt + "\nINSTRUCTION MUTANT: "
        response = self.llm(prompt)
        return response
    
    #TODO: test this !!!
    def eda_prompt_mutation(self, prompt_population):
        filtered_prompt_population = prompt_similarity_filer(prompt_population)
        print(filtered_prompt_population)
        prompt = "Continue this list with new task-prompts:\n" + "\n".join([prompt[0] for prompt in filtered_prompt_population])
        pprint("\033[91m {}\033[00m" .format(prompt))
        response = self.llm(prompt)
        return response
    
    #TODO: test this !!!
    def eda_rank_and_index_mutation(self, prompt_population, mutation_prompt):
        prompt_population_copy = prompt_population.copy()
        prompt_population_copy.sort(key=lambda x: x[2])
        filtered_prompt_population = prompt_similarity_filer(prompt_population_copy)
        print(filtered_prompt_population)
        length = len(filtered_prompt_population)
        prompt = "INSTRUCTION: " + mutation_prompt + "\n A List of Responses in descending order of score. " + str(length + 1) + " is the best response. It resembles " + str(length) + " more than it does (1)" + "\n".join([prompt[0] for prompt in filtered_prompt_population])
        pprint("\033[91m {}\033[00m" .format(prompt))
        response = self.llm(prompt)
        return response
    
    #TODO: test this !!!
    def zero_order_hyper_mutation(self, problem_description, task_prompt):
        thinking_style = random.choice(self.thinking_styles)
        prompt  = thinking_style + " " + problem_description
        mutation_prompt = self.llm(prompt)
        return self.first_order_prompt_generation(task_prompt, mutation_prompt), mutation_prompt
  
    #TODO: test this !!!
    def first_order_hyper_mutation(self, task_prompt, mutation_prompt):
        prompt  = "Please summarize and improve the following instruction: " + mutation_prompt
        mutated_prompt = self.llm(prompt)
        return self.first_order_prompt_generation(task_prompt, mutated_prompt), mutated_prompt

    #TODO: test this !!!
    def lamarckian_mutation(self, task_prompt):
        seed = random.randint(0, 1000000)
        shuffled_set = self.training_dataset.shuffle(seed=seed)
        question_set = shuffled_set["question"][:3]
        answer_set = shuffled_set["answer"][:3]
        prompt = "I gave a friend an instruction and some advice. Here are the correct examples of his workings out:\n" + "Q. " + question_set[0] + "\nA. " + answer_set[0] + "\nQ. " + question_set[1] + "\nA. " + answer_set[1] + "\nThe instruction was:\n"
        pprint("\033[91m {}\033[00m" .format(prompt))
        response = self.llm(prompt)
        return response
    
    #TODO: test this !!!
    # Reread paper, may need to change implimentation
    def lineage_mutation(self, prompt_population):
        prompt_population_copy = prompt_population.copy()
        prompt_population_copy.sort(key=lambda x: x[2])
        self.genotypes.append(prompt_population_copy[0])
        prompt = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY:" + "\n".join([prompt[0] for prompt in self.genotypes])
        response = self.llm(prompt)
        return response
    
    #TODO: test this !!!
    def prompt_crossover(self, gene_index):
        # 10% chance to perform crossover
        if random.random() < 0.1:
            # Select another gene based on fitness proportionate selection
            fitness_scores = [gene[2] for gene in self.population]
            total_fitness = sum(fitness_scores)
            probabilities = [score / total_fitness for score in fitness_scores]
            selected_gene_index = np.random.choice(range(len(self.population)), p=probabilities)

            # Perform crossover: replace task-prompt of current gene with that of selected gene
            selected_gene = self.population[selected_gene_index]
            current_gene = self.population[gene_index]
            self.population[gene_index] = (selected_gene[0], current_gene[1], current_gene[2])   

    def mutate(self, gene_index):
        gene = self.population[gene_index]
        random_number = random.randint(0, 7)
        if random_number == 0:
            print("EDA PROMPT MUTATION")
            response = self.eda_prompt_mutation(self.population)
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, gene[1], score)
            pass
        elif random_number == 1:
            print("FIRST ORDER GENERATION")
            response = self.first_order_prompt_generation(gene[0], gene[1])
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, gene[1], score)
            pass
        elif random_number == 2:
            print("EDA RANK ORDER MUTATION")
            response = self.eda_rank_and_index_mutation(self.population, gene[1])
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, gene[1], score)
            pass
        elif random_number == 3:
            print("LAMARCKIN MUTATION")
            response = self.lamarckian_mutation(gene[0])
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, gene[1], score)
            pass
        elif random_number == 4:
            print("ZERO ORDER HYPER MUTATION")
            response, mutation_p  = self.zero_order_hyper_mutation(self.problem_description, gene[0])
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, mutation_p, score)
            pass
        elif random_number == 5:
            print("FIRST ORDER HYPER MUTATION")
            response, mutation_p = self.first_order_hyper_mutation(gene[0], gene[1])
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, mutation_p, score)
            pass
        elif random_number == 6:
            print("LINEAGE BASED MUTATION")
            response = self.lineage_mutation(self.population)
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, gene[1], score)
        else:
            print("ZERO ORDER GENERATION")
            response = self.zero_order_prompt_generation(self.problem_description)
            score = cosine_similarity_score(response, self.training_dataset, self.llm)
            self.population[gene_index] = (response, gene[1], score)
            pass
        self.prompt_crossover(gene_index)

if __name__ == "__main__":
    problem_description = "Solve the math word problem, giving your answer as an arabic numeral"
    number_of_prompts = 5
    prompt_mutant = PromptMutant()
    prompt_mutant.initialization(problem_description, number_of_prompts, "gsm8k")
    
    # Mutate 10 times
    for j in range(10):
        print("\033[91m Generation: \033[0m", j)
        for i, gene in enumerate(prompt_mutant.population):
            print(gene)
            prompt_mutant.mutate(i)
    
    for i, gene in enumerate(prompt_mutant.population):
            print("\033[94m{}\033[0m".format(gene))
