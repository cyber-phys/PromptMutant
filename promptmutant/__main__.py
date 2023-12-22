import argparse
from .core import PromptMutant
from pprint import pprint
from IPython.display import clear_output
from time import sleep
import sqlite3
import os
from datetime import datetime
import sys
import signal

def initialize_database():
    db_exists = os.path.exists('promptbreeder.db')
    conn = sqlite3.connect('promptbreeder.db')  # This will create the file if it does not exist
    if not db_exists:
        cursor = conn.cursor()

        # Execute SQL commands to create tables if the database was not found
        cursor.executescript("""
        -- Runs Table
        CREATE TABLE Runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            init_prompt TEXT,
            number_prompts INTEGER NOT NULL,
            number_mutations INTEGER NOT NULL
        );

        -- Prompts Table
        CREATE TABLE Prompts (
            prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            generation INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            run_id INTEGER NOT NULL,
            mutation_prompt_id INTEGER NOT NULL,
            FOREIGN KEY (run_id) REFERENCES Runs (run_id),
            FOREIGN KEY (mutation_prompt_id) REFERENCES MutationPrompts (mutation_prompt_id)
        );

        -- Mutation Prompts Table
        CREATE TABLE MutationPrompts (
            mutation_prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            generation INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            run_id INTEGER NOT NULL,
            FOREIGN KEY (run_id) REFERENCES Runs (run_id)
        );

        -- Parent-Child Relationship for Prompts
        CREATE TABLE PromptGenealogy (
            child_id INTEGER NOT NULL,
            parent_id INTEGER NOT NULL,
            FOREIGN KEY (child_id) REFERENCES Prompts (prompt_id),
            FOREIGN KEY (parent_id) REFERENCES Prompts (prompt_id)
        );

        -- Parent-Child Relationship for Mutation Prompts
        CREATE TABLE MutationPromptGenealogy (
            child_id INTEGER NOT NULL,
            parent_id INTEGER NOT NULL,
            FOREIGN KEY (child_id) REFERENCES Prompts (prompt_id),
            FOREIGN KEY (parent_id) REFERENCES Prompts (prompt_id)
        );

        -- Fitness Scores Table
        CREATE TABLE FitnessScores (
            score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            score REAL NOT NULL,
            evaluation_criteria TEXT,
            scored_at TIMESTAMP NOT NULL,
            FOREIGN KEY (prompt_id) REFERENCES Prompts (prompt_id),
            FOREIGN KEY (run_id) REFERENCES Runs (run_id)
        );

        -- Mutations Table
        CREATE TABLE Mutations (
            mutation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP NOT NULL
        );

        -- Linking Mutations to Prompts
        CREATE TABLE PromptMutations (
            prompt_id INTEGER NOT NULL,
            mutation_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            FOREIGN KEY (prompt_id) REFERENCES Prompts (prompt_id),
            FOREIGN KEY (mutation_id) REFERENCES Mutations (mutation_id),
            FOREIGN KEY (run_id) REFERENCES Runs (run_id)
        );

        -- Thinking Styles Table (optional, if thinking styles are predefined and stored)
        CREATE TABLE ThinkingStyles (
            style_id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT
        );
        """)
        conn.commit()
    
    conn.close()

def insert_run(dataset_name, start_time, end_time, prompt, nPrompts, nMutations):
    conn = sqlite3.connect('promptbreeder.db')
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO Runs (dataset_name, start_time, end_time, init_prompt, number_prompts, number_mutations)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (dataset_name, start_time, end_time, prompt, nPrompts, nMutations))

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def get_run(run_id):
    conn = sqlite3.connect('promptbreeder.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM Runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()

    conn.close()
    return run

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description="Run Promptbreeder")
    parser.add_argument("-prompt", type=str, help="Initial prompt for Promptbreeder")
    parser.add_argument("-dataset", type=str, help="Name of dataset on huggingface for evaluation")
    parser.add_argument("-nPrompts", type=int, help="Number of prompts in the population")
    parser.add_argument("-nMutations", type=int, help="Number of times to run mutation")
    parser.add_argument("-runId", type=int, help="Existing run ID to load", default=None)
    args = parser.parse_args()

    print("""                                               
                 .:;+:.                           
               :x+;+;+xx;.                        
             .+x:      :xx:                       
             ;x;    .   :xx;                       
              ;+:. ..     :x+.                      
                `|||/      +Xx                     
                        ,:xxx                     
                    .:+xxX    ::::.              
                   +xX+:    .+xxxxxxx.            
                 .xxx.     +xx+:   :xx+           
                  +xx:    :xx:     :xx+           
                   +xx+:  :xx:;   xxx;            
                     :xxx::xxxx   :::              
                 :;;:  :xxx:xxx:  `/                 
                :x+.;x  ;Xx;:xxx                  
         :+x+:  .;  ;:;.+xxx:xxx   .xxxx;.        
        ;x:.:xx  \  :xx+.+xxxxXxx .+x+..:+x        
        :+   :x;    .xx+xxxxXxxx.+x;    ;x        
        .    .xx:    +xxxxxxxxxxxx+     ;;        
             ::xxxxxxxxxxxxxxxxxx+.    ./            
             .;:+xxxxxxxxxxxxxxxxx;               
       .:+:::xxxxxxxxxxxxxxxxxx+;xxx;             
    .;++::+xxxxxxxXxx++xXx;:xXx++:++xx+...,        
  .::::::::::::::::::::::::::::::::::::::::..
  ..............PROMPT MUTANT.................""")
    
    initialize_database()
    # Check if a runId is provided, if not, create a new run
    if args.runId is None:
      current_datetime = datetime.now()
      timestamp_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
      run_id = insert_run(args.dataset, timestamp_str, None, args.prompt, args.nPrompts, args.nMutations)
    else:
        run_id = args.runId
    
    run_details = get_run(run_id)
    print(run_details)

    prompt_mutant = PromptMutant()
    prompt_mutant.initialization(run_details[0], run_details[4], run_details[5], run_details[1])

    # # Mutate n times
    for j in range(run_details[6]):
        print("\033[91m Generation: \033[0m", j)
        for i, gene in enumerate(prompt_mutant.read_prompts_from_db(j, prompt_mutant.run_id)):
            prompt_mutant.mutate(i,j)
        clear_output(wait=True)

if __name__ == "__main__":
    main()