import argparse
from .core import PromptMutant

def main():
    parser = argparse.ArgumentParser(description="Run Promptbreeder")
    parser.add_argument("-prompt", type=str, help="Initial prompt for Promptbreeder")
    parser.add_argument("-dataset", type=str, help="Name of dataset on huggingface for evaluation")
    parser.add_argument("-nPrompts", type=int, help="Number of prompts in the population")
    parser.add_argument("-nMutations", type=int, help="Number of times to run mutation")
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

    prompt_mutant = PromptMutant()
    prompt_mutant.initialization(args.prompt, args.nPrompts, args.dataset)

    # Mutate n times
    for j in range(args.nMutations):
        print("\033[91m Generation: \033[0m", j)
        for i, gene in enumerate(prompt_mutant.population):
            print(gene)
            prompt_mutant.mutate(i)
    
    # Print Evolved Prompts
    for i, gene in enumerate(prompt_mutant.population):
            print("\033[94m{}\033[0m".format(gene))

if __name__ == "__main__":
    main()