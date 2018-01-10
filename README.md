# udacity_aind_term1
This repository stores the 4 projects I did in term 1 of Udacity's Artifical intelligence Nano Degree.

## Project 1: Sudoku, under AIND-sudoku
In this project, I solved Sudoku using techniques like elimination, "only_choice", "naked_twins", and search to solve a Sudoku problem efficiently, if a solution exists. This problem introduces a key concept in AI called "Constraint propagation".

## Project 2: An adversarial Search Agent, under AIND-Isolation
In this project, I built different game agents to play the game "isolation".
The first agent uses minimax search, the second uses minimax search with alpha-beta pruning, a technique that greatly reduces the size of search space. The third agent is based on the second agent, except that it uses iterative deepening. 
Different heuristic functions were used to improve the performance of the game agent. It turned out that a heuristic function that uses the weighted sum of two quantities, the difference between the number of moves of the current player and that of its opponent, and the square of the distance from the center of the board, yielded the best performance. The performance of different methods are compared in ./AIND-Isolation/heuristic_analysis.pdf.

## Project 3: A domain-independent planner, under AIND-Planning
In this project, I solved several problems in classical PDDL (Planning Domain Definition Language) for the air cargo domain.
I first solved the air cargo planning problem using non-heuristic planning methods including breadth-first, depth-first, and uniform cost search (Dijkstra algorithm). 
Then I solved the problem using A* with a heuristic that estimates the minimum number of actions that must be carried out to reach the goal state from the current state (similar to the straight distance from a location to another location in Google map), by igoring the preconditions.
Finally, I solved the same problem using a special data structure called plan graph. Unlike trees or graphs, plan graph is polynoimal instead of exponential, suitable for solving large-scale planning problems. In this part, I used A* with another heuristic function called level sum, which estimates the sum of level costs of all goals in the problem.
The performance of different methods are compared in ./AIND-Planning/heuristic_analysis.pdf.


## Project 3: Use HMMs to recognize American Sign Language, under AIND-Recognizer
In this project, I built hidden Markov models that can recognize words communicated using the American Sign Language (ASL). 
In the first step, raw information were extracted from videos, such as the x, y location of the left hand, right hand, and nose in different frames. Then different features were calculated using the raw information, such as the distance between the hand and the nose, normalized cartesian coordinates, polar coordinates, and the differences in values between one frame and the next frame.
In the following step, Gaussian hidden Markov models (HMM) were trained using different groups of features. In HMMs, the number of hidden states are unknown hyperparameters that should be chosen. For this purpose, I used 3 different model selction techniques to select the number of hidden states, namely Bayesian Information Criterion (BIC), Discriminative Information Criterion (DIC), and cross-validation (CV).
The performance of models using different features and model selection techniques are summarized in ./AIND-Recognizer/asl_recognizer.ipynb.
