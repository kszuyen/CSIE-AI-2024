# CSIE5400 - HW1
## Introduction
The code for this project consists of several Python files, some of which you will need to read and understand in order to complete the assignment, and some of which you can ignore. You can download all the code and supporting files as a search.zip.

Files you'll edit:

| File                             | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| Files you'll edit:               |                                                            |
| search.py                        | Where all of your search algorithms will reside.           |
| searchAgents.py                  | Where all of your search-based agents will reside.         |
| Files you might want to look at: |                                                            |
| pacman.py                        | The main file that runs Pacman games.                      |
| game.py                          | The logic behind how the Pacman world works.               |
| util.py                          | Useful data structures for implementing search algorithms. |
| Supporting files you can ignore: |                                                            |
| graphicsDisplay.py               | Graphics for Pacman                                        |
| graphicsUtils.py                 | Support for Pacman graphics                                |
| textDisplay.py                   | ASCII graphics for Pacman                                  |
| ghostAgents.py                   | Agents to control ghosts                                   |
| keyboardAgents.py                | Keyboard interfaces to control Pacman                      |
| layout.py                        | Code for reading layout files and storing their contents   |
| autograder.py                    | Project autograder                                         |
| testParser.py                    | Parses autograder test and solution files                  |
| testClasses.py                   | General autograding test classes                           |
| test_cases/                      | Directory containing the test cases for each question      |
| searchTestClasses.py             | Project 1 specific autograding test classes                |

**Files to Edit and Submit**: You will fill in portions of search.py and searchAgents.py during the assignment. Once you have completed the assignment, you will submit these files to Gradescope (for instance, you can upload all .py files in the folder). Please do not change the other files in this distribution.

**Evaluation**: Your code will be autograded for technical correctness. Please do not change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation – not the autograder’s judgements – will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.

**Academic Dishonesty**: We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else’s code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don’t try. We trust you all to submit your own work only; please don’t let us down. If you do, we will pursue the strongest consequences available to us.

## Welcome to Pacman
After downloading the code, unzipping it, and changing to the directory, you should be able to play a game of Pacman by typing the following at the command line:

```bash
python pacman.py
```

Pacman lives in a shiny blue world of twisting corridors and tasty round treats. Navigating this world efficiently will be Pacman’s first step in mastering his domain.

The simplest agent in searchAgents.py is called the GoWestAgent, which always goes West (a trivial reflex agent). This agent can occasionally win:
```bash
python pacman.py --layout testMaze --pacman GoWestAgent
```

But, things get ugly for this agent when turning is required:
```bash
python pacman.py --layout tinyMaze --pacman GoWestAgent
```
If Pacman gets stuck, you can exit the game by typing CTRL-c into your terminal.

Soon, your agent will solve not only tinyMaze, but any maze you want.

Note that <code>pacman.py</code> supports a number of options that can each be expressed in a long way (e.g., <code>--layout</code>) or a short way (e.g., <code>-l</code>). You can see the list of all options and their default values via:
```bash
python pacman.py -h
```
Also, all of the commands that appear in this project also appear in commands.txt, for easy copying and pasting. In UNIX/Mac OS X, you can even run all these commands in order with bash commands.txt.

## Q1. Depth First Search

In <code>searchAgents.py</code>, you’ll find a fully implemented SearchAgent, which plans out a path through Pacman’s world and then executes that path step-by-step. The search algorithms for formulating a plan are not implemented – that’s your job.

First, test that the <code>SearchAgent</code> is working correctly by running:
```bash
python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
```
The command above tells the <code>SearchAgent</code> to use <code>tinyMazeSearch</code> as its search algorithm, which is implemented in <code>search.py</code>. Pacman should navigate the maze successfully.

Now it’s time to write full-fledged generic search functions to help Pacman plan routes! Pseudocode for the search algorithms you’ll write can be found in the lecture slides. Remember that a search node must contain not only a state but also the information necessary to reconstruct the path (plan) which gets to that state.

**Important note**: All of your search functions need to return a list of actions that will lead the agent from the start to the goal. These actions all have to be legal moves (valid directions, no moving through walls).

**Important note**: Make sure to use the <code>Stack</code>, <code>Queue</code> and <code>PriorityQueue</code> data structures provided to you in <code>util.py</code>! These data structure implementations have particular properties which are required for compatibility with the autograder.

*Hint*: Each algorithm is very similar. Algorithms for DFS, BFS, UCS, and A* differ only in the details of how the fringe is managed. So, concentrate on getting DFS right and the rest should be relatively straightforward. Indeed, one possible implementation requires only a single generic search method which is configured with an algorithm-specific queuing strategy. (Your implementation need not be of this form to receive full credit).

Implement the depth-first search (DFS) algorithm in the depthFirstSearch function in search.py. To make your algorithm complete, write the graph search version of DFS, which avoids expanding any already visited states.

Your code should quickly find a solution for:
```bash
python pacman.py -l tinyMaze -p SearchAgent
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze -z .5 -p SearchAgent
```
The Pacman board will show an overlay of the states explored, and the order in which they were explored (brighter red means earlier exploration). Is the exploration order what you would have expected? Does Pacman actually go to all the explored squares on his way to the goal?

Hint: If you use a Stack as your data structure, the solution found by your DFS algorithm for mediumMaze should have a length of 130 (provided you push successors onto the fringe in the order provided by getSuccessors; you might get 246 if you push them in the reverse order). Is this a least cost solution? If not, think about what depth-first search is doing wrong.

Grading: Please run the below command to see if your implementation passes all the autograder test cases.
```bash
python autograder.py -q q1
```

## Q2. Breadth First Search

Implement the breadth-first search (BFS) algorithm in the breadthFirstSearch function in search.py. Again, write a graph search algorithm that avoids expanding any already visited states. Test your code the same way you did for depth-first search.

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```
Does BFS find a least cost solution? If not, check your implementation.

*Hint*: If Pacman moves too slowly for you, try the option –frameTime 0.

*Note*: If you’ve written your search code generically, your code should work equally well for the eight-puzzle search problem without any changes.

```bash
python eightpuzzle.py
```
Grading: Please run the below command to see if your implementation passes all the autograder test cases.
```bash
python autograder.py -q q2
```

## Q3. Uniform Cost Search
While BFS will find a fewest-actions path to the goal, we might want to find paths that are “best” in other senses. Consider <code>mediumDottedMaze</code> and <code>mediumScaryMaze</code>.

By changing the cost function, we can encourage Pacman to find different paths. For example, we can charge more for dangerous steps in ghost-ridden areas or less for steps in food-rich areas, and a rational Pacman agent should adjust its behavior in response.

Implement the uniform-cost graph search algorithm in the <code>uniformCostSearch</code> function in <code>search.py</code>. We encourage you to look through <code>util.py</code> for some data structures that may be useful in your implementation. You should now observe successful behavior in all three of the following layouts, where the agents below are all UCS agents that differ only in the cost function they use (the agents and cost functions are written for you):
```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```

*Note*: You should get very low and very high path costs for the StayEastSearchAgent and StayWestSearchAgent respectively, due to their exponential cost functions (see searchAgents.py for details).

Grading: Please run the below command to see if your implementation passes all the autograder test cases.
```bash
python autograder.py -q q3
```

## Q4. A* Search (null heuristic)
Implement A* graph search in the empty function <code>aStarSearch</code> in <code>search.py</code>. A* takes a heuristic function as an argument. Heuristics take two arguments: a state in the search problem (the main argument), and the problem itself (for reference information). The <code>nullHeuristic</code> heuristic function in <code>search.py<code> is a trivial example.

You can test your A* implementation on the original problem of finding a path through a maze to a fixed position using the Manhattan distance heuristic (implemented already as <code>manhattanHeuristic</code> in <code>searchAgents.py</code>).
```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```
You should see that A* finds the optimal solution slightly faster than uniform cost search (about 549 vs. 620 search nodes expanded in our implementation, but ties in priority may make your numbers differ slightly). What happens on <code>openMaze</code> for the various search strategies?

Grading: Please run the below command to see if your implementation passes all the autograder test cases.
```bash
python autograder.py -q q4
```

## Q5. Beadth First Search (Finding All the Corners)

The real power of A* will only be apparent with a more challenging search problem. Now, it’s time to formulate a new problem and design a heuristic for it.

In corner mazes, there are four dots, one in each corner. Our new search problem is to find the shortest path through the maze that touches all four corners (whether the maze actually has food there or not). Note that for some mazes like <code>tinyCorners</code>, the shortest path does not always go to the closest food first! Hint: the shortest path through <code>tinyCorners</code> takes 28 steps.

*Note*: Make sure to complete Question 2 before working on Question 5, because Question 5 builds upon your answer for Question 2.

Implement the <code>CornersProblem</code> search problem in <code>searchAgents.py</code>. You will need to choose a state representation that encodes all the information necessary to detect whether all four corners have been reached. Now, your search agent should solve:
```bash
python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```
To receive full credit, you need to define an abstract state representation that does not encode irrelevant information (like the position of ghosts, where extra food is, etc.). In particular, do not use a Pacman <code>GameState</code> as a search state. Your code will be very, very slow if you do (and also wrong).

An instance of the <code>CornersProblem</code> class represents an entire search problem, not a particular state. Particular states are returned by the functions you write, and your functions return a data structure of your choosing (e.g. tuple, set, etc.) that represents a state.

Furthermore, while a program is running, remember that many states simultaneously exist, all on the queue of the search algorithm, and they should be independent of each other. In other words, you should not have only one state for the entire <code>CornersProblem</code> object; your class should be able to generate many different states to provide to the search algorithm.

*Hint 1*: The only parts of the game state you need to reference in your implementation are the starting Pacman position and the location of the four corners.

*Hint 2*: When coding up <code>getSuccessors</code>, make sure to add children to your successors list with a cost of 1.

Our implementation of <code>breadthFirstSearch</code> expands just under 2000 search nodes on <code>mediumCorners</code>. However, heuristics (used with A* search) can reduce the amount of searching required.

Grading: Please run the below command to see if your implementation passes all the autograder test cases.
```bash
python autograder.py -q q5
```

## Q6. A* Search (Corners Problem: Heuristic)
*Note*: Make sure to complete Question 4 before working on Question 6, because Question 6 builds upon your answer for Question 4.

Implement a non-trivial, consistent heuristic for the <code>CornersProblem</code> in <code>cornersHeuristic</code>.
```bash
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```
Note: <code>AStarCornersAgent</code> is a shortcut for
```bash
-p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic
```
**Admissibility vs. Consistency**: Remember, heuristics are just functions that take search states and return numbers that estimate the cost to a nearest goal. More effective heuristics will return values closer to the actual goal costs. To be admissible, the heuristic values must be lower bounds on the actual shortest path cost to the nearest goal (and non-negative). To be consistent, it must additionally hold that if an action has cost c, then taking that action can only cause a drop in heuristic of at most c.

Remember that admissibility isn’t enough to guarantee correctness in graph search – you need the stronger condition of consistency. However, admissible heuristics are usually also consistent, especially if they are derived from problem relaxations. Therefore it is usually easiest to start out by brainstorming admissible heuristics. Once you have an admissible heuristic that works well, you can check whether it is indeed consistent, too. The only way to guarantee consistency is with a proof. However, inconsistency can often be detected by verifying that for each node you expand, its successor nodes are equal or higher in in f-value. Moreover, if UCS and A* ever return paths of different lengths, your heuristic is inconsistent. This stuff is tricky!

**Non-Trivial Heuristics**: The trivial heuristics are the ones that return zero everywhere (UCS) and the heuristic which computes the true completion cost. The former won’t save you any time, while the latter will timeout the autograder. You want a heuristic which reduces total compute time, though for this assignment the autograder will only check node counts (aside from enforcing a reasonable time limit).

**Grading**: Your heuristic must be a non-trivial non-negative consistent heuristic to receive any points. Make sure that your heuristic returns 0 at every goal state and never returns a negative value. Depending on how few nodes your heuristic expands, you’ll be graded:
| Number of nodes expanded | Grade |
|-------------------------|-------|
| more than 2000         | 0/9   |
| at most 2000            | 3/9   |
| at most 1600            | 6/9   |
| at most 1200            | 9/9   |

Remember: If your heuristic is inconsistent, you will receive no credit, so be careful!

Grading: Please run the below command to see if your implementation passes all the autograder test cases.
```bash
python autograder.py -q q6
```
