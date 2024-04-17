"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    print("Solution:", [s, s, w, s, w, w, s, w])
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # initialize
    stackA = util.Stack()
    visited = set()
    visiting = problem.getStartState()
    path = dict()

    if problem.isGoalState(visiting):
        return []
    visited.add(visiting)

    for successors in problem.getSuccessors(visiting):
        stackA.push(successors[0])
        path[successors[0]] = [successors[1]]

    # count=0
    while stackA.isEmpty() == False:
        visiting = stackA.pop()
        if visiting not in visited:
            if problem.isGoalState(visiting):
                return path[visiting]
            visited.add(visiting)
            for successors in problem.getSuccessors(visiting):
                if successors[0] not in visited:
                    stackA.push(successors[0])
                    path[successors[0]] = path[visiting] + [successors[1]]
    # util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    queueA = util.Queue()
    visited = set()
    visiting = problem.getStartState()
    path = dict()

    if problem.isGoalState(visiting):
        return []
    visited.add(visiting)

    for successors in problem.getSuccessors(visiting):
        queueA.push(successors[0])
        visited.add(successors[0])
        path[successors[0]] = [successors[1]]

    while queueA.isEmpty() == False:
        visiting = queueA.pop()
        if problem.isGoalState(visiting):
            return path[visiting]
        visited.add(visiting)
        for successors in problem.getSuccessors(visiting):
            if successors[0] not in visited:
                queueA.push(successors[0])
                visited.add(successors[0])
                path[successors[0]] = path[visiting] + [successors[1]]
    # util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    cost = dict()
    path = dict()
    visited = set()
    # initialize
    visiting = problem.getStartState()
    cost[visiting] = 0
    path[visiting] = []
    visited.add(visiting)

    for successor in problem.getSuccessors(visiting):
        cost[successor[0]] = cost[visiting] + successor[2]
        visited.add(successor[0])
        pqueue.push(successor[0], cost[successor[0]])
        path[successor[0]] = [successor[1]]

    while pqueue.isEmpty() == False:
        visiting = pqueue.pop()
        if problem.isGoalState(visiting):
            return path[visiting]
        visited.add(visiting)
        for successor in problem.getSuccessors(visiting):
            if successor[0] not in cost.keys():
                cost[successor[0]] = cost[visiting] + successor[2]
                visited.add(successor[0])
                pqueue.push(successor[0], cost[successor[0]])
                path[successor[0]] = path[visiting] + [successor[1]]
            else:
                if cost[visiting] + successor[2] < cost[successor[0]]:
                    cost[successor[0]] = cost[visiting] + successor[2]
                    visited.add(successor[0])
                    pqueue.push(successor[0], cost[successor[0]])
                    path[successor[0]] = path[visiting] + [successor[1]]
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    cost = dict()
    astarscore = dict()
    path = dict()
    visited = set()
    # initialize
    visiting = problem.getStartState()
    cost[visiting] = 0
    astarscore[visiting] = heuristic(visiting, problem)
    path[visiting] = []
    visited.add(visiting)

    for successor in problem.getSuccessors(visiting):
        cost[successor[0]] = cost[visiting] + successor[2]
        astarscore[successor[0]] = cost[successor[0]] + heuristic(successor[0], problem)
        visited.add(successor[0])
        pqueue.push(successor[0], astarscore[successor[0]])
        path[successor[0]] = [successor[1]]

    while pqueue.isEmpty() == False:
        visiting = pqueue.pop()
        if problem.isGoalState(visiting):
            return path[visiting]
        visited.add(visiting)
        for successor in problem.getSuccessors(visiting):
            if successor[0] not in cost.keys():
                cost[successor[0]] = cost[visiting] + successor[2]
                astarscore[successor[0]] = cost[successor[0]] + heuristic(successor[0], problem)
                visited.add(successor[0])
                pqueue.push(successor[0], astarscore[successor[0]])
                path[successor[0]] = path[visiting] + [successor[1]]
            else:
                if cost[visiting] + successor[2] < cost[successor[0]]:
                    cost[successor[0]] = cost[visiting] + successor[2]
                    astarscore[successor[0]] = cost[successor[0]] + heuristic(successor[0], problem)
                    visited.add(successor[0])
                    pqueue.push(successor[0], astarscore[successor[0]])
                    path[successor[0]] = path[visiting] + [successor[1]]
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
