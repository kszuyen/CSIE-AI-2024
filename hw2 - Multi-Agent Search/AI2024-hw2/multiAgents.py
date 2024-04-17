# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import sys
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

import time


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        originalFood = currentGameState.getFood()

        maxInt = manhattanDistance((0, 0), (originalFood.width, originalFood.height))

        nearestGhost = maxInt
        for ghostPos in currentGameState.getGhostPositions():
            dis = manhattanDistance(newPos, ghostPos)
            nearestGhost = dis if dis < nearestGhost else nearestGhost
        x, y = newPos
        if originalFood[x][y] and nearestGhost > 1:
            return maxInt

        # print currentGameState.getGhostPositions()
        nearestFood = maxInt
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    dis = manhattanDistance((x, y), newPos)
                    if dis < nearestFood:
                        nearestFood = dis

        return 0 if nearestGhost <= 1 else maxInt - nearestFood
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def miniMax(gameState, index, currentDepth):
            maxScore = -sys.maxsize
            legalMoves = gameState.getLegalActions(index)
            action = Directions.STOP
            for move in legalMoves:
                successor = gameState.generateSuccessor(index, move)
                temp = minValue(successor, index + 1, currentDepth)
                if maxScore < temp:
                    maxScore = temp
                    action = move
            return action

        def maxValue(gameState, index, currentDepth):
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            index = (gameState.getNumAgents() - 1) % index
            maxScore = -sys.maxsize
            legalMoves = gameState.getLegalActions(index)
            for move in legalMoves:
                successor = gameState.generateSuccessor(index, move)
                maxScore = max(maxScore, minValue(successor, index + 1, currentDepth))
            return maxScore

        def minValue(gameState, index, currentDepth):
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            minScore = sys.maxsize
            legalMoves = gameState.getLegalActions(index)
            if index + 1 == gameState.getNumAgents():  # last ghost, so next is pacman -> depth + 1
                for move in legalMoves:
                    successor = gameState.generateSuccessor(index, move)
                    minScore = min(minScore, maxValue(successor, index, currentDepth + 1))
            else:
                for move in legalMoves:
                    successor = gameState.generateSuccessor(index, move)
                    minScore = min(minScore, minValue(successor, index + 1, currentDepth))
            return minScore

        currentDepth = 0  # set current depth to 0
        return miniMax(gameState, self.index, currentDepth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def alphaBeta_miniMax(gameState, index, currentDepth):
            alpha = -sys.maxsize
            beta = sys.maxsize
            maxScore = -sys.maxsize
            legalMoves = gameState.getLegalActions(index)
            action = Directions.STOP
            for move in legalMoves:
                successor = gameState.generateSuccessor(index, move)
                temp = minValue(successor, index + 1, currentDepth, alpha, beta)
                if maxScore < temp:
                    maxScore = temp
                    action = move
                alpha = max(alpha, maxScore)

            return action

        def maxValue(gameState, index, currentDepth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            index = (gameState.getNumAgents() - 1) % index
            maxScore = -sys.maxsize
            legalMoves = gameState.getLegalActions(index)
            for move in legalMoves:
                successor = gameState.generateSuccessor(index, move)
                maxScore = max(maxScore, minValue(successor, index + 1, currentDepth, alpha, beta))
                if maxScore > beta:
                    return maxScore
                alpha = max(alpha, maxScore)
            return maxScore

        def minValue(gameState, index, currentDepth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            minScore = sys.maxsize
            legalMoves = gameState.getLegalActions(index)
            if index + 1 == gameState.getNumAgents():  # last ghost, so next is pacman -> depth + 1
                for move in legalMoves:
                    successor = gameState.generateSuccessor(index, move)
                    minScore = min(minScore, maxValue(successor, index, currentDepth + 1, alpha, beta))
                    if minScore < alpha:
                        return minScore
                    beta = min(beta, minScore)
            else:
                for move in legalMoves:
                    successor = gameState.generateSuccessor(index, move)
                    minScore = min(minScore, minValue(successor, index + 1, currentDepth, alpha, beta))
                    if minScore < alpha:
                        return minScore
                    beta = min(beta, minScore)
            return minScore

        currentDepth = 0  # set current depth to 0

        return alphaBeta_miniMax(gameState, self.index, currentDepth)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
