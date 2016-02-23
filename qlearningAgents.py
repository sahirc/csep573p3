# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.q_table = util.Counter() # Init the table to a dictionary of all 0's

    def getQValue(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def computeValueFromQValues(self, state):
        q_values = []
        for action in self.getLegalActions(state):
          q_values.append(self.getQValue(state, action))

        if q_values: 
          return max(q_values)
        else: 
          return 0

    def computeActionFromQValues(self, state):
        best_actions = []
        for action in self.getLegalActions(state):
          if (self.getQValue(state, action) == self.getValue(state)): # This is the best q value we can hope to obtain
            best_actions.append(action)

        if len(best_actions) == 1:
          return best_actions[0]
        elif len(best_actions) > 1:
          return random.choice(best_actions)
        else: 
          return None

    def getAction(self, state):
        # Acc to some probability, we take a random action
        # Otherwise, we follow the best action available
        if util.flipCoin(self.epsilon):
          return random.choice(self.getLegalActions(state))
        else:
          return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        self.q_table[(state, action)] = (
          1 - self.alpha)*self.q_table[(state, action)] + \
          self.alpha*( reward + self.discount*self.getValue(nextState))
          
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature in features:
            q_value = q_value + (self.weights[feature] * features[feature])
        return q_value


    def update(self, state, action, nextState, reward):
        correction = reward + self.discount*self.getValue(nextState) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * correction * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
