# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # is value interation q value iteration?
        
        # initialize values
          # can i just use self.values or do i need to make a copy every time a take an actiom
        for i in range(self.iterations):# plus 1 or -1?
          copy = self.values.copy()

          for state in self.mdp.getStates():  
            if not self.mdp.isTerminal(state):
              max_action = None
              max_value  = float("-inf") 
              reward = 0 # init
              for action in self.mdp.getPossibleActions(state):
                x = 0
                for nextState, transitionProb in self.mdp.getTransitionStatesAndProbs(state, action):
                  reward = self.mdp.getReward(state, action, nextState)
                  x += transitionProb * (reward + self.discount * copy[nextState])               
                if x > max_value:    
                  max_value = max(max_value,x)
              self.values[state] = max_value

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        totalsum = 0
        reward = 0
        for nextstate, transition in self.mdp.getTransitionStatesAndProbs(state, action):
          reward = self.mdp.getReward(state, action, nextstate)
          totalsum += transition * (reward + self.discount * self.getValue(nextstate)) # right way to access previous value?
        return totalsum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxvalue = -100000000
        bestaction = None
        for action in self.mdp.getPossibleActions(state):
          valueforthisaction = self.getQValue(state, action) # is this right? 
          if valueforthisaction > maxvalue:
              bestaction = action
              maxvalue = valueforthisaction
        return bestaction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
      for i in range(self.iterations): # plus 1 or -1?
        copy = self.values.copy()
        #for state in self.mdp.getStates():
        state = self.mdp.getStates()[i % len(self.mdp.getStates())]
        if not self.mdp.isTerminal(state):
          max_value  = float("-inf") 
          reward = 0  
          for action in self.mdp.getPossibleActions(state):
            x = 0
            for nextState, transitionProb in self.mdp.getTransitionStatesAndProbs(state, action):
              reward = self.mdp.getReward(state, action, nextState)
              x += transitionProb * (reward + self.discount * copy[nextState])               
            if x > max_value:    
              max_value = max(max_value, x)
          self.values[state] = max_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
    def runValueIteration(self):
      import util, math
      #pDict = util.Counter()
      #for s in self.mdp.getStates():  #get predecessor 
        #pDict[s] = self.getPredecessors(s)

      pq = util.PriorityQueue()

      for s in self.mdp.getStates():
        if not self.mdp.isTerminal(s):
          highvalue = self.highestQvalue(s)
          diff = math.fabs(self.values[s] - highvalue)
          pq.update(s, -diff)   # takes a self?
      for i in range(self.iterations):
        if pq.isEmpty(): 
          return  
        s = pq.pop()
        if not self.mdp.isTerminal(s):
          self.values[s] = self.highestQvalue(s)     # highestQvalue? s's value??
        for p in self.getPredecessors(s): #not sure if i need to initalize pdict and use that 
          diff = math.fabs(self.values[p] - self.highestQvalue(p)) #same as above
          if diff > self.theta:
            pq.update(p, -diff)

    def getPredecessors(self, state): 
      pList = set()
      for possiblePred in self.mdp.getStates():
        for action in self.mdp.getPossibleActions(possiblePred):
          for nextstateofpp, prob in self.mdp.getTransitionStatesAndProbs(possiblePred, action):
            if nextstateofpp == state and prob > 0: # check this, what if state is terminal?
              pList.add(possiblePred) 
      return pList
    
    def highestQvalue(self, state): #should it take self
      maxnum = float("-inf") 
      for action in self.mdp.getPossibleActions(state):
        #for nextstate, p in mdp.getTransitionStatesAndProbs(state, action):
        x = self.getQValue(state, action)  
        maxnum = max(maxnum, x)
      return maxnum
