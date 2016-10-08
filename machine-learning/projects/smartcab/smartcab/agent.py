import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.states_dict= dict()
        self.Q = []
        self.gamma = 0.7 #Q learnging Gamma constant
        self.t = 1 #time initialization
        self.alpha = 1/self.t
        self.action_idxs = zip(range(0,4), Environment.valid_actions)
        self.use_random_action = True
        self.L = list()
        
        self.found = False
        
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        self.t += 1
        self.alpha = 1./self.t
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
#        deadline = self.env.get_deadline(self)

        # TODO: Update state
        curr_state = inputs
        curr_state['next_waypoint'] = self.planner.next_waypoint()
#        states['deadline'] = self.env.get_deadline(self)
        self.state = (curr_state)


        # TODO: Select action according to your policy
        if self.alpha > 0.000025:
            
            
            self.L.append(self.use_random_action)
            found_state = False
            if self.use_random_action:
                action = random.choice(Environment.valid_actions)        
            else:
                for (a,b) in self.Q:
                    if a == curr_state and len(self.Q) > 130:
#                        print b
                        act_idx = b.index(max(b))
                        action = Environment.valid_actions[act_idx]    
#                        print action
#                        action = random.choice(Environment.valid_actions)   
#                        print "this learned action: " + str(curr_state)
                        found_state = True
                        break
                if not found_state:
                    action = random.choice(Environment.valid_actions)        
#                    print "curr_state not found "  + str(len(self.Q))
                        
                         
            action_idx = Environment.valid_actions.index(action)
#            print " ...." + str(action)
            
        
            # Execute action and get reward
            reward = self.env.act(self, action)
#            print 'reward: ' + str(reward)
        ################ Q-learning equation ############################
        #Q (state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        
        #Q(1, 5) = R(1, 5) + 0.8 * Max[Q(5, 1), Q(5, 4), Q(5, 5)] = 100 + 0.8 * 0 = 100
        #################################################################        
#            print '*'*31
            
            
            self.found = False
            for k,v in self.states_dict.iteritems():                   
                if curr_state == v:
#                    print 'found an instance: ' + str(v)
                    self.use_random_action = False
#                    self.L.append(self.use_random_action)
                    try:
#                        Q-learning equation V->(1-alpha)V + alpha(X)
                        self.Q[k][1][action_idx] = (1 - self.alpha) * (self.Q[k][1][action_idx]) +\
                        (self.alpha) * (reward + self.gamma*max(self.Q[action_idx][1]))
                        
                    except IndexError:
                        # random initialization of action vector range [-0.5 2.5]
                        self.Q.append((curr_state, [3*random.random() - 0.5 for i in xrange(4)]))
                        self.Q[k][1][action_idx] = (1 - self.alpha) * (self.Q[k][1][action_idx]) +\
                        (self.alpha) * (reward + self.gamma*max(self.Q[-1][1]))
#                    print "k: " + str(k)
        #                self.Q[k] = q.append(zip(Environment.valid_actions, [0]*4))
                    self.found = True
                    break            
                    
            if self.found == False:
                print self.alpha
#                use random action on next iteration when curr_state is not found in dictionary of states
                self.use_random_action = True
#                self.L.append(self.use_random_action)
#                print '\n\n\nnot found'      
#                print curr_state
#                for k,v in self.states_dict.iteritems():
#                    print v
                    
                #add the curr_state to dictionary of states
                self.states_dict[len(self.states_dict)] = curr_state   
                
                #initialize instance with 3-typle [state, actions, rewards]                
                self.Q.append((curr_state, [3*random.random() - 0.5 for i in xrange(4)]))  
#                print self.Q
                try:
                    self.Q[-1][1][action_idx] = (1 - self.alpha) * (self.Q[-1][1][action_idx]) +\
                    (self.alpha) * (reward + self.gamma*max(self.Q[action_idx][1]))
        
                except IndexError:
#                    the future action, Q[-2], is used to estimate Q[-1]
                    self.Q.append((curr_state, [3*random.random() - 0.5 for i in xrange(4)]))
                    self.Q[-2][1][action_idx] = (1 - self.alpha) * (self.Q[-2][1][action_idx]) +\
                    (self.alpha) * (reward + self.gamma*max(self.Q[-1][1]))                                
        #            self.Q[-1][1][action_idx] = reward
                
            
            
        # TODO: Learn policy based on state, action, reward
                    
        else:
            pass
#        print len(self.Q)
#        print len(self.states_dict)
        
                   
#        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
