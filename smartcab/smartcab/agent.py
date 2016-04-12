import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import OrderedDict
import random 
import numpy as np
import matplotlib.pyplot as plt
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha=1 # learning rate
        self.timer=1 # timer for modify learning rate
        self.lamda=0 # reward discount parameter to adjust
        self.life=0;
        self.failure_times=0;
        self.q_value=OrderedDict(); # Q value
        self.q_value_ground_truth=OrderedDict();# Q value ground true use as a metric
        self.q_inti_value_Flag=True; # whether to initialize Q value
        self.penalty_times=0
        self.learn_time=8
        self.q_value_valid=[]
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.timer=1
        self.life=self.life+1
        # TODO: Prepare for a new trip; reset any variables here, if required
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        if deadline == 0 and self.life>=self.learn_time:
            self.failure_times+=1
        # TODO: Update state
        #Available state
        AvailableInformation=[('Next Way Point:', self.next_waypoint),
        ('light',inputs['light']),
        ('On Comming:',inputs['oncoming']),
        ('Left:',inputs['left']),
        ('Right:',inputs['right']),
        ('Deadline',deadline)]
        # validComingAgents=[None, 'right', 'left', 'leftRight','forward',
        # 'forwardRight','forwardLeft','forwardLeftRight']
        # rightCar = 1 if 
        # comingAgents=validComingAgents[]

        self.state=AvailableInformation[0:2];# states without deadline,just light and next_waypoint as state
        # print 'Model>>>>  Light: ',inputs['light'],"  Next_waypoint:", self.next_waypoint
        # TODO: Select action according to your policy
        # random action

        action= random.choice(self.env.valid_actions)
        if self.q_inti_value_Flag==False and self.life>=self.learn_time:
            # print 'QValue>>>> Forward: ',self.q_value[(inputs['light'],self.next_waypoint,'forward')],\
            #     'Left: ',self.q_value[(inputs['light'],self.next_waypoint,'left')],\
            #     'Right: ',self.q_value[(inputs['light'],self.next_waypoint,'right')]
            ran=random.randint(1,10)#x% probability to follow Q
            if ran>3: # parameter to adjust
                action = self.get_max_a_r(inputs['light'],self.next_waypoint)[1]
        # Execute action and get reward
        # print 'Action>>>> ', action
        # print '-------------------------------------------'
        reward = self.env.act(self, action)
        if reward<0 and self.life>=self.learn_time:
            self.penalty_times=self.penalty_times+1
        # TODO: Learn policy based on state, action, reward
       
        #initial Q value only do once on the robot born set all value to zero
        if self.q_inti_value_Flag:
            for light_cond in ['green', 'red']:
                for next_waypoint_cond in ['forward','left','right']:
                    for act in self.env.valid_actions:
                        self.q_value[(light_cond,next_waypoint_cond,act)]=0
                        
                        # initial ground truth
                        if act == None:
                            self.q_value_ground_truth[(light_cond,next_waypoint_cond,act)]=1;
                        else:
                            if light_cond == 'red' and act != 'right':
                                self.q_value_ground_truth[(light_cond,next_waypoint_cond,act)]=-1;
                            else:
                                self.q_value_ground_truth[(light_cond,next_waypoint_cond,act)]= 2 if act == next_waypoint_cond else 0.5 


                        self.q_inti_value_Flag=False
        #update Q value by equation q=(1-alpha)*q+alpha(reward+lamda*expected_next_reward)
        else:
            # get new state after the action to calculate expected_next_reward
            self.alpha=1/float(self.timer)
            new_next_waypoint = self.planner.next_waypoint() # 
            new_inputs = self.env.sense(self)

            self.q_value[(inputs['light'],self.next_waypoint,action)]=\
                (1-self.alpha)*self.q_value[(inputs['light'],self.next_waypoint,action)]+\
                self.alpha*(reward+self.lamda*self.get_max_a_r(new_inputs['light'],new_next_waypoint)[0])
        if self.life<self.learn_time:
            self.q_value_valid.append(self.get_q_valid())
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    # define a function to get Q and action
    # input light_cond: current light condiction
    # input next_waypoint_cond: current next waypoint
    # output (expected Q value, best action)
    def get_max_a_r(self,light_cond,next_waypoint_cond):
        acts=['forward','left','right',None]
        values = (self.q_value[light_cond,next_waypoint_cond,'forward'],
            self.q_value[light_cond,next_waypoint_cond,'left'],
            self.q_value[light_cond,next_waypoint_cond,'right'],
            self.q_value[light_cond,next_waypoint_cond,None])
        maxValue =max(values)
        act_choices_index=[]

        for index in range(0,4):
            if values[index]==maxValue:
                act_choices_index.append(index)
        act=random.choice([acts[i] for i in act_choices_index])
        return (maxValue,act)


    def get_q_valid(self):
        q_vector=[]
        q_ground_truth=[]
        for light_cond in ['green', 'red']:
                for next_waypoint_cond in ['forward','left','right']:
                    for act in self.env.valid_actions:
                        q_vector.append(self.q_value[light_cond,next_waypoint_cond,act])
                        q_ground_truth.append(self.q_value_ground_truth[light_cond,next_waypoint_cond,act])
        q_vector=np.array(q_vector)
        q_ground_truth=np.array(q_ground_truth)
        q_vector_l2_norm=np.sum(q_vector*q_vector)**0.5
        q_ground_truth_l2_norm=np.sum(q_ground_truth*q_ground_truth)**0.5
        valid=np.sum(q_vector*q_ground_truth)/(q_vector_l2_norm*q_ground_truth_l2_norm)
        return valid


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=108)  # press Esc or close pygame window to quit
    #print 'Success rate: ',(100.0-a.failure_times)/100.0,'  penalty_times: ',a.penalty_times
    
    print 'Q Value: ',a.q_value

if __name__ == '__main__':
    run()
