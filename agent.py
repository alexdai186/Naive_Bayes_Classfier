import constants
import module
import OpenNero
import constants
import Queue

class Turret(OpenNero.AgentBrain):
    """
    Simple Rotating Turret
    """
    def __init__(self):
        OpenNero.AgentBrain.__init__(self)
        self.team = constants.OBJECT_TYPE_TEAM_1

    def initialize(self, init_info):
        self.actions = init_info.actions
        self.sensors = init_info.sensors
        self.group = "Turret"
        self.previous_fire =  0
        return True

    def start(self, time, sensors):
        self.org = None
        self.net = None
        self.state.label = "Turret"
        self.group = "Turret"
        a = self.actions.get_instance()
        for x in range(len(self.actions)):
            a[x] = 0
            if x == 1:
              a[x] = 0
        return a

    def act(self,time,sensors,reward):
        a = self.actions.get_instance()
        for x in range(len(self.actions)):
            a[x] = 0
            if x == 1:
              a[x] = .15

        return a

    def get_team(self):
        return self.team

    def end(self,time,reward):
        return True

    def destroy(self):
        return True

class FirstPersonAgent(OpenNero.AgentBrain):
    """
    A human-controlled agent with a Firset Person View
    """
    key_buffer = Queue.Queue(5) # key buffer
    @classmethod
    def control_fps(cls, action):
        try:
            cls.key_buffer.put_nowait(action)
        except Queue.Full:
            try:
                cls.key_buffer.get_nowait()
                cls.key_buffer.put_nowait(action)
            except Queue.Empty, Queue.Full:
                pass # should rarely happen
    def __init__(self):
        OpenNero.AgentBrain.__init__(self) # do not remove!
        self.group = "FirstPersonAgent"
    def initialize(self, init_info):
        self.action_info = init_info.actions
        return True
    def get_team(self):
        # we are not on either team 1 or 2! we are just watching.
        return 0
    def key_action(self):
        action = self.action_info.get_instance() # create a zero action
        try:
            key = FirstPersonAgent.key_buffer.get_nowait()
            if key is not None and key in constants.FIRST_PERSON_ACTIONS:
                (movement, turn) = constants.FIRST_PERSON_ACTIONS[key]
                action[0] = movement
                action[1] = turn
        except Queue.Empty:
            pass # no keys were pressed
        return action
    def start(self, time, observations):
        return self.key_action()
    def act(self, time, observations, reward):
        return self.key_action()
    def end(self, time, reward):
        return True
