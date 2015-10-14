class Experience:
    def __init__(self, previous_seq, current_action, current_reward, current_seq):
        self.prev_seq = previous_seq
        self.curr_action = current_action
        self.curr_reward = current_reward
        self.curr_seq = current_seq

    def __str__(self):
        return("EXPERIENCE: (PREVIOUS SEQUENCE: {0}, CURRENT ACTION: {1}, CURRENT REWARD: {2}, CURRENT SEQUENCE: {3})\n\n".format(self.prev_seq, self.curr_action, self.curr_reward, self.curr_seq))

    def __repr__(self):
        return self.__str__()
