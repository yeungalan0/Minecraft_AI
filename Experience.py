class Experience:
    def __init__(self, previous_seq, current_action, current_reward, current_seq):
        self.prev_seq = previous_seq
        self.curr_action = current_action
        self.curr_reward = current_reward
        self.curr_seq = current_seq
