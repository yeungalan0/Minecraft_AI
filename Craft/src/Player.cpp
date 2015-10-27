#include <map>
#include <array>
#include <string>
#include <iostream>
#include <algorithm>

class Action {
public:
	Action(int break_block, double updown_rot, double leftright_rot, double forwardback, int leftright);
};

Action::Action(int _break_block = 0, double _updown_rot = 0.0, double _leftright_rot = 0.0, double _forwardback = 0, int _leftright = 0) {
	int break_block = _break_block;
	double updown_rot = _updown_rot;
	double leftright_rot = _leftright_rot;
	double forwardback = _forwardback;
	int leftright = _leftright;
}

class Sequence {
public:
	std::array<int, 7056> frame_1; // Each frame will be 84^2 in size
	Action action_1;
	std::array<int, 7056> frame_2;
	Action action_2;
	std::array<int, 7056> frame_3;
	Action action_3 = -1;
	std::array<int, 7056> frame_4;
	Sequence();
	int add_frame(std::array<int, 7056> frame);
	int add_action(Action action);
	void add_element(std::array<int, 7056> frame, Action action);
	void to_CNN_input();
};

Sequence::Sequence() {
	frame_1 = {-1};
	action_1 = -1; //Why does this work?
	frame_2 = {-1};
    action_2 = -1;
    frame_3 = {-1};
	action_3 = -1;
	frame_4 = {-1};
}

int Sequence::add_frame(std::array<int, 7056> frame) {
	if(frame_1[0] == -1) {
		frame_1 = frame;
		return 0;
	}
	else if(frame_2[0] == -1) {
		frame_2 = frame;
		return 0;
	}
	else if(frame_3[0] == -1) {
		frame_3 = frame;
		return 0;
	}
	else if(frame_4[0] == -1) {
		frame_4 = frame;
		return 0;
	}
	return 1;
}

void Sequence::to_CNN_input() {
	//TODO: Work on this
}

int Sequence::add_action(Action action) {
	// TODO: Deep copy action
	// TODO: get equality and uninitialized check working
	// if(action_1 == -1) {
	// 	action_1 = action;
	// 	return 0;
	// }
	// else if(action_2 == -1) {
	// 	action_2 = action;
	// 	return 0;
	// }
	// else if(action_3 == -1) {
	// 	action_3 = action;
	// 	return 0;
	// }
	// return 1;
}

void Sequence::add_element(std::array<int, 7056> frame, Action action) {
	// The idea here was to pass one NULL parameter in. There has got to be a better way!
	
	// if(frame != NULL) {
	// 	int returned = add_frame(frame);
	// }
	// else {
	// 	int returned = add_action(action);
	// }
	// if(returned == 1) {
	// 	// Create a new sequence object
	// }
}

class Experience {
public:
	Experience();
};

class Player {
public:                    // begin public section
	Player(); // Player Constructor
};

// constructor of Player,
Player::Player()
{
	int previous_reward = 0;
	int total_score = 0;
	
}

int main() {
	return 0;
}
