import numpy as np
import re

LABELS = {"ACTION after scale x, y, z", "desired goal", "achieved goal", "reward", "epoch"}

actions = []
observations = []
next_observations = []
rewards = []
episode_returns = np.zeros((100,))
episode_starts = []

# TEMP 
desired_goal = []
achieved_goal = []


def data_processing():
    with open('opolo-baselines/sim_2_real/uArm_real_data.txt') as f:
        line = f.readline()
        while line:
            # Check labels
            # print(line)

            if line == "\n":
                line = f.readline()
                continue    

            split_line = line.split(":")
            label = split_line[0]
            if label not in LABELS: 
                line = f.readline()
                continue

            print(label)

            if label == "epoch":
                pass
            elif label == "reward":
                val = float(split_line[1].strip())
                rewards.append(val)

            else:
                # List type data
                # Find values between square brackets 
                strip_line = line[line.find("[")+1:line.find("]")]
                lst = strip_line.split()
                lst = [x.strip(',') for x in lst]
                lst = [float(x) for x in lst] 
                # print(lst)

                if label == "ACTION after scale x, y, z": 
                    actions.append(lst)
                if label == "desired goal": 
                    desired_goal.append(lst)
                if label == "achieved goal":
                    achieved_goal.append(lst)

            # print(line)
            line = f.readline()
        f.close()
    return actions, desired_goal, achieved_goal, rewards


if __name__ == "__main__":
    actions, desired_goal, achieved_goal, rewards = data_processing()

    actions = np.asarray(actions) 
    desired_goal = np.asarray(desired_goal) 
    achieved_goal = np.asarray(achieved_goal) 
    rewards = np.asarray(rewards)[:20]
    observations = np.concatenate((achieved_goal, actions), axis=1).astype('float32')[:20]
    actions = np.asarray(actions) [:20]

    # Inflate to 100 epochs
    actions = np.repeat(actions, 5, axis=0)
    observations = np.repeat(observations, 5, axis=0)
    rewards = np.repeat(rewards, 5, axis=0)
    next_observations = observations[1:]

    episode_returns = np.cumsum(rewards, dtype=float)
    episode_returns = np.array([episode_returns[49], episode_returns[99]])

    episode_starts = np.zeros((100,))
    episode_starts[0] = 1
    episode_starts[50] = 1

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'next_obs': next_observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    np.savez("opolo-baselines/simulation_grounding/real_traj/Uarm_data.npz", **numpy_dict)

