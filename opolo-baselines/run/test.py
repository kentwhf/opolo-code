import numpy as np

dir = r"C:\Users\kentw\OneDrive - University of Toronto\NSERC MIE 2021\GARAT-another-implementation\opolo-baselines\simulation_grounding\real_traj\FetchReach-v1_transitions_10000.npz"
with np.load(dir) as data:
    print(data['actions'])