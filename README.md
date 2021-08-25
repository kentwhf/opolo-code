# *GARAT*: Generative Adversarial Reinforced Action Transformation

Research code to accompany the paper: [*An Imitation from Observation Approach to Transfer Learning with Dynamics Mismatch*](https://arxiv.org/pdf/2008.01594.pdf).

### Project Scope:

We work on a Sim2Real/Sim2Sim Interface to resolve dynamics mismatch for find and touch tasks. Specifically, we are using the GARAT framework to learn action transformation policy (ATP) by an imitation learning method (TRPOGAIFO) based on target environment samples, and then updating target policy to be deployed at target environment through RL algorithms (DDPG + HER). 

---

### Branch:
- master: all of our changes since *OPOLO*
- temp: temporary branch with miscellaneous updates that are not consistent with the current git log, but may show clear our immediate changes

---

### Training *GARAT*:

- Example: run on the FetchReach-v1 task
- Obtain a source policy from running DDPG+HER in FetchReach-v1, <code>opolo-baselines/run/test.zip</code> in our case
- Collect demonstrated trajectories by function <code>generate_target_traj(rollout_policy_path, env, save_path, n_episodes, n_transitions, seed)</code>
  - Function may be called and executed in training an ATP
  - One can reduce the dimensionality of samples used
- Train an ATP by <code>opolo-baselines/run/train_agent_custom.py</code>
- Update target policy to be deployed at target environment by <code>opolo-baselines/simulation_grounding/train_target_policy.py</code>

---

### Evaluating *ATP*:

<pre><code>python opolo-baselines/simulation_grounding/plot_state_distributions.py</code></pre>

Results can be found at: <code>opolo-baselines/atp_plots/</code>

---

#### Reminders:

- One can tune hyperparameters for the used grounding algorithm at:
<pre><code>opolo-baselines/hyperparams/</code></pre>

- We use a script to convert raw text to desired trajectory format of numpy dictionary, due to unstable connection with our UArm Swift robotic arm.

<pre><code>python opolo-baselines/sim_2_real/data_processing.py</code></pre>

- ATPs can be found at:

<pre><code>opolo-baselines\run\test\logs\trpo-gaifo\trpogaifo\FetchReach-v1</code></pre>

where <code>rank0</code> is of full-length samples and gamma = 0.95, <code>rank1</code> is of reduced-length samples and gamma = 0.95,  and <code>rank1</code> is of reduced-length samples and gamma = 0.1

- For Windows users, WSL2 is recommended for the purpose of using the <code>mujoco</code> library