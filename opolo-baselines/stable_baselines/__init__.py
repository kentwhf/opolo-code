from stable_baselines.a2c import A2C
from stable_baselines.acer import ACER
from stable_baselines.acktr import ACKTR
from stable_baselines.deepq import DQN
from stable_baselines.her import HER
from stable_baselines.ppo2 import PPO2
from stable_baselines.td3 import OPOLO, TD3, TD3BCO, TD3DAC, TD3DACFO, SAIL
from stable_baselines.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines.ddpg import DDPG
    from stable_baselines.ddpg import DDPGfD
    from stable_baselines.gail import GAIL
    from stable_baselines.ppo1 import PPO1
    from stable_baselines.trpo_mpi import TRPO, TRPOGAIL, TRPOGAIFO
del mpi4py

__version__ = "2.10.0a0"
