# Import the necessary modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .plotting import *
from .train import *
from .model import *
from .loss_hyperparameters import *
from .environment import *
from .a2c import *
from .initializations import *
from .walls import *
from .maze import *
from .walls_build import *
from .io import *
from .model_planner import *
from .planning import *
from .human_utils_maze import *
from .priors import *
from .walls_baselines import *

# Set plotting style
matplotlib.rcParams["font.size"] = 16
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

