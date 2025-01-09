class EnvironmentDimensions:
    def __init__(self, Nstates, Nstate_rep, Naction, T, Larena):
        """
        用于描述环境维度的类
        """
        self.Nstates = Nstates          # 状态数量
        self.Nstate_rep = Nstate_rep    # 状态表示的维度
        self.Naction = Naction          # 动作数量
        self.T = T                      # 时间步长
        self.Larena = Larena            # 环境的大小


class Environment:
    def __init__(self, initialize, step, dimensions):
        """
        表示环境的类，包含初始化和执行步长的方法，以及环境维度信息
        """
        self.initialize = initialize    # 初始化环境的方法
        self.step = step                # 进行单步更新的方法
        self.dimensions = dimensions    # 环境的维度信息（EnvironmentDimensions 对象）


class WorldState:
    def __init__(self, agent_state=None, environment_state=None, planning_state=None):
        """
        表示世界状态的类，包含：
        - agent_state: 智能体的状态
        - environment_state: 环境的状态
        - planning_state: 规划的状态
        """
        self.agent_state = agent_state
        self.environment_state = environment_state
        self.planning_state = planning_state
