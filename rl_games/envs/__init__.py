

from rl_games.envs.connect4_network import ConnectBuilder
from rl_games.envs.test_network import TestNetBuilder
from rl_games.envs.multi_input_network import Multi_Input_A2CBuilder
from rl_games.envs.multi_input_network_replay import Multi_Input_Replay_A2CBuilder
from rl_games.envs.multi_input_network_minimal import Multi_Input_Minimal_A2CBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('connect4net', ConnectBuilder)
model_builder.register_network('testnet', TestNetBuilder)
model_builder.register_network('multi_input_net', Multi_Input_A2CBuilder)
model_builder.register_network('multi_input_net_replay', Multi_Input_Replay_A2CBuilder)
model_builder.register_network('multi_input_net_minimal', Multi_Input_Minimal_A2CBuilder)
