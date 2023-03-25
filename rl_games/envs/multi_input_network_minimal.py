from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.network_builder import *

class Multi_Input_Minimal_A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shapes = kwargs.pop('input_shape')
            #assert(type(input_shapes) is dict)
            vec_input = input_shapes["vec"]
            vec_num_inputs = vec_input[0]
            img_input = input_shapes["img"]
            img_num_inputs = img_input
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.linear_proj = nn.Sequential()
            self.img_feature_encoder = nn.Sequential()
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.cnn_out_mlp = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            if self.has_cnn_out:
                actions_num -= self.cnn_out['shape'][0]

            linear_proj_args = {
            'input_size' : vec_num_inputs,
            'units' : [self.linear_projection['shape']],
            'activation' : self.linear_projection['activation'],
            'norm_func_name' : None,
            'dense_func' : torch.nn.Linear,
            'd2rl' : False,
            'norm_only_first_layer' : False
            }

            self.linear_proj = self._build_mlp(**linear_proj_args)
            
            if self.has_cnn:
                if self.permute_input:
                    img_num_inputs = torch_ext.shape_whc_to_cwh(img_num_inputs)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : img_num_inputs, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.has_cnn_out:
                    inp_size = self.image_feature_encoder['shape'][-1] if self.has_image_feature_encoder else self._calc_input_size(img_num_inputs, self.actor_cnn)
                    cnn_out_args = {
                        'input_size' : inp_size,
                        'units' : self.cnn_out['shape'],
                        'activation' : self.cnn_out['activation'],
                        'norm_func_name' : None,
                        'dense_func' : torch.nn.Linear,
                        'd2rl' : False,
                        'norm_only_first_layer' : False
                        }
                    self.cnn_out_mlp = self._build_mlp(**cnn_out_args)

            if self.has_image_feature_encoder:
                img_feature_encoder_args = {
                'input_size' : self._calc_input_size(img_num_inputs, self.actor_cnn),
                'units' : self.image_feature_encoder['shape'],
                'activation' : self.image_feature_encoder['activation'],
                'norm_func_name' : None,
                'dense_func' : torch.nn.Linear,
                'd2rl' : False,
                'norm_only_first_layer' : False
                }

                self.img_encoder = self._build_mlp(**img_feature_encoder_args)

                mlp_input_shape = self.linear_projection['shape'] + self.image_feature_encoder['shape'][-1]

            else:
                mlp_input_shape = self.linear_projection['shape'] + self._calc_input_size(img_num_inputs, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)
                if self.has_cnn_out:
                    self.sigma_cnn_out = nn.Parameter(torch.zeros(self.cnn_out['shape'][0], requires_grad=True, dtype=torch.float32), requires_grad=True)
                    self.sigma_act_cnn_out = self.activations_factory.create('None')

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)
                if self.has_cnn_out:
                    sigma_init(self.sigma_cnn_out)  

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            img = obs['img']
            vec = obs['vec']
            #states = obs_dict.get('rnn_states', None)
            #seq_length = obs_dict.get('seq_length', 1)
            #dones = obs_dict.get('dones', None)
            #bptt_len = obs_dict.get('bptt_len', 0)
            
            vec_out = self.linear_proj(vec)
            vec_out = vec_out.flatten(1)

            img = img.permute((0, 3, 1, 2))

            out = img
            out = self.actor_cnn(out)
            out = out.flatten(1)
            self.cnn_output = out = self.img_encoder(out)
            out = torch.cat((out,vec_out),dim=1)               
            out = self.actor_mlp(out)

            value = self.value_act(self.value(out))

            mu = self.mu_act(self.mu(out))
            sigma = self.sigma_act(self.sigma)
            cnn_mu = self.cnn_out_mlp(self.cnn_output)
            action = torch.cat((mu,cnn_mu),dim=len(mu.shape) + (-1)) #-1
            sigma_cnn = self.sigma_act_cnn_out(self.sigma_cnn_out)
            sigma = torch.cat((sigma,sigma_cnn),dim=0)
            return action, sigma, value#, torch.tensor([])
                    
        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def load(self, params):
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = False
            self.has_space = 'space' in params

            if self.has_space:
                self.is_continuous = True
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            self.linear_projection = params['linear_projection']
            
            if 'image_feature_encoder' in params:
                self.has_image_feature_encoder = True
                self.image_feature_encoder = params['image_feature_encoder']
            else:
                self.has_image_feature_encoder = False

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
                if 'cnn_out' in params['cnn']:
                    self.has_cnn_out = True
                    self.cnn_out = params['cnn']['cnn_out']
                else:
                    self.has_cnn_out = False
            else:
                self.has_cnn = False
                self.has_cnn_out = False

    def build(self, name, **kwargs):
        net = Multi_Input_Minimal_A2CBuilder.Network(self.params, **kwargs)
        return net
