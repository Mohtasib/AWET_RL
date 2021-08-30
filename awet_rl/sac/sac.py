from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update, update_learning_rate
from stable_baselines3.sac.policies import SACPolicy

from my_rl.common.off_policy_algorithm import OffPolicyAlgorithm
from my_rl.common.buffers import ExtendedReplayBuffer

class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def pre_train_actor(
        self,
        replay_buffer: ExtendedReplayBuffer,
        gradient_steps: int = 1000,
        batch_size: int = 100,
        actor_lr: float = 1e-3,
        C_l: float = 0.5,
        L_2: float = 0.01,
        ):

        # Set learning rates:
        update_learning_rate(self.actor.optimizer, actor_lr)
        if self.ent_coef_optimizer is not None:
            update_learning_rate(self.ent_coef_optimizer, actor_lr)

        mse_loss = th.nn.MSELoss()

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Get target actions from the expert data
            target_actions = replay_data.actions
            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            current_actions, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            else:
                ent_coef = self.ent_coef_tensor

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, current_actions), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # actor_log_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_log_loss = (log_prob - min_qf_pi).mean()
            actor_bc_loss = mse_loss(current_actions, target_actions)

            actor_loss = (actor_log_loss*float(1e-3)) + (actor_bc_loss/float(batch_size))

            # Optimize the actor
            # self.actor.optimizer.weight_decay=L_2
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            # self.actor.optimizer.weight_decay=0.00

    def pre_train_actor_temp(
        self,
        replay_buffer: ExtendedReplayBuffer,
        gradient_steps: int = 1000,
        batch_size: int = 100,
        actor_lr: float = 1e-3,
        C_l: float = 0.5,
        L_2: float = 0.01,
        ):

        # Set learning rates:
        update_learning_rate(self.actor.optimizer, actor_lr)
        if self.ent_coef_optimizer is not None:
            update_learning_rate(self.ent_coef_optimizer, actor_lr)

        mse_loss = th.nn.MSELoss()

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Get target actions from the expert data
            target_actions = replay_data.actions
            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            current_actions, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, current_actions), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # actor_log_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_log_loss = (log_prob - min_qf_pi).mean()
            actor_bc_loss = mse_loss(current_actions, target_actions)

            actor_loss = (actor_log_loss*float(1e-3)) + (actor_bc_loss/float(batch_size))

            # Optimize the actor
            # self.actor.optimizer.weight_decay=L_2
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            # self.actor.optimizer.weight_decay=0.00

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Get target actions from the expert data
            target_actions = replay_data.actions
            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            current_actions, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            else:
                ent_coef = self.ent_coef_tensor

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

    def train(
        self, 
        expert_replay_buffer: ExtendedReplayBuffer, 
        gradient_steps: int, 
        batch_size: int = 100,
        C_e: float = 0.5, # The percentage of the expert sample in the final sampled batch
        C_l: float = 0.5,
        L_2: float = 0.01,
        C_clip: float = 0.01,
        AA_mode: bool = False,
        ) -> None:

        N_a = int(batch_size*(1.0-C_e)) # Batch size of the agent replay sample
        N_e = int(batch_size*C_e) # Batch size of the expert sample

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        mse_loss = th.nn.MSELoss()

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            agent_replay_data = self.replay_buffer.sample(N_a, env=self._vec_normalize_env)
            expert_replay_data = expert_replay_buffer.sample(N_e, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            if N_a > 0.0:
                agent_current_actions, agent_log_prob = self.actor.action_log_prob(agent_replay_data.observations)
                agent_log_prob = agent_log_prob.reshape(-1, 1)
            else:
                agent_current_actions, agent_log_prob = None, None
            if N_e > 0.0:
                expert_current_actions, expert_log_prob = self.actor.action_log_prob(expert_replay_data.observations)
                expert_log_prob = expert_log_prob.reshape(-1, 1)
            else:
                expert_current_actions, expert_log_prob = None, None

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                if N_a > 0.0:
                    agent_ent_coef_loss = -(self.log_ent_coef * (agent_log_prob + self.target_entropy).detach()).mean()
                else:
                    agent_ent_coef_loss = None
                if N_e > 0.0:
                    expert_ent_coef_loss = -(self.log_ent_coef * (expert_log_prob + self.target_entropy).detach()).mean()
                else:
                    expert_ent_coef_loss = None

                # ent_coef_loss = (agent_ent_coef_loss * (1.0-C_e)) + (expert_ent_coef_loss * C_e)
                ent_coef_loss = agent_ent_coef_loss + expert_ent_coef_loss
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            if N_a > 0.0:
                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(agent_replay_data.next_observations)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(agent_replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    agent_target_q_values = agent_replay_data.rewards + (1 - agent_replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
                agent_current_q_values = self.critic(agent_replay_data.observations, agent_replay_data.actions)

                # Compute critic loss
                agent_critic_loss = sum([F.mse_loss(current_q, agent_target_q_values) for current_q in agent_current_q_values])
                agent_critic_loss = th.clip(agent_critic_loss, 0, C_clip)

            else:
                agent_critic_loss = 0.0 

            if N_e > 0.0:
                # Get target Q-values from the expert data
                expert_target_q_values = expert_replay_data.extras # Note: the q_values are stored in the expert replay buffer as extras

                # Get current Q-values estimates for each critic network
                expert_current_q_values = self.critic(expert_replay_data.observations, expert_replay_data.actions)

                # Compute critic loss
                expert_critic_loss = sum([F.mse_loss(current_q, expert_target_q_values) for current_q in expert_current_q_values])
            else:
                expert_critic_loss = 0.0

            if AA_mode:
                # Calculate the agent advantage based on the q_values:
                q_a = min([th.mean(q_values) for q_values in agent_current_q_values])
                q_e = min([th.mean(q_values) for q_values in expert_current_q_values])
                agent_advantage = q_a / (q_a + q_e)
                # critic_loss = (agent_critic_loss * agent_advantage * (1.0-C_e)) + (expert_critic_loss * (1.0 - agent_advantage) * C_e)
                critic_loss = (agent_critic_loss * agent_advantage) + (expert_critic_loss * (1.0 - agent_advantage))
            else:
                # critic_loss = (agent_critic_loss * (1.0-C_e)) + (expert_critic_loss * C_e)
                critic_loss = agent_critic_loss + expert_critic_loss
            
            # critic_loss = critic_loss * 0.5 # not sure about this but it is used in the original impementation
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            # self.critic.optimizer.weight_decay=L_2
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            # self.critic.optimizer.weight_decay=0.0

            if N_a > 0.0:
                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Mean over all critic networks
                q_values_pi = th.cat(self.critic.forward(agent_replay_data.observations, agent_current_actions), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                agent_actor_loss = (ent_coef * agent_log_prob - min_qf_pi).mean()
            else:
                agent_actor_loss = 0.0

            if N_e > 0.0:

                # Get target actions from the expert data
                expert_target_actions = expert_replay_data.actions

                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Mean over all critic networks
                q_values_pi = th.cat(self.critic.forward(expert_replay_data.observations, expert_current_actions), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                expert_actor_q_loss = (ent_coef * expert_log_prob - min_qf_pi).mean()

                expert_actor_bc_loss = mse_loss(expert_current_actions, expert_target_actions)
                expert_actor_loss = (expert_actor_q_loss * float(1e-3)) + (expert_actor_bc_loss / float(N_e))

            else:
                expert_actor_loss = 0.0

            actor_loss = ((agent_actor_loss * float(1e-3) * (1.0-C_e)) + (expert_actor_loss * C_e))
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            # self.actor.optimizer.weight_decay=L_2
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            # self.actor.optimizer.weight_decay=0.00

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def train_old(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,

        expert_data_path: str = None,
        gradient_steps: int = 1000,
        C_e: float = 0.5,
        C_l: float = 0.5,
        L_2: float = 0.01,
        C_clip: float = 0.01,
        num_demos: int = 100,
        AA_mode: bool = False,
        ET_mode: bool = False,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,

            expert_data_path=expert_data_path,
            gradient_steps=gradient_steps,
            C_e=C_e,
            C_l=C_l,
            L_2=L_2,
            C_clip=C_clip,
            num_demos=num_demos,
            AA_mode=AA_mode,
            ET_mode=ET_mode,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
