from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
from random import sample
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update, update_learning_rate
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import should_collect_more_steps

from my_rl.common.off_policy_algorithm import OffPolicyAlgorithm
from my_rl.common.buffers import ExtendedReplayBuffer

class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
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
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(TD3, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
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

        mse_loss = th.nn.MSELoss()

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Get target actions from the expert data
            target_actions = replay_data.actions

            # Get current actions estimates for actor network
            current_actions = self.actor(replay_data.observations)

            # Compute actor loss
            actor_q_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
            actor_bc_loss = mse_loss(current_actions, target_actions)

            actor_loss = (actor_q_loss*float(1e-3)) + (actor_bc_loss/float(batch_size))

            # Optimize the actor
            # self.actor.optimizer.weight_decay=L_2
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            # self.actor.optimizer.weight_decay=0.00

            # update the target networks
            self.actor_target.load_state_dict(self.actor.state_dict())

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

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        mse_loss = th.nn.MSELoss()

        for gradient_step in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            agent_replay_data = self.replay_buffer.sample(N_a, env=self._vec_normalize_env)
            expert_replay_data = expert_replay_buffer.sample(N_e, env=self._vec_normalize_env)

            if N_a > 0.0:
                with th.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = agent_replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target(agent_replay_data.next_observations) + noise).clamp(-1, 1)

                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(agent_replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    agent_target_q_values = agent_replay_data.rewards + (1 - agent_replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
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
                critic_loss = (agent_critic_loss * agent_advantage * (1.0-C_e)) + (expert_critic_loss * (1.0 - agent_advantage) * C_e)
            else:
                critic_loss = (agent_critic_loss * (1.0-C_e)) + (expert_critic_loss * C_e)
            
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            # self.critic.optimizer.weight_decay=L_2
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            # self.critic.optimizer.weight_decay=0.0

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                if N_a > 0.0:
                    # Compute actor loss
                    agent_actor_loss = -self.critic.q1_forward(agent_replay_data.observations, self.actor(agent_replay_data.observations)).mean()
                else:
                    agent_actor_loss = 0.0

                if N_e > 0.0:

                    # Get target actions from the expert data
                    expert_target_actions = expert_replay_data.actions

                    # Get current actionds estimates for actor network
                    expert_current_actions = self.actor(expert_replay_data.observations)

                    # Compute actor loss
                    expert_actor_q_loss = -self.critic.q1_forward(expert_replay_data.observations, self.actor(expert_replay_data.observations)).mean()
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

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
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

        return super(TD3, self).learn(
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
        return super(TD3, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []