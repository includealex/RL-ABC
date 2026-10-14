import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from datetime import datetime
from rl_framework.Utils import setLogger, set_seed
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)

    def reset(self):
        pass  # for compatibility with OU noise

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=800, fc2_dims=600, fc3_dims=512, fc4_dims=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1. / np.sqrt(self.fc4.weight.data.size()[0])
        nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(fc4_dims)

        self.mu = nn.Linear(fc4_dims, n_actions)
        f_mu = 0.003
        nn.init.uniform_(self.mu.weight.data, -f_mu, f_mu)
        nn.init.uniform_(self.mu.bias.data, -f_mu, f_mu)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        actions = torch.tanh(self.mu(x))  # Output bounded in [-1, 1]
        return actions

class Critic(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=800, fc2_dims=600, fc3_dims=512, fc4_dims=256):
        super(Critic, self).__init__()

        # STATE pathway
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1. / np.sqrt(self.fc4.weight.data.size()[0])
        nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(fc4_dims)

        # ACTION pathway (injected late)
        self.action_value = nn.Linear(n_actions, fc4_dims)

        # Final output: Q-value
        self.q = nn.Linear(fc4_dims, 1)
        f_q = 0.003
        nn.init.uniform_(self.q.weight.data, -f_q, f_q)
        nn.init.uniform_(self.q.bias.data, -f_q, f_q)

    def forward(self, state, action):
        s = F.relu(self.bn1(self.fc1(state)))
        s = F.relu(self.bn2(self.fc2(s)))
        s = F.relu(self.bn3(self.fc3(s)))
        s = (self.bn4(self.fc4(s)))

        a = F.relu(self.action_value(action))

        # Combine state and action features
        x = F.relu(s + a)
        q_value = self.q(x)

        return q_value

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d, action_mask, next_action_mask):
        """Store one transition and its transition-specific action masks."""
        self.buffer.append((s, a, r, s2, d, action_mask, next_action_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        transition_columns = tuple(zip(*batch))
        if len(transition_columns) != 7:
            raise ValueError(
                "Replay transitions must contain current and next action masks. "
                "Legacy five-field replay buffers cannot be used for dynamic masking."
            )

        (
            state,
            action,
            reward,
            next_state,
            done,
            action_mask,
            next_action_mask,
        ) = map(np.stack, transition_columns)

        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device),
            torch.FloatTensor(action_mask).to(device),
            torch.FloatTensor(next_action_mask).to(device),
        )

    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filepath, expected_obs_dim=None):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)

        # Support files that contain either a deque directly or a ReplayBuffer.
        loaded_buffer = getattr(loaded_buffer, "buffer", loaded_buffer)
        if loaded_buffer and len(loaded_buffer[0]) != 7:
            raise ValueError(
                "This replay buffer predates dynamic action masking and does not "
                "contain current/next masks. Collect a new replay buffer before "
                "training the masked agent."
            )
        if loaded_buffer and expected_obs_dim is not None:
            state_width = np.asarray(loaded_buffer[0][0]).reshape(-1).size
            next_state_width = np.asarray(loaded_buffer[0][3]).reshape(-1).size
            expected_obs_dim = int(expected_obs_dim)
            if state_width != expected_obs_dim or next_state_width != expected_obs_dim:
                raise ValueError(
                    "Replay buffer observation width is incompatible with the "
                    f"current environment: found {state_width}/{next_state_width}, "
                    f"expected {expected_obs_dim}. Collect a new buffer or "
                    "explicitly migrate the old one."
                )

        self.buffer = deque(loaded_buffer, maxlen=self.buffer.maxlen)
            

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class DDPGAgent:
    def __init__(self, env, convert= True, alpha=1e-4, beta=1e-3, batch_size=128, gamma=0.99, tau=0.005, max_size=1000000, noise_type='gaussian',
                 log_interval=100, eval_interval=1000, num_steps=1000000, seed=0, exp="", load=False, load_tag="last"):
        set_seed(seed)
        self.seed = int(seed)
        self.device= device
        
        self.env = env
        self.convert= convert
        self.input_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.actor = Actor(self.input_dims, self.n_actions).to(device)
        self.actor_target = Actor(self.input_dims, self.n_actions).to(device)
        self.critic = Critic(self.input_dims, self.n_actions).to(device)
        self.critic_target = Critic(self.input_dims, self.n_actions).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=beta)

        self.buffer = ReplayBuffer(capacity=max_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if noise_type == 'ou':
            self.noise = OUNoise(self.n_actions)
        elif noise_type == 'gaussian':
            self.noise = GaussianNoise(self.n_actions)
        else:
            raise ValueError("Invalid noise type. Choose 'ou' or 'gaussian'.")

        # Logging and evaluation
        self.steps = 0
        self.episodes = 0
        self.evaluation_count = 0
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.train_rewards = []
        self.logs = []
        self.exp = exp
        self.load = load
        self.load_tag = "last" if str(load_tag).lower() in ("latest",) else str(load_tag).lower()

        # TensorBoard
        self.summary_dir = os.path.join(os.getcwd(), 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{self.summary_dir}/{self.exp}")

        # CSV logger
        self.logger, self.file_handler = setLogger(False, 'ddpg_logger.csv',
                                                  ('episode', 'epsilon'))

        # Debug logger
        debug_logger_headers = ('step', 'action', 'noise', 'source')
        self.logger_debug, self.file_handler_debug = setLogger(False, 'ddpg_debug_logger.csv', debug_logger_headers)

        # For evaluation. Keep each process in a unique directory so files
        # from separate runs never mix or overwrite one another.
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id = f"run_{started_at}_pid{os.getpid()}"
        self.plots_root = f"plots_{self.seed}"
        self.logs_root = f"logs_{self.seed}"
        self.plots_dir = os.path.join(self.plots_root, self.run_id)
        self.logs_dir = os.path.join(self.logs_root, self.run_id)
        self.models_dir = f"models_{self.seed}"
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.best_eval_particles = -1
        self.best_eval_log = None

    '''def choose_action(self, state, greedy=0.5, log_source="train"):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = greedy * self.noise.sample()
        action_noisy = action + noise
        action_noisy = np.clip(action_noisy, -1, 1)
        # Rescale to env action space if needed
        low = self.env.action_space.low
        high = self.env.action_space.high
        action_rescaled = low + (0.5 * (action_noisy + 1.0) * (high - low))
        action_rescaled = np.clip(action_rescaled, low, high)
        # Debug log
        return action_noisy'''
   

    def choose_action(self, state, greedy=0.5, log_source="train"):
        if state is None:
            obs_dim = self.env.observation_space.shape[0]
            state = np.zeros(obs_dim, dtype=np.float32)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        noise = greedy * self.noise.sample()
        action_noisy = action + noise
        action_noisy = np.clip(action_noisy, -1, 1)
        return action_noisy

    def get_greedy(self, episode, n_episodes, greedy_start=0.5, greedy_end=0.05):
        """Linearly decay greedy (noise scale) from greedy_start to greedy_end over n_episodes."""
        greedy = greedy_start - (greedy_start - greedy_end) * (episode / max(1, n_episodes - 1))
        return max(greedy, greedy_end)

    def train(self, n_episodes=1000, max_steps=1000, greedy=0.5, start=0):
        episode_max_steps = (
            max_steps
            if max_steps is not None
            else getattr(self.env, "max_num_of_vars", None)
        )
        if episode_max_steps is None or int(episode_max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer")
        episode_max_steps = int(episode_max_steps)

        scores = []
        for ep in range(start, n_episodes,1):
            greedy_ep = self.get_greedy(ep, n_episodes, greedy_start=greedy, greedy_end=0.05)
            reset_out = self.env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            if state is None:
                print(f"Warning: reset returned None at episode {ep + 1}, skipping episode")
                continue
            self.reset_noise()
            score = 0
            episode_actions = []
            done = False
            for episode_step in range(episode_max_steps):
                if type(state) == tuple:
                    state = state[0]
                action = self.choose_action(state, greedy=greedy_ep, log_source="train")
                next_state, reward, terminated, truncated, info = self.env.step(action, self.convert)
                done = bool(
                    terminated
                    or truncated
                    or episode_step + 1 >= episode_max_steps
                )
                if next_state is None:
                    obs_dim = self.env.observation_space.shape[0]
                    next_state = np.zeros(obs_dim, dtype=np.float32)
                    done = True

                # returned_action is the fixed 4-D policy-space action that was
                # effective in the environment. Do not use info["masked_action"]
                # here; that is the beamline-length simulator action vector.
                masked_action = info["returned_action"]
                action_mask = info["returned_mask"]
                next_action_mask = info["returned_mask_next"]
                self.remember(
                    state,
                    masked_action,
                    reward,
                    next_state,
                    done,
                    action_mask,
                    next_action_mask,
                )
                self.learn()
                state = next_state
                score += reward
                self.steps += 1
                episode_actions.append(masked_action)
                if done:
                    self.writer.add_scalar('Rewards/Reward', reward, self.steps)
                    self.writer.add_scalar('Rewards/NumberOfParticles', info['number_of_particles'], self.steps)
                    break
            self.episodes = ep + 1
            if self.episodes % self.eval_interval == 0:
                    self.evaluate()
            scores.append(score)
            self.train_rewards.append(score)
            self.logger.writerow({'episode': ep+1, 'epsilon': greedy_ep})
            #self.logger.writerow({'episode': ep+1, 'reward': reward, 'actions': episode_actions, 'output_file': info.get('output_file', '')})
            self.file_handler.flush()
        return scores
    

    def _log_debug(self, step, action, noise, source):
        ep_info = {
            'step': step,
            'action': action,
            'noise': noise,
            'source': source
        }
        self.logger_debug.writerow(ep_info)
        self.file_handler_debug.flush()

    def reset_noise(self):
        self.noise.reset()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        (
            state,
            action,
            reward,
            next_state,
            done,
            action_mask,
            next_action_mask,
        ) = self.buffer.sample(self.batch_size)

        # The replay action should already be masked, but applying its mask here
        # enforces the invariant at the critic boundary as well.
        action = action * action_mask

        with torch.no_grad():
            # Each transition can have a different next-element mask.
            next_action = self.actor_target(next_state) * next_action_mask
            target_q = self.critic_target(next_state, next_action)
            y = reward + (1 - done) * self.gamma * target_q

        q = self.critic(state, action)
        critic_loss = F.mse_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the actor only through action dimensions that are valid for
        # each sampled current state. Multiplication also gives exactly zero
        # gradient to inactive output dimensions.
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        policy_action = self.actor(state) * action_mask
        actor_loss = -self.critic(state, policy_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        # TensorBoard logging
        self.writer.add_scalar('loss/critic', critic_loss.item(), self.steps)
        self.writer.add_scalar('loss/actor', actor_loss.item(), self.steps)

        if self.steps % self.log_interval == 0:
            print(f"Step {self.steps}: Critic loss {critic_loss.item():.4f}, Actor loss {actor_loss.item():.4f}")

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def remember(self, s, a, r, s2, d, action_mask, next_action_mask):
        action = np.asarray(a, dtype=np.float32).reshape(-1)
        action_mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        next_action_mask = np.asarray(next_action_mask, dtype=np.float32).reshape(-1)

        for name, value in (
            ("action", action),
            ("action_mask", action_mask),
            ("next_action_mask", next_action_mask),
        ):
            if value.size != self.n_actions:
                raise ValueError(
                    f"{name} must contain {self.n_actions} values, got {value.size}"
                )

        # Store only the action that could actually affect this transition.
        self.buffer.push(
            s,
            action * action_mask,
            r,
            s2,
            d,
            action_mask.copy(),
            next_action_mask.copy(),
        )


    def evaluate(self, episodes=1):
        self.evaluation_count += 1
        eval_name = (
            f"ddpg_eval_episode_{self.episodes:04d}_"
            f"step_{self.steps:07d}_eval_{self.evaluation_count:03d}"
        )
        eval_log_file = os.path.join(self.logs_dir, f"{eval_name}.csv")
        eval_plot_file = os.path.join(self.plots_dir, f"{eval_name}.png")
        eval_logger_headers = ('episode', 'rewards', 'number_of_particles', 'output_file', 'episode_reward', 'actions')
        eval_logger, eval_file_handler = setLogger(False, eval_log_file, eval_logger_headers)

        self.actor.eval()
        previous_eval_mode = getattr(self.env, "eval_mode", False)
        self.env.eval_mode = True
        print("####### DDPG EVALUATING #########")
        best_particles_this_eval = -1
        total_reward = 0.0
        for i in range(episodes):
            episode_rewards = []
            episode_reward = 0.0
            done = False
            reset_out = self.env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            episode_actions = []
            t = 0
            eval_max_steps = int(getattr(self.env, "max_num_of_vars", 1000))
            while not done and t < eval_max_steps:
                if type(state)== tuple:
                    state= state[0]
                action = self.choose_action(state, greedy=0.0, log_source="eval")
                next_state, reward, terminated, truncated, info = self.env.step(action, self.convert)
                done = bool(terminated or truncated or t + 1 >= eval_max_steps)
                if next_state is None:
                    obs_dim = self.env.observation_space.shape[0]
                    next_state = np.zeros(obs_dim, dtype=np.float32)
                    done = True
                episode_reward += reward
                state = next_state
                episode_rewards.append(reward)
                episode_actions.append(info.get("returned_action", action))
                t += 1
                eval_info = {
                    'episode': i + 1,
                    'rewards': reward,
                    'output_file': info.get("output_file", ""),
                    'number_of_particles': info.get("number_of_particles", 0),
                    'episode_reward': episode_reward,
                    'actions': info["dict_vars"] 
                }
                eval_logger.writerow(eval_info)
                eval_file_handler.flush()

            final_particles = info.get("number_of_particles", 0)
            best_particles_this_eval = max(best_particles_this_eval, final_particles)
            total_reward += episode_reward

            print("Episode:", i + 1)
            print('rewards:', reward)
            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, t+1), episode_rewards, 'r-o', label='Sum Episode Reward')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title(f'DDPG Evaluation at Training Step {self.steps}')
            plt.legend()
            plt.grid(True)
            plt.savefig(eval_plot_file)
            plt.close()

        eval_file_handler.close()
        avg_reward = total_reward / max(int(episodes), 1)

        #self.buffer.save(f"ddpg_checkpoint_intermediate_{self.seed}.pkl")
        if best_particles_this_eval > self.best_eval_particles:
            self.best_eval_particles = best_particles_this_eval
            self.best_eval_log = (eval_log_file, eval_plot_file)
            self.save_models(tag="best")
            print(f'New best model! number_of_particles: {best_particles_this_eval} at step {self.steps}')
        # Save after updating the best score so the resumable checkpoint and
        # the best-actor file describe the same evaluation state.
        self.save_models(tag="last")

        print(f'DDPG Evaluation completed at step {self.steps}')
        print(f'Evaluation log kept at: {eval_log_file}')
        print(f'Average reward: {avg_reward:.2f} over {episodes} episodes')
        print(f'Best number_of_particles so far: {self.best_eval_particles}')
        print("@@@@@@@ DONE DDPG EVALUATING @@@@@@@")
        self.env.eval_mode = previous_eval_mode
        self.actor.train()

    def save_models(self, tag="last"):
        os.makedirs(self.models_dir, exist_ok=True)
        if tag == "best":
            torch.save(self.actor.state_dict(), os.path.join(self.models_dir, "ddpg_actor_best.pth"))
        else:
            torch.save({
                'steps': self.steps,
                'episodes': self.episodes,
                'evaluation_count': self.evaluation_count,
                'run_id': self.run_id,
                'best_eval_particles': self.best_eval_particles,
                'input_dims': self.input_dims,
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            }, os.path.join(self.models_dir, "ddpg_checkpoint_last.pth"))
        print(f"DDPG Models saved successfully ({tag}) at step {self.steps}.")

    def _validate_checkpoint_observation_dim(self, actor_state, checkpoint_name):
        """Reject checkpoints built for the former 57/58-value observation."""
        first_layer = actor_state.get('fc1.weight')
        if first_layer is None:
            return
        saved_input_dims = int(first_layer.shape[1])
        if saved_input_dims != self.input_dims:
            raise ValueError(
                f"{checkpoint_name} uses observation size {saved_input_dims}, "
                f"but this environment uses {self.input_dims} values "
                "(beam/aperture features plus angle and tilt). Start a new run "
                "or explicitly migrate the checkpoint and replay buffer."
            )

    def load_models(self, tag="last"):
        if tag == "best":
            checkpoint_path = os.path.join(self.models_dir, "ddpg_actor_best.pth")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Requested checkpoint not found: {checkpoint_path}. "
                    "Set training.load_model to false to train from scratch."
                )
            actor_state = torch.load(checkpoint_path, map_location=self.device)
            self._validate_checkpoint_observation_dim(
                actor_state, "Best actor checkpoint"
            )
            self.actor.load_state_dict(actor_state)
            print("Loaded best actor for evaluation.")
        else:
            checkpoint_path = os.path.join(self.models_dir, "ddpg_checkpoint_last.pth")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Requested checkpoint not found: {checkpoint_path}. "
                    "Set training.load_model to false to train from scratch."
                )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self._validate_checkpoint_observation_dim(
                checkpoint['actor'], "DDPG training checkpoint"
            )
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.steps = checkpoint['steps']
            self.episodes = checkpoint.get('episodes', 0)
            self.evaluation_count = checkpoint.get('evaluation_count', 0)
            self.best_eval_particles = checkpoint['best_eval_particles']
            print(f"Resumed training from step {self.steps} (best particles: {self.best_eval_particles}).")
