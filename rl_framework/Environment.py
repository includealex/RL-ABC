import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import StandardScaler
from rl_framework.Utils import compute_covariance_matrix_mean, reset_specific_keys, setLogger, create_feature_matrix, configure_initial_particles, create_seeded_tracking_file, nn_observation_size
from rl_framework.Utils_magnet_actions import (
    ACTION_CHANNEL_INDEX,
    normalize_reset_keys,
    reset_policy_mask,
    split_control_variable,
)
from rl_framework.Elegant_edited import ElegantWrapper
import math
import os
import logging


logger = logging.getLogger(__name__)
MAX_RESET_RETRIES = 3


class ACCElegantEnvironment(gym.Env):
    """Gymnasium environment with a fixed 59-value observation for ``n_bins=5``.

    The final two values are the signed bend angle and tilt of the upcoming
    controllable magnet.  Both are zero for an end-of-line observation, and
    angle is zero for quadrupoles.
    """

    metadata = {"render_modes": []}

    def __init__(self,stage = None,n_bins= 5, init_num_particles= None, set_initial_particles=True, results_path= ' ', input_beamline_file= "machine.lte", input_beam_file= "track", beamline_name= 'machine', output_beamline_file='updated_machine.lte', elegant_input_filename='elegant_input.lte', reset_specific_keys_bool= True, reset_keys=None, validate_generated_lattice=True, logger=None, file_handler=None, elegant_path= "", sddsPath= "", override_dynamic_command= False, overridden_command= " ", seed=0):
        self.override_dynamic_command= override_dynamic_command
        self.overridden_command= overridden_command
        self.seed = int(seed)
        self.input_file_path = input_beamline_file
        self.base_input_beam_file = input_beam_file
        self.output_file = output_beamline_file
        self.elegant_input_filename = elegant_input_filename
        self.beamline_name = beamline_name
        self.elegantPath = elegant_path
        self.sddsPath = sddsPath
        self.results_path= results_path
        self.inputs_dir = f"inputs_{self.seed}"
        self.n_bins= n_bins
        self.initial_reward= 1
        self.init_num_particles = init_num_particles
        self.configured_init_num_particles = init_num_particles
        self.set_initial_particles = set_initial_particles
        self.validate_generated_lattice = bool(validate_generated_lattice)
        self.reset_specific_keys_bool = bool(reset_specific_keys_bool)
        self.reset_keys = normalize_reset_keys(reset_keys)
        self._initial_particle_count_reported = False
        self.input_beam_file = create_seeded_tracking_file(
            self.base_input_beam_file,
            self.elegant_input_filename,
            self.results_path,
            self.seed
        )
        self.wrapper = self._make_wrapper()

        '''
        print("#############$$$$$$$$$$$$$$$$$$$$$ Env DEBUG $$$$$$$$$$$$$$##############")

        print("###### from Environment ######")
        print("Results Path: ", self.results_path)
        print("override_dynamic_command:  ", override_dynamic_command)
        print("overridden_command:  ", overridden_command)
        print("self.wrapper.overrid_dynmaic_commnad:  ", self.wrapper.overrid_dynmaic_commnad)
        print("self.wrapper.overrideen_command:  ", self.wrapper.overrideen_command)

        print("###### Outside Environment ######")
        print("#############$$$$$$$$$$$$$$$$$$$$$ Env DEBUG $$$$$$$$$$$$$$##############")
        '''

        self.max_num_of_vars=  len(self.wrapper.chroneological_order_controllable_vars)
        self.max_num_of_states = self.wrapper.max_itteration
        if stage != None and stage < self.max_num_of_vars:
            self.stage = stage  # Stage for curriculum learning
        else:
            self.stage= None
        self.stage_mask = None  # Mask for controlling variables based on the stage
        self.observation = None
        self.variables = self.wrapper.chroneological_variables
        self._sync_action_registry()
        self._set_action_space(self.variables)
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32)
        self.logger = logger
        self.file_handler = file_handler
        # When True, _log_episode records current_element as None so that
        # evaluation rollouts are clearly distinguishable from training rows.
        # The custom DDPG path never sets this flag (default False), so its
        # logging behaviour is completely unchanged.
        self.eval_mode = False


        # Initialize the stage mask if stage is not None
        if self.stage is not None:
            self.stage_mask = self._get_stage_mask()
        #rest the self.iteration to zero because _get_stage_mask changes it, we can also call reset() outside to get the same effect.
        self.iteration= 0

    def _set_action_space(self, variables):
        """Set the action space based on the list of variables. set correct max, min ranges for each variable"""
        #keep this action space to be used for convert vars and other functions , but it will not reflect the actual action space.
        action_low = []
        action_high = []

        for var in variables:
            if "K1" in var:
                action_low.append(-20)
                action_high.append(20)
            elif any(x in var for x in ["VKICK", "HKICK", "FSE"]):
                action_low.append(-0.005)
                action_high.append(0.005)
            else:
                # Default range for other variables (if any)
                action_low.append(-1)
                action_high.append(1)

        action_low = np.array(action_low, dtype=np.float32)
        action_high = np.array(action_high, dtype=np.float32)

        self.action_space_fake = spaces.Box(low=action_low, high=action_high)
        self.action_space_fake.n = self.action_space_fake.low.size

        #Real action space, this one to be used for the actual environment and it will also be used to define the shape of the output of the policy network.
        action_low=   np.array([-20, -0.005, -0.005, -0.005], dtype=np.float32)
        action_high= np.array([20, 0.005, 0.005, 0.005], dtype=np.float32)

        self.action_space = spaces.Box(low=action_low, high=action_high)
        self.action_space.n = self.action_space.low.size

    def _sync_action_registry(self):
        """Attach and validate the occurrence-specific registry built by the wrapper."""
        registry = self.wrapper.magnet_action_registry
        if tuple(self.variables) != registry.variable_names:
            raise RuntimeError(
                "Wrapper control variables changed while rebuilding the environment"
            )
        self.magnet_action_registry = registry
        self.controllable_parameters_by_element = registry.parameters_by_element()
        self.magnet_action_inventory = registry.inventory(
            self.reset_specific_keys_bool,
            reset_keys=self.reset_keys,
        )

    def _convert_variables(self, values):
        """Convert values from the range [-1, 1] to the correct range based on the variable names."""
        converted_variables = []
        for i, value in enumerate(values):
            range_min = self.action_space.low[i]
            range_max = self.action_space.high[i]
            converted_variable = value * (range_max - range_min) / 2 + (range_max + range_min) / 2
            converted_variables.append(converted_variable)

        return np.round(converted_variables,4)

    def _get_stage_mask(self):
        """
        Generate a stage mask based on the current stage.
        The stage mask determines which variables are controlled in the current stage.

        Returns:
        np.array: A binary mask indicating the variables to control.
        """
        if hasattr(self, "magnet_action_registry"):
            return self.magnet_action_registry.cumulative_environment_mask(
                self.stage - 1,
                reset_specific_keys_bool=self.reset_specific_keys_bool,
                dtype=np.float64,
                reset_keys=self.reset_keys,
            )
        stage_mask = np.zeros(len(self.variables))
        count = 0

        # Iterate through the stages and accumulate the count of variables to control
        for i in range(self.stage):
            self.iteration = i
            count_current, _ = self._check_number_of_variables_to_be_set_at_this_iteration()
            count += count_current
            stage_mask[:count] = 1

        return stage_mask

    def _observation_dim(self):
        return nn_observation_size(self.n_bins)

    def _fallback_observation(self):
        dim = getattr(self, "observation_shape", self._observation_dim())
        return np.zeros(dim, dtype=np.float32)

    def _log_simulation_error(self, context, error_message=""):
        message = f"{context} (seed={self.seed}): {error_message}\n"
        logger.warning(message.strip())
        with open("error_log.txt", "a") as error_file:
            error_file.write(message)

    def _make_wrapper(self):
        return ElegantWrapper(
            self.input_file_path,
            self.input_beam_file,
            self.beamline_name,
            self.output_file,
            elegant_path=self.elegantPath,
            sddsPath=self.sddsPath,
            results_path=self.results_path,
            elegant_input_filename=self.elegant_input_filename,
            overrid_dynmaic_commnad=self.override_dynamic_command,
            overrideen_command=self.overridden_command,
            seed=self.seed,
            n_bins=self.n_bins,
            reset_specific_keys_bool=self.reset_specific_keys_bool,
            reset_keys=self.reset_keys,
            validate_generated_lattice=self.validate_generated_lattice,
        )

    def _run_reset_simulation(self):
        last_error = "unknown error"
        for attempt in range(1, MAX_RESET_RETRIES + 1):
            try:
                self.wrapper = self._make_wrapper()
                self._sync_action_registry()
                if self.wrapper.max_itteration == 0:
                    raise RuntimeError("Lattice preprocessing produced no watch points")

                values = np.zeros(len(self.wrapper.chroneological_variables))
                elegant_input, success, dict_vars = self.wrapper.run_elegant_simulation(values)
                if not success:
                    raise RuntimeError(
                        self.wrapper.last_elegant_error or "Elegant simulation failed on reset"
                    )

                observations, reward, output_file, done = self.wrapper.get_results(
                    self.initial_number_of_particles
                )
                if observations is None:
                    raise RuntimeError("get_results returned None observation on reset")
                return observations, reward
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Reset attempt %s/%s failed for seed=%s: %s",
                    attempt,
                    MAX_RESET_RETRIES,
                    self.seed,
                    exc,
                )

        self._log_simulation_error("reset", last_error)
        raise RuntimeError(
            f"Environment reset failed after {MAX_RESET_RETRIES} attempts: "
            f"{last_error}"
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        if seed is not None and int(seed) != self.seed:
            old_seed = self.seed
            self.seed = int(seed)
            self.inputs_dir = f"inputs_{self.seed}"
            # Update all seeded filenames/paths from old seed to new seed so
            # the newly-created .ele file references the correct paths.
            self.elegant_input_filename = self.elegant_input_filename.replace(
                f"_{old_seed}.", f"_{self.seed}."
            )
            self.results_path = self.results_path.replace(
                f"_{old_seed}/", f"_{self.seed}/"
            )
            self.output_file = self.output_file.replace(
                f"_{old_seed}.", f"_{self.seed}."
            )
            self.input_beam_file = create_seeded_tracking_file(
                self.base_input_beam_file,
                self.elegant_input_filename,
                self.results_path,
                self.seed
            )
        sdds2stream_path = os.path.join(self.elegantPath, "sdds2stream")
        self.initial_number_of_particles, beam_mode, beam_path = configure_initial_particles(
            self.input_beam_file + ".ele",
            self.set_initial_particles,
            self.configured_init_num_particles,
            sdds2stream_path=sdds2stream_path,
        )
        # In external-beam mode the SDDS file, never the YAML value, is the
        # source of truth. Keep the public attribute synchronized for logging.
        self.init_num_particles = self.initial_number_of_particles
        source_detail = f" from {beam_path}" if beam_path else ""
        if not self._initial_particle_count_reported:
            logger.info(
                "Training initial particles: %s (%s%s)",
                self.initial_number_of_particles,
                beam_mode,
                source_detail,
            )
            print(
                f"Training initial particles: "
                f"{self.initial_number_of_particles} ({beam_mode}{source_detail})"
            )
            self._initial_particle_count_reported = True
        # initial number of partciles is actually new and it should be equal to self.initial_reward
        self._set_initial_number_of_particles(self.initial_number_of_particles)
        self.done = False
        self.reward = 0.0
        self.iteration = 0
        self.previous_mask_len = 0
        self.mask_len = 0
        self.actions_ = np.zeros((len(self.variables),))
        observations, reward = self._run_reset_simulation()

        if reward != 0:
            self.initial_reward = reward
            self.number_of_particle_prev = self.initial_reward

        self.observation_shape = self._observation_dim()
        self.observation = self._validated_observation(observations, "reset")
        return self.observation, {}

    def _validated_observation(self, observation, context):
        """Return a finite float32 vector and fail on accidental schema drift."""
        result = np.asarray(observation, dtype=np.float32).reshape(-1)
        expected = self._observation_dim()
        if result.size != expected:
            raise RuntimeError(
                f"{context} produced {result.size} observation values; expected "
                f"{expected} (57 beam/aperture + angle + tilt for n_bins=5)."
            )
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f"{context} produced a non-finite observation")
        return result

    def _set_initial_number_of_particles(self,new):
        self.initial_number_of_particles = new


    def _check_number_of_variables_to_be_set_at_this_iteration(self):
        """Return the action-variable counts for the current and next elements."""
        if hasattr(self, "magnet_action_registry"):
            current = self.magnet_action_registry.spec_for_iteration(self.iteration)
            next_spec = self.magnet_action_registry.spec_for_iteration(
                self.iteration, offset=1
            )
            return (
                0 if current is None else len(current.parameters),
                0 if next_spec is None else len(next_spec.parameters),
            )
        return (
            len(self._variables_for_iteration(0)),
            len(self._variables_for_iteration(1)),
        )

    @staticmethod
    def _split_variable_name(variable):
        """Return ``(element_name, action_channel)`` for an agent variable."""
        return split_control_variable(variable)

    def _variables_for_iteration(self, offset=0):
        if hasattr(self, "magnet_action_registry"):
            spec = self.magnet_action_registry.spec_for_iteration(
                self.iteration, offset=offset
            )
            return [] if spec is None else list(spec.variable_names)
        element_order = self.wrapper.chroneological_order_controllable_vars
        element_index = self.iteration + offset
        if not 0 <= element_index < len(element_order):
            return []
        element_name = element_order[element_index]
        return [
            variable
            for variable in self.variables
            if self._split_variable_name(variable)[0] == element_name
        ]

    def _action_indices_for_iteration(self, offset=0):
        if hasattr(self, "magnet_action_registry"):
            mask = self.magnet_action_registry.availability_policy_mask(
                self.iteration, offset=offset, dtype=np.float64
            )
            return list(np.flatnonzero(mask))
        channel_index = ACTION_CHANNEL_INDEX
        indices = []
        for variable in self._variables_for_iteration(offset):
            _, parameter = self._split_variable_name(variable)
            if parameter is None:
                raise RuntimeError(f"Unknown controllable variable: {variable}")
            indices.append(channel_index[parameter])
        return indices

    def _get_mask(self, mask_len):
        """Generate a mask for the current iteration.
        len(variables)= 10
        mask_len=3
        output= [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        """
        if hasattr(self, "magnet_action_registry"):
            return self.magnet_action_registry.cumulative_environment_mask(
                self.iteration,
                reset_specific_keys_bool=self.reset_specific_keys_bool,
                dtype=np.float64,
                reset_keys=self.reset_keys,
            )
        mask = np.zeros(len(self.variables))
        mask[:mask_len] = 1
        return mask

    def _get_action_mask(self, count, offset=0):
        if hasattr(self, "magnet_action_registry"):
            mask = self.magnet_action_registry.availability_policy_mask(
                self.iteration, offset=offset, dtype=np.float64
            )
            if int(mask.sum()) != count:
                raise RuntimeError(
                    f"Expected {count} action channels, found {int(mask.sum())}"
                )
            return mask
        mask = np.zeros(self.action_space.n)
        indices = self._action_indices_for_iteration(offset)
        if len(indices) != count:
            raise RuntimeError(
                f"Expected {count} action channels, found {len(indices)}"
            )
        mask[indices] = 1
        return mask

    def _get_new_action(self, count, action, offset=0):
        if hasattr(self, "magnet_action_registry"):
            values = self.magnet_action_registry.policy_values(
                self.iteration,
                action,
                reset_specific_keys_bool=False,
                offset=offset,
            )
            if len(values) != count:
                raise RuntimeError(
                    f"Expected {count} action channels, found {len(values)}"
                )
            return values
        indices = self._action_indices_for_iteration(offset)
        if len(indices) != count:
            raise RuntimeError(
                f"Expected {count} action channels, found {len(indices)}"
            )
        return np.asarray(action)[indices]

    def _correct_action(self, action):
        """
        Ensure that only the new variables corresponding to the current iteration are updated,
        while keeping the previously set variables unchanged.

        Parameters:
        action (np.array): The action values to be applied.

        Returns:
        np.array: The combined action values with the new variables updated and previous variables unchanged.

        Example:
        inputs:
        len(variables)= 10
        mask_len=5
        previous_mask_len= 3

        outputs:
        action_mask= [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]
        previous_actions_mask= [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]
        """
        if hasattr(self, "magnet_action_registry"):
            spec = self.magnet_action_registry.spec_for_iteration(self.iteration)
            if spec is None or len(action) != len(spec.parameters):
                raise RuntimeError(
                    f"Invalid environment action at iteration {self.iteration}"
                )
            result = np.zeros(len(self.variables), dtype=np.float64)
            if self.iteration > 0:
                prior_mask = self.magnet_action_registry.cumulative_environment_mask(
                    self.iteration - 1,
                    reset_specific_keys_bool=self.reset_specific_keys_bool,
                    dtype=np.float64,
                    reset_keys=self.reset_keys,
                ).astype(bool)
                result[prior_mask] = self.actions_[prior_mask]
            for variable_index, value in zip(spec.variable_indices, action):
                result[variable_index] = value
            return result

        # Create a mask for the current iteration's variables
        action_mask = np.zeros(len(self.variables))
        #action_mask[self.previous_mask_len:self.mask_len] = 1
        action_mask[self.previous_mask_len:self.mask_len]= action

        # Apply the mask to the new action values
        ###new_action = action * action_mask

        # Create a mask for the previously set variables
        previous_actions_mask = np.zeros(len(self.variables))
        previous_actions_mask[:self.previous_mask_len] = 1

        # Combine the previous actions with the new action values
        #previous_Actions = (previous_actions_mask * self.actions_) + new_action #!!!!
        previous_Actions = (previous_actions_mask * self.actions_) + action_mask
        return previous_Actions


    def _log_episode(self, info):
        """Log the episode information."""
        ep_info = {
            'reward': info["reward"],
            'done': info["done"],
            'itteration': info["itteration"],
            'current_element': None if self.eval_mode else info["output_file"],
            'dict_vars': info["dict_vars"],
            'initial_number_of_particles': info["initial_number_of_particles"],
            'number_of_particles': info["number_of_particles"]
        }
        self.logger.writerow(ep_info)
        self.file_handler.flush()


    def _get_returned_action_mask(self, count, offset=0):
        """
        4-D mask for actor/critic, independent of self.action_mask/self.mask.
        Action layout: [K1, HKICK, VKICK, FSE]

        Channels are selected from the actual parameters present on the current
        occurrence. When reset_specific_keys_bool is true, channels listed in
        reset_keys stay reset to zero.
        """
        if hasattr(self, "magnet_action_registry"):
            mask = self.magnet_action_registry.policy_mask(
                self.iteration,
                reset_specific_keys_bool=self.reset_specific_keys_bool,
                offset=offset,
                dtype=np.float64,
                reset_keys=self.reset_keys,
            )
            available = self.magnet_action_registry.availability_policy_mask(
                self.iteration, offset=offset, dtype=np.float64
            )
            if int(available.sum()) != count:
                raise RuntimeError(
                    f"Expected {count} action channels, found {int(available.sum())}"
                )
            return mask
        mask = np.zeros(self.action_space.n, dtype=np.float64)
        indices = self._action_indices_for_iteration(offset)
        if len(indices) != count:
            raise RuntimeError(
                f"Expected {count} action channels, found {len(indices)}"
            )
        enabled = reset_policy_mask(
            self.reset_specific_keys_bool,
            dtype=np.float64,
            reset_keys=self.reset_keys,
        )
        for index in indices:
            if enabled[index]:
                mask[index] = 1.0
        return mask

    def step(self, action, convert=False):
        """
        Perform a training step in the environment with stage learning applied.
        """
        # Keep the policy-space action separate from the converted simulator
        # action. The replay buffer always expects the fixed 4-D layout:
        # [K1, HKICK, VKICK, FSE].
        returned_action = np.asarray(action, dtype=np.float64).ravel().copy()
        if convert:
            action = self._convert_variables(action)
        # Determine the number of variables to set in this iteration
        num_of_vars_iteration_current, num_of_vars_iteration_next = self._check_number_of_variables_to_be_set_at_this_iteration()

        if num_of_vars_iteration_current not in (1, 2, 3, 4):
            raise RuntimeError(
                "The current controllable element must expose one to four "
                f"variables; got {num_of_vars_iteration_current} at iteration {self.iteration}"
            )
        if num_of_vars_iteration_next not in (0, 1, 2, 3, 4):
            raise RuntimeError(
                "The next controllable element must expose zero to four "
                f"variables; got {num_of_vars_iteration_next} at iteration {self.iteration}"
            )

        # These actor/critic masks are deliberately independent of
        # self.action_mask and self.mask, which are used by the environment.
        returned_mask = self._get_returned_action_mask(
            num_of_vars_iteration_current, offset=0
        )
        returned_mask_next = self._get_returned_action_mask(
            num_of_vars_iteration_next, offset=1
        )

        if returned_action.size != self.action_space.n:
            raise ValueError(
                f"Expected a {self.action_space.n}-D action, got {returned_action.size} values"
            )

        # Preserve the fixed 4-D shape and zero only the policy components that
        # do not apply to the current element.
        returned_action *= returned_mask
        effective_action = np.asarray(action, dtype=np.float64).reshape(-1) * returned_mask
        self.mask_len = self.previous_mask_len + num_of_vars_iteration_current
        self.action_mask = self._get_action_mask(
            num_of_vars_iteration_current, offset=0
        )
        new_actions = self._get_new_action(
            num_of_vars_iteration_current, effective_action, offset=0
        )

        self.mask = self._get_mask(self.mask_len)


        self.actions_ = self._correct_action(new_actions)

        # Apply the stage mask to restrict actions based on the current stage
        if self.stage_mask is not None:
            self.actions_ = self.actions_ * self.stage_mask

        self.previous_mask_len = self.mask_len
        elegant_input, success, dict_vars = self.wrapper.run_elegant_simulation(self.actions_)

        if not os.path.exists(self.inputs_dir):
            os.makedirs(self.inputs_dir)

        if elegant_input is not None:
            try:
                with open(os.path.join(self.inputs_dir, "elegant_input" + str(self.iteration) + ".txt"), "w") as file:
                    file.write(elegant_input)
            except (TypeError, OSError) as e:
                logger.warning("Could not save elegant input debug file: %s", e)

        if success:
            observations_dataframe, reward, output_file, done = self.wrapper.get_results(
                self.initial_number_of_particles
            )
        else:
            observations_dataframe, reward, output_file, done = None, 0, None, True

        if not success or observations_dataframe is None:
            self._log_simulation_error(
                "step",
                self.wrapper.last_elegant_error or f"iteration={self.iteration}",
            )
            self.observation = self._fallback_observation()
            self.done = True
            self.reward = -1.0
            info = {
                "dict_vars": dict_vars if dict_vars is not None else {},
                "actions": self.actions_,
                "itteration": self.iteration,
                "reward": self.reward,
                "output_file": output_file,
                "done": self.done,
                "masked_action": self.actions_,
                "observations_dataframe": self.observation,
                "initial_number_of_particles": self.initial_number_of_particles,
                "number_of_particles": self.wrapper.get_num_particles(),
                "returned_action": returned_action,
                "returned_mask": returned_mask,
                "returned_mask_next": returned_mask_next,
                "available_action_mask": self.action_mask,
                "reset_action_mask": reset_policy_mask(
                    self.reset_specific_keys_bool,
                    reset_keys=self.reset_keys,
                ),
                "environment_action_mask": self.magnet_action_registry.environment_mask(
                    self.iteration,
                    reset_specific_keys_bool=self.reset_specific_keys_bool,
                    dtype=np.float64,
                    reset_keys=self.reset_keys,
                ),
            }
            self._log_episode(info)
            return self.observation, float(self.reward), True, False, info

        self.number_of_particle_curr = reward

        reward= self.number_of_particle_curr/ self.initial_reward


        #reward= (reward)/ self.initial_reward
        #print("reward <= 3 ==", reward <= 3)
        self.done= done
        if self.number_of_particle_curr <= 3:
            self.done = True  # We lost all particles
            reward -= math.sqrt(abs((self.max_num_of_vars**2)- ((self.iteration+1)**2)))*(1/self.max_num_of_vars) #add penality for losing all particles before the end of the beamline
            #reward = -1 * math.sqrt(abs((self.max_num_of_vars**2)- ((self.iteration+1)**2)))*(100/self.max_num_of_vars) #add penality for losing all particles before the end of the beamline


        #also reward >3:  fix this
        elif self.number_of_particle_curr > 3:
            reward *=  self.number_of_particle_curr/ self.number_of_particle_prev

            #this means reward= (curr_particles/ inital_particls)+ (curr_particles/prev_particles)

            self.number_of_particle_prev= self.number_of_particle_curr
            if output_file == "final_WP":
                #reward = round((self.iteration+1) * reward, 3)
                ####reward = round(1 * reward, 3) # add extra one percent
                #if output_file ==  "final_WP":#reached the end without losing all particles
                    #reward += temp_particles  ## add an extra positive feedback to the reward
                self.done = True  # We reached the end of the beamline

        self.reward = reward # sum of rewards

        # self.reward += reward
        # ReMOVE THIS LINE
        if observations_dataframe is not None:
            #observations_dataframe.to_csv('observs/REMOVE_observations_' + str(self.iteration) + "__" + output_file + '.csv', index=False)

            # Preprocess the observation data

            self.observation = self._validated_observation(
                observations_dataframe, f"step {self.iteration}"
            )

            #self.reward is actual reward "independant of the number of initial particles"
            #whereas reward is the previous reward defined by previous function
        elif self.observation is None:
            self.observation = self._fallback_observation()
            self._log_simulation_error("step", f"missing observation at iteration={self.iteration}")

        #UNDERSTANDING ITERATIONS A BIT
        #extra precautions so we don't get over the range of variables that we can set. in reality
        #there should be two iterations one for watch points "this defines how many states we have
        #and how many watch points" the other one is for elements "this one defines how many magnets
        #do we have" in theory they should be the same but in reality we donc't care about the first
        #observation so stage=2 checks until WBM1_1, and sets the first three elements so these two iterations are one number apart
        if (self.iteration< self.max_num_of_vars):
            self.iteration += 1
        else:
            print("itteration is over the max length of the beamline")
            self.done= True
            print("we set done to True")

        if self.stage == self.iteration:
            self.done= True

        #'reward', 'initial number of particles','number of particles', 'done', 'itteration','current_element','dict_vars'
        # Fill info dictionary
        info = {
            "dict_vars": dict_vars,
            "actions": self.actions_,
            "itteration": self.iteration,
            "reward": self.reward,
            "output_file": output_file,
            "done": self.done,
            "masked_action": self.actions_,
            "observations_dataframe": observations_dataframe,
            "initial_number_of_particles": self.initial_number_of_particles,
            "number_of_particles": self.wrapper.get_num_particles(),
            "returned_action": returned_action,
            "returned_mask": returned_mask,
            "returned_mask_next": returned_mask_next,
            "available_action_mask": self.action_mask,
            "reset_action_mask": reset_policy_mask(
                self.reset_specific_keys_bool,
                reset_keys=self.reset_keys,
            ),
            "environment_action_mask": self.magnet_action_registry.environment_mask(
                max(self.iteration - 1, 0),
                reset_specific_keys_bool=self.reset_specific_keys_bool,
                dtype=np.float64,
                reset_keys=self.reset_keys,
            ),
        }

        # Evaluations have their own CSV files. Keep the main environment CSV
        # limited to noisy training episodes.
        if self.done and not self.eval_mode:
            self._log_episode(info)


        return self.observation, float(self.reward), self.done, False, info

    def get_number_of_particles(self):
        return self.wrapper.num_particles

    def stage_learning(self):
        #this function is for curriculim learning, we will train to solve the first stage then make
        #our task more complicated incrementally by increasing the stages.
        pass
