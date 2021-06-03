

class DeprecatedEasyBringObjectTaskSampler(BringObjectAbstractTaskSampler):

    _TASK_TYPE = EasyBringObjectTask

    def __init__(self, **kwargs) -> None:
        print('resolve todos')

        super(DeprecatedEasyBringObjectTaskSampler, self).__init__(**kwargs)

        possible_initial_locations = (
            "datasets/apnd-dataset/valid_agent_initial_locations.json"
        )
        if self.sampler_mode == "test":
            possible_initial_locations = (
                "datasets/apnd-dataset/deterministic_valid_agent_initial_locations.json"
            )
        with open(possible_initial_locations) as f:
            self.possible_agent_reachable_poses = json.load(f)


        self.all_possible_points = []
        for scene in self.scenes:
            for object_pair in self.objects:
                init_object, goal_object = object_pair
                if False:
                    valid_position_adr = "datasets/apnd-dataset/valid_object_pairs/valid_{}_to_{}_in_{}.json".format(
                        init_object, goal_object, scene
                    )
                else:
                    #LATER_TODO remove this
                    valid_position_adr = "datasets/apnd-dataset/valid_object_positions/valid_{}_positions_in_{}.json".format(
                        init_object, scene
                    )
                try:
                    with open(valid_position_adr) as f:
                        data_points = json.load(f)
                except Exception:
                    print("Failed to load", valid_position_adr)
                    ForkedPdb().set_trace()
                    continue

                #LATER_TODO
                if False:
                    self.all_possible_points += data_points['locations']
                else:
                    self.all_possible_points += data_points[scene]


        scene_names = set([x['scene_name'] for x in self.all_possible_points])

        if len(set(scene_names)) < len(self.scenes):
            print("Not all scenes appear")

        print(
            "Len dataset",
            len(self.all_possible_points),
        )
        for (i, x) in enumerate(self.all_possible_points):
            x['index'] = i
        if (
                self.sampler_mode != "train"
        ):  # Be aware that this totally overrides some stuff
            self.deterministic_data_list = []
            if True:
                self.deterministic_data_list = self.all_possible_points
            else:
                #LATER_TODO later on do this for test
                for scene in self.scenes:
                    for object in self.objects:
                        valid_position_adr = "datasets/apnd-dataset/deterministic_tasks/tasks_{}_positions_in_{}.json".format(
                            object, scene
                        )
                        try:
                            with open(valid_position_adr) as f:
                                data_points = json.load(f)
                        except Exception:
                            print("Failed to load", valid_position_adr)
                            continue
                        visible_data = [
                            dict(scene=scene, index=i, datapoint=data)
                            for (i, data) in enumerate(data_points[scene])
                        ]
                        self.deterministic_data_list += visible_data

        #LATER_TODO I think this is pretty important and make sure we are fine with it, we definitely can't do that
        random.shuffle(self.all_possible_points)


        if self.sampler_mode == "test":
            random.shuffle(self.deterministic_data_list)
            self.max_tasks = self.reset_tasks = len(self.deterministic_data_list)

    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        data_point = self.get_source_target_indices()
        #LATER_TODO double check the random shuffle things we do here

        scene_name = data_point["scene_name"]
        init_location = data_point['init_location']
        goal_location = data_point['goal_location']
        agent_state = data_point["initial_agent_pose"]

        assert init_location["scene_name"] == goal_location["scene_name"] == scene_name

        if self.env is None:
            self.env = self._create_environment()

        self.env.reset(
            scene_name=scene_name, agentMode="arm", agentControllerType="mid-level"
        )

        event1, event2, event3 = initialize_arm(self.env.controller)

        this_controller = self.env

        def put_object_in_location(location_point):

            object_id = location_point['object_id']
            location = location_point['object_location']
            event = transport_wrapper(
                this_controller,
                object_id,
                location,
            )
            return event

        event = put_object_in_location(init_location)

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )

        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), BringObjImageVisualizer)
        ]

        initial_object_info = self.env.get_object_by_id(init_location["object_id"])
        initial_agent_location = self.env.controller.last_event.metadata["agent"]
        initial_hand_state = self.env.get_absolute_hand_state()

        task_info = {
            'source_object_id': init_location['object_id'],
            'goal_object_id': goal_location['object_id'],
            "init_location": init_location,
            "goal_location": goal_location,
            'agent_initial_state': initial_agent_location,
            'initial_object_location':initial_object_info,
            'initial_hand_state': initial_hand_state,
        }

        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = init_location
            task_info["visualization_target"] = goal_location

        self._last_sampled_task = self._TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        if self.sampler_mode == "train":
            return None
        else:
            return min(self.max_tasks, len(self.deterministic_data_list))

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return (
            self.total_unique - self.sampler_index
            if self.sampler_mode != "train"
            else (float("inf") if self.max_tasks is None else self.max_tasks)
        )

    def get_source_target_indices(self):
        if self.sampler_mode == "train":
            init_location = self.all_possible_points[self.sampler_index]
            data_point = dict(scene_name=init_location['scene_name'], init_location=init_location, goal_location=init_location)

            scene_name = data_point['scene_name']

            self.sampler_index += 1
            if self.sampler_index >= len(self.all_possible_points):
                self.sampler_index = 0
                random.shuffle(self.all_possible_points)


            initial_agent_pose = data_point['init_location']['agent_pose']
            # selected_agent_init_loc = random.choice(
            #     self.possible_agent_reachable_poses[scene_name]
            # )
            # initial_agent_pose = {
            #     "name": "agent",
            #     "position": {
            #         "x": selected_agent_init_loc["x"],
            #         "y": selected_agent_init_loc["y"],
            #         "z": selected_agent_init_loc["z"],
            #     },
            #     "rotation": {
            #         "x": -0.0,
            #         "y": selected_agent_init_loc["rotation"],
            #         "z": 0.0,
            #     },
            #     "cameraHorizon": selected_agent_init_loc["horizon"],
            #     "isStanding": True,
            # }
            data_point["initial_agent_pose"] = initial_agent_pose
        else:  # we need to fix this for test set, agent init location needs to be fixed, therefore we load a fixed valid agent init that is previously randomized
            init_location = self.all_possible_points[self.sampler_index]
            data_point = dict(scene_name=init_location['scene_name'], init_location=init_location, goal_location=init_location)
            initial_agent_pose = data_point['init_location']['agent_pose']

            data_point["initial_agent_pose"] = initial_agent_pose
            self.sampler_index += 1

        #LATER_TODO reomve
        data_point["initial_agent_pose"] = data_point['init_location']['agent_pose']

        return data_point


    def old_get_source_target_indices(self):
        ForkedPdb().set_trace()
        if self.sampler_mode == "train":
            data_point = self.all_possible_points[self.sampler_index]

            scene_name = data_point['scene_name']

            self.sampler_index += 1
            if self.sampler_index >= len(self.all_possible_points):
                self.sampler_index = 0
                random.shuffle(self.all_possible_points)



            selected_agent_init_loc = random.choice(
                self.possible_agent_reachable_poses[scene_name]
            )
            initial_agent_pose = {
                "name": "agent",
                "position": {
                    "x": selected_agent_init_loc["x"],
                    "y": selected_agent_init_loc["y"],
                    "z": selected_agent_init_loc["z"],
                },
                "rotation": {
                    "x": -0.0,
                    "y": selected_agent_init_loc["rotation"],
                    "z": 0.0,
                },
                "cameraHorizon": selected_agent_init_loc["horizon"],
                "isStanding": True,
            }
            data_point["initial_agent_pose"] = initial_agent_pose
        else:  # we need to fix this for test set, agent init location needs to be fixed, therefore we load a fixed valid agent init that is previously randomized
            data_point = self.all_possible_points[self.sampler_index]
            scene_name = data_point['scene_name']
            datapoint_original_index = self.all_possible_points[self.sampler_index][
                "index"
            ]
            selected_agent_init_loc = self.possible_agent_reachable_poses[scene_name][
                datapoint_original_index
            ]
            initial_agent_pose = {
                "name": "agent",
                "position": {
                    "x": selected_agent_init_loc["x"],
                    "y": selected_agent_init_loc["y"],
                    "z": selected_agent_init_loc["z"],
                },
                "rotation": {
                    "x": -0.0,
                    "y": selected_agent_init_loc["rotation"],
                    "z": 0.0,
                },
                "cameraHorizon": selected_agent_init_loc["horizon"],
                "isStanding": True,
            }
            data_point["initial_agent_pose"] = initial_agent_pose
            self.sampler_index += 1

        #LATER_TODO reomve
        data_point["initial_agent_pose"] = data_point['init_location']['agent_pose']

        return data_point


