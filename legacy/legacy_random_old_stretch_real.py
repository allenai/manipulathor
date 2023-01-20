def normalize_real_kinect_image(frame,size=224):
    assert (frame.shape[0], frame.shape[1]) == (KINECT_REAL_W, KINECT_REAL_H)
    current_size = frame.shape
    bigger_size = max(current_size[0], current_size[1])
    ratio = size / bigger_size
    w,h = (int(current_size[0] * ratio), int(current_size[1] * ratio))

    frame = cv2.resize(frame,(h,w))
    if len(frame.shape) == 3:
        result = np.zeros((size, size, frame.shape[2]))
    elif len(frame.shape) == 2:
        result = np.zeros((size, size))
    start_w = int(size / 2 - w / 2)
    end_w = start_w + w
    start_h = int(size / 2 - h / 2)
    end_h = start_h + h
    result[start_w:end_w,start_h:end_h] = frame
    if len(frame.shape) == 2: #it is depth image
        result[result > MAX_KINECT_DEPTH] = MAX_KINECT_DEPTH
        result[result < MIN_KINECT_DEPTH] = 0
    return result.astype(frame.dtype)

class KinectArmMaskSensor(Sensor):
    def __init__(self, uuid: str = "arm_mask_kinect", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        kinect_raw_value = env.kinect_raw_frame
        mask = get_binary_mask_of_arm(kinect_raw_value)
        mask = normalize_real_kinect_image(mask)
        return np.expand_dims(mask, axis=-1)

def kinect_reshape(frame):
    frame = frame.copy()

    desired_w, desired_h = KINECT_RESIZED_W, KINECT_RESIZED_H
    # original_size = desired_h
    assert frame.shape[0] == frame.shape[1]
    original_size = frame.shape[0]
    fraction = max(desired_h, desired_w) / original_size
    beginning = original_size / 2 - desired_w / fraction / 2
    end = original_size / 2 + desired_w / fraction / 2
    frame[:int(beginning), :] = 0
    frame[int(end):, :] = 0
    if len(frame.shape) == 2: #it is depth image
        frame[frame > MAX_KINECT_DEPTH] = MAX_KINECT_DEPTH
        frame[frame < MIN_KINECT_DEPTH] = 0
    # if len(frame.shape) == 3 and frame.shape[2] == 3:
    #     # frame = clip_rgb_kinect_frame(frame)
    #     pass
    if (len(frame.shape) == 3 and frame.shape[2] ==1) or len(frame.shape) == 2:
        cropped_frame = frame[int(beginning):int(end), :]
        frame[int(beginning):int(end), :] = clip_depth_kinect_frame(cropped_frame)

    return frame

 @property
def kinect_frame(self) -> np.ndarray:
    """Returns rgb image corresponding to the agent's egocentric view."""
    frame = self.controller.last_event.third_party_camera_frames[0].copy()
    frame = remove_nan_inf_for_frames(frame, 'kinect_frame')
    return kinect_reshape(frame)
@property
def kinect_depth(self) -> np.ndarray:
    """Returns rgb image corresponding to the agent's egocentric view."""
    depth_frame = self.controller.last_event.third_party_depth_frames[0].copy()
    depth_frame = remove_nan_inf_for_frames(depth_frame, 'depth_kinect')

    if np.sum(depth_frame != self.controller.last_event.third_party_depth_frames[0].copy()) > 10:
        raise Exception('Depth is nan again even after removing nan?')

    return kinect_reshape(depth_frame)


class RealKinectArmPointNavEmulSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor, depth_sensor:Sensor, arm_mask_sensor:Sensor, uuid: str = "arm_point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        self.depth_sensor = depth_sensor
        self.arm_mask_sensor = arm_mask_sensor
        uuid = '{}_{}'.format(uuid, type)

        self.min_xyz = np.zeros((3))

        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")
        super().__init__(**prepare_locals_for_super(locals()))
    def get_camera_int_ext(self,env):
        #TODO all these values need to be checked
        fov=max(KINECT_FOV_W, KINECT_FOV_H)#TODO are you sure? it should be smaller one I think
        agent_state = env.controller.last_event.metadata['agent']
        camera_horizon = 45
        camera_xyz = np.array([agent_state['position'][k] for k in ['x','y','z']])
        camera_rotation = (agent_state['rotation']['y'] + 90) % 360
        return fov, camera_horizon, camera_xyz, camera_rotation
    def get_observation(
            self, env: StretchRealEnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        #TODO remove
        if self.type == 'destination':
            return self.dummy_answer
        mask = (self.mask_sensor.get_observation(env, task, *args, **kwargs)) #TODO this is called multiple times?
        depth_frame_original = self.depth_sensor.get_observation(env, task, *args, **kwargs).squeeze(-1)

        if task.num_steps_taken() == 0:
            self.pointnav_history_aggr = []
            self.real_prev_location = None
            self.belief_prev_location = None


        fov, camera_horizon, camera_xyz, camera_rotation = self.get_camera_int_ext(env)
        arm_world_coord = None

        if mask.sum() != 0:

            midpoint_agent_coord = get_mid_point_of_object_from_depth_and_mask(mask, depth_frame_original, self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device)
            if not torch.any(torch.isnan(midpoint_agent_coord) + torch.isinf(midpoint_agent_coord)):
                self.pointnav_history_aggr.append((midpoint_agent_coord.cpu(), 1, task.num_steps_taken()))

        arm_mask = self.arm_mask_sensor.get_observation(env, task, *args, **kwargs) #TODO this is also called twice
        if arm_mask.sum() == 0: #Do we want to do some approximations or no?
            arm_world_coord = None #TODO approax for this
        else:
            arm_world_coord = get_mid_point_of_object_from_depth_and_mask(arm_mask, depth_frame_original, self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device)
            # distance_in_agent_coord = midpoint_agent_coord - arm_location_in_camera
            # result = distance_in_agent_coord.cpu()
        # if arm_mask.sum() != 0 or mask.sum() != 0:
        #
        #     import cv2
        #     cv2.imwrite('/Users/kianae/Desktop/image.png', env.kinect_frame[:,:,::-1])
        #     cv2.imwrite('/Users/kianae/Desktop/mask.png', mask.squeeze().numpy() * 255)
        #     cv2.imwrite('/Users/kianae/Desktop/arm_mask.png', arm_mask * 255)
        result = self.history_aggregation(camera_xyz, camera_rotation, arm_world_coord, task.num_steps_taken())
        return result
    def history_aggregation(self, camera_xyz, camera_rotation, arm_world_coord, current_step_number):
        if len(self.pointnav_history_aggr) == 0 or arm_world_coord is None:
            return self.dummy_answer
        else:
            weights = [1. / (current_step_number + 1 - num_steps) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_weights = sum(weights)
            total_sum = [mid * (1. / (current_step_number + 1 - num_steps)) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_sum = sum(total_sum)
            midpoint = total_sum / total_weights
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)
            midpoint_agent_coord = torch.Tensor([midpoint_agent_coord['position'][k] for k in ['x','y','z']])

            arm_world_coord = dict(position=dict(x=arm_world_coord[0], y=arm_world_coord[1], z=arm_world_coord[2]), rotation=dict(x=0,y=0,z=0))
            arm_state_agent_coord = convert_world_to_agent_coordinate(arm_world_coord, agent_state)
            arm_state_agent_coord = torch.Tensor([arm_state_agent_coord['position'][k] for k in ['x','y','z']])

            distance_in_agent_coord = midpoint_agent_coord - arm_state_agent_coord

            return distance_in_agent_coord.cpu()



class RealIntelAgentBodyPointNavEmulSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor, depth_sensor:Sensor, uuid: str = "point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        self.depth_sensor = depth_sensor
        uuid = '{}_{}'.format(uuid, type)

        self.min_xyz = np.zeros((3))
        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")


        super().__init__(**prepare_locals_for_super(locals()))
    def get_camera_int_ext(self,env):
        #TODO all these values need to be checked
        fov=max(INTEL_FOV_W, INTEL_FOV_H) #TODO are you sure? it should be smaller one I think
        agent_state = env.controller.last_event.metadata['agent']
        camera_horizon = 0
        camera_xyz = np.array([agent_state['position'][k] for k in ['x','y','z']])
        camera_rotation = agent_state['rotation']['y']
        return fov, camera_horizon, camera_xyz, camera_rotation




    def get_observation(
            self, env: StretchRealEnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask = (self.mask_sensor.get_observation(env, task, *args, **kwargs))

        if task.num_steps_taken() == 0:
            self.pointnav_history_aggr = []
            self.real_prev_location = None
            self.belief_prev_location = None


        fov, camera_horizon, camera_xyz, camera_rotation = self.get_camera_int_ext(env)


        if mask.sum() != 0:
            depth_frame_original = self.depth_sensor.get_observation(env, task, *args, **kwargs).squeeze(-1)
            middle_of_object = get_mid_point_of_object_from_depth_and_mask(mask, depth_frame_original, self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device)
            if not (torch.any(torch.isnan(middle_of_object) + torch.isinf(middle_of_object))): #TODO NOW double check
                self.pointnav_history_aggr.append((middle_of_object.cpu(), 1, task.num_steps_taken()))

            # result = middle_of_object.cpu()
        # else:
        #     result = self.dummy_answer
        result = self.history_aggregation(camera_xyz, camera_rotation, task.num_steps_taken())
        return result
    def history_aggregation(self, camera_xyz, camera_rotation, current_step_number):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            weights = [1. / (current_step_number + 1 - num_steps) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_weights = sum(weights)
            total_sum = [mid * (1. / (current_step_number + 1 - num_steps)) for mid,num_pixels,num_steps in self.pointnav_history_aggr]
            total_sum = sum(total_sum)
            midpoint = total_sum / total_weights
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            distance_in_agent_coord = dict(x=midpoint_agent_coord['position']['x'], y=midpoint_agent_coord['position']['y'], z=midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            agent_centric_middle_of_object = agent_centric_middle_of_object
            return agent_centric_middle_of_object




#With open CV
# def get_observation(
#         self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
# ) -> Any:
#     if env.last_image_changed is False and self.cache is not None:
#         return self.cache
#
#     # mask = np.zeros((224, 224, 1))
#     # mask[90:110, 90:110] = 1
#     # mask[90:110, :20] = 1
#     img = env.current_frame
#     resized_image = cv2.resize(img, dsize=(224,224))
#     global input_received, center_x, center_y
#     center_x, center_y = -1, -1
#     input_received = False
#
#     def on_click(event, x, y, p1, p2):
#         global center_x, center_y, input_received
#         if event == cv2.EVENT_LBUTTONDOWN:
#             center_x = x
#             center_y = y
#             print((x, y))
#             input_received = True
#             cv2.destroyWindow("image")
#             cv2.destroyAllWindows()
#
#         if event == cv2.EVENT_RBUTTONDOWN:
#             center_x = -1
#             center_y = -1
#             print((-1,-1))
#             input_received = True
#
#     def normalize_number(x, w):
#         if x < 0:
#             x = 0
#         if x > w:
#             x = w
#         return x
#
#     cv2.imshow("image", resized_image[:,:,[2,1,0]])
#     cv2.setMouseCallback('image', on_click)
#     while not input_received:
#         k = cv2.waitKey(100)
#         # if k == 27:
#         #     print('ESC')
#         #     cv2.destroyAllWindows()
#         #     break
#         # if cv2.getWindowProperty('image',1) == -1 :
#         #     break
#     cv2.destroyWindow("image")
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)
#
#     self.window_size = 20 TODO do I want to change the size of this one maybe?
#     mask = np.zeros((224, 224, 1))
#     if center_y == -1 and center_x == -1:
#         mask[:,:] = 0.
#     else:
#         offset = self.window_size / 2
#         object_boundaries = center_x - offset, center_y - offset, center_x + offset, center_y + offset
#         x1, y1, x2, y2 = [int(normalize_number(i, 224)) for i in object_boundaries]
#         mask[y1:y2, x1:x2] = 1.
#     self.cache = mask
#     return mask