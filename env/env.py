import socket, os
import struct
import time
import bson
import cv2
import numpy as np
import gym
import robot
import pprint

class CMD:
    STEP = "step"
    RESET = "reset"
    INFO = "info"
    INIT = "init"
    SKIP_FRAME = "skip"
    GET_STATE = "getstate"

class UEGym():

    def __init__(self, ip = "127.0.0.1", port=5000):

        self._ip = ip
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._socket.bind((self._ip, self._port))
        self._socket.listen(1)

        print(f"Waiting for connection on: {self._ip}:{self._port}")
        self._connection, self._address = self._socket.accept() # Wait For Conenction
        print("-"*40)
        print(f"Connected! Adress: {self._address}.")
        print("-" * 40)

        self._robot_list = []
        self._robot_initial_pose = [1.57, -0.35, 3.14, -2.0, 0, -1.0,  1.57, 0]
        self.action_size = None

        self._episode_len = 100 # n steps per episode
        self._nStepsInit = 350  # Frames to advance on initialization to ensure robot is in initial position
        self._nFramesToSkipPerStep = 1

    def recv_msg(self, waitResponse = True) -> dict:
        raw_msgLen = self._connection.recv(4) # read first 4-bytes that represent size of the remaining buffer
        if not raw_msgLen:
            return dict()

        msgLen = int.from_bytes(raw_msgLen, byteorder="big")
        if waitResponse: time.sleep(0.08)

        ret = self._connection.recv(msgLen)
        return bson.decode(ret)

    def send_msg(self, msg : dict = None, cmd = None):
        if cmd is not None:
            msg["cmd"] = cmd

        binaryMsg = bson.encode(msg)
        self._connection.send(binaryMsg)

    def requestInfo(self):

        self.send_msg(msg=dict(), cmd=CMD.INFO)
        print("[LOG] Waiting for response on info request...")
        response= self.recv_msg()

        self._robot_list = []
        self.action_size = None
        for key in response:
            robot_info = response[key]
            newRobot = robot.Robot(robot_info["name"],
                                   robot_info["controllers"],
                                   robot_info["lower"],
                                   robot_info["upper"])

            self._robot_list.append(newRobot)

            if self.action_size == None:
                self.action_size = len(robot_info["lower"])
        print("[LOG] Information gathered.")
        #return self._robot_list

    def init(self, initial_pose : np.ndarray):
        if type(initial_pose) != np.ndarray:
            raise ValueError(f"Actions must be of type 'np.array', but instead got {type(initial_pose)}")

        actions_dict = self._parse_actions(initial_pose)
        actions_dict["len"] = self._episode_len

        self.send_msg(actions_dict, cmd=CMD.INIT)
        self.recv_msg() # Wait for response

        # Skip N frames (wait for robot to get into initial position
        self._skip_n_frames(self._nStepsInit)

        self.send_msg(dict(), cmd=CMD.GET_STATE)
        observation_dict = self.recv_msg()
        return self._parse_observations(observation_dict)

    def reset(self):
        self.send_msg(dict(), cmd=CMD.RESET)
        self.recv_msg() # Wait for response on 'reset' to confirm it's done

        self.requestInfo()

        n_robots = len(self._robot_list)
        initial_pose_array = np.asarray([self._robot_initial_pose * n_robots],
                                        dtype=np.float64).reshape(n_robots, self.action_size)

        return self.init(initial_pose_array)[0] # return only states, without rewards and dones

    def step(self, actions : np.ndarray):
        if type(actions) != np.ndarray:
            raise ValueError(f"Actions must be of type 'np.array', but instead got {type(actions)}")

        actions_dict = self._parse_actions(actions)
        self.send_msg(actions_dict, cmd=CMD.STEP)
        self.recv_msg()

        self._skip_n_frames(self._nFramesToSkipPerStep)

        self.send_msg(dict(), cmd=CMD.GET_STATE)
        observation_dict =  self.recv_msg()

        # for robot in observation_dict["robots"]:
        #     w, h = robot["image"]["width"], robot["image"]["height"]
        #     buf_as_np_array = np.frombuffer(robot["image"]["data"], np.uint8)
        #     rgb = buf_as_np_array.reshape((h, w, 3))
        #
        #     rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #     cv2.imshow(robot["name"], rgb)
        # cv2.waitKey(1)

        return self._parse_observations(observation_dict)

    def _skip_n_frames(self, nFrames):
        msg = {
            "frames": nFrames
        }
        for i in range(nFrames):
            self.send_msg(dict(), cmd=CMD.SKIP_FRAME)
            self.recv_msg(True) # Wait for response
        print("Done")

    def _parse_actions(self, actions : np.ndarray):
        actions_dict = {
            "robots": list()
        }
        for robot, a in zip(self._robot_list, actions):
            r_dict = robot.unscaledActionDict(a)
            actions_dict["robots"].append(r_dict)

        return actions_dict

    def _parse_observations(self, observation_dict):
        images_list = list()
        states_list = list()
        rewards = list()
        dones = list()

        for robot in observation_dict["robots"]:
            # IMAGE
            w, h = robot["image"]["width"], robot["image"]["height"]
            image_array = np.frombuffer(robot["image"]["data"], np.uint8)
            rgb_image = image_array.reshape((h, w, 3))

            images_list.append(rgb_image)

            # STATE
            positions = robot["joints"]["position"]
            velocities = robot["joints"]["velocity"]
            state = list()
            state.extend(positions)
            state.extend(velocities)

            states_list.append(state)

            rewards.append(robot["reward"])
            dones.append(robot["done"])

        next_states = {
            "images": images_list,
            "states": states_list
        }

        return next_states, rewards, dones

if __name__ == "__main__":

    env = UEGym()
    time.sleep(1) # Wait for simulation to initialize

    obs = env.reset()

    time.sleep(10)

    #timer_np = np.array(timer_list)
    #print(f"Mean: {timer_np.mean()} Std: {timer_np.std()}")
