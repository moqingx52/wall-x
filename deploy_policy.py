import asyncio
import threading
import time
from typing import Any, Dict

import numpy as np

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError as exc:
    raise ImportError("wall-x deploy_policy requires msgpack and msgpack-numpy") from exc

try:
    import websockets
except ImportError as exc:
    raise ImportError("wall-x deploy_policy requires websockets") from exc


def encode_obs(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "head_rgb": observation["observation"]["head_camera"]["rgb"],
        "left_rgb": observation["observation"]["left_camera"]["rgb"],
        "right_rgb": observation["observation"]["right_camera"]["rgb"],
        "state": observation["joint_action"]["vector"],
    }


class WallXRemoteModel:
    def __init__(self, usr_args: Dict[str, Any]):
        self.uri = usr_args["wallx_server_uri"]
        self.request_timeout_s = float(usr_args.get("wallx_request_timeout_s", 60.0))
        self.connect_timeout_s = float(usr_args.get("wallx_connect_timeout_s", 60.0))
        self.dataset_names = usr_args.get("wallx_dataset_names", "x2_normal")
        self.action_dim = int(usr_args.get("wallx_action_dim", 14))
        self.state_dim = int(usr_args.get("wallx_state_dim", 14))
        self.exec_horizon = int(usr_args.get("wallx_exec_horizon", 4))
        self.action_scale = float(usr_args.get("wallx_action_scale", 1.0))
        self.strict_action_dim = bool(usr_args.get("wallx_strict_action_dim", True))
        self.camera_key_head = usr_args.get("wallx_camera_key_head", "face_view")
        self.camera_key_left = usr_args.get("wallx_camera_key_left", "left_wrist_view")
        self.camera_key_right = usr_args.get("wallx_camera_key_right", "right_wrist_view")

        self._websocket = None
        self._metadata = None
        self._loop = None
        self._thread = None
        self._instruction = None

    async def _connect(self):
        if self._websocket is not None:
            return
        self._websocket = await websockets.connect(
            self.uri,
            open_timeout=self.connect_timeout_s,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )
        hello = await asyncio.wait_for(self._websocket.recv(), timeout=self.request_timeout_s)
        self._metadata = msgpack.unpackb(hello)

    async def _predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._websocket is None:
            await self._connect()
        assert self._websocket is not None
        await asyncio.wait_for(
            self._websocket.send(msgpack.packb(payload)),
            timeout=self.request_timeout_s,
        )
        response_raw = await asyncio.wait_for(
            self._websocket.recv(),
            timeout=self.request_timeout_s,
        )
        response = msgpack.unpackb(response_raw)
        if isinstance(response, bytes):
            raise RuntimeError(response.decode("utf-8", errors="ignore"))
        return response

    async def _close(self):
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

    def _start_background_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(target=self._start_background_loop, daemon=True)
            self._thread.start()
            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.request_timeout_s + 5.0)

    def _prepare_state(self, obs: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.ndim != 1:
            state = state.reshape(-1)
        if state.shape[0] < self.state_dim:
            raise ValueError(
                f"state dim too small: got {state.shape[0]}, expected >= {self.state_dim}"
            )
        return state[: self.state_dim]

    def _parse_actions(self, response: Dict[str, Any]) -> np.ndarray:
        if "predict_action" in response:
            action = np.asarray(response["predict_action"], dtype=np.float32)
            if action.ndim == 3:
                action = action[0]
        elif "action" in response:
            action = np.asarray(response["action"], dtype=np.float32)
        else:
            raise KeyError(f"wall-x response missing action field: keys={list(response.keys())}")

        if action.ndim != 2:
            raise ValueError(f"invalid action shape: {action.shape}, expected [T, A]")

        if action.shape[1] < self.action_dim:
            raise ValueError(
                f"action dim too small: got {action.shape[1]}, expected >= {self.action_dim}"
            )
        if self.strict_action_dim and action.shape[1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {action.shape[1]}, expected {self.action_dim}"
            )
        return action[:, : self.action_dim]

    def predict_actions(self, obs: Dict[str, Any], instruction: str) -> np.ndarray:
        state = self._prepare_state(obs)
        payload = {
            self.camera_key_head: obs["head_rgb"],
            self.camera_key_left: obs["left_rgb"],
            self.camera_key_right: obs["right_rgb"],
            "prompt": instruction,
            "state": state,
            "dataset_names": self.dataset_names,
        }
        response = self._run_async(self._predict(payload))
        return self._parse_actions(response)

    def reset(self):
        self._instruction = None

    def close(self):
        try:
            self._run_async(self._close())
        except Exception:
            pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        self._thread = None
        self._websocket = None
        self._metadata = None


def get_model(usr_args: Dict[str, Any]):
    return WallXRemoteModel(usr_args)


def eval(TASK_ENV, model: WallXRemoteModel, observation: Dict[str, Any]):
    if model._instruction is None:
        model._instruction = TASK_ENV.get_instruction()

    obs = encode_obs(observation)
    actions = model.predict_actions(obs, model._instruction)
    execute_steps = min(model.exec_horizon, actions.shape[0])

    for idx in range(execute_steps):
        action = actions[idx].astype(np.float32, copy=False) * model.action_scale
        TASK_ENV.take_action(action)
        if idx != execute_steps - 1:
            observation = TASK_ENV.get_obs()
            encode_obs(observation)


def reset_model(model: WallXRemoteModel):
    model.reset()
