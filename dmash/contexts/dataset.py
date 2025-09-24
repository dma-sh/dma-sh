from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding


DataType = Union[np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def _insert_recursively_episodic(
    dataset_dict: DatasetDict, data_dict: DatasetDict,
    insert_index: int, episode_index: int, fill_until_insert_index: bool
):
    if isinstance(dataset_dict, np.ndarray):
        if fill_until_insert_index:
            assert insert_index + 1 == data_dict.shape[0]
            dataset_dict[episode_index, :insert_index + 1] = data_dict
        else:
            if len(data_dict.shape) == 0:
                dataset_dict[episode_index, insert_index] = data_dict
            elif data_dict.shape[0] == 1:
                dataset_dict[episode_index, insert_index] = data_dict.squeeze()
            else:
                raise TypeError()
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            if k in data_dict.keys():
                _insert_recursively_episodic(
                    dataset_dict[k], data_dict[k], insert_index, episode_index, fill_until_insert_index
                )
    else:
        raise TypeError()


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> Dict:
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return batch

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        assert 0 < ratio and ratio < 1
        train_index = np.index_exp[: int(self.dataset_len * ratio)]
        test_index = np.index_exp[int(self.dataset_len * ratio):]

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio):]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        data_dict = data_dict.copy()
        if "info" in data_dict:
            data_dict.pop("info")
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)


class ContextReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        context_size: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(next_observation_space, gym.spaces.Box)

        dataset_dict = dict(
            observations=np.empty((capacity, context_size, *observation_space.shape), dtype=observation_space.dtype),
            next_observations=np.empty((capacity, context_size, *observation_space.shape), dtype=observation_space.dtype),
            actions=np.empty((capacity, context_size, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity, context_size,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=np.float32),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        data_dict = data_dict.copy()
        if "info" in data_dict:
            data_dict.pop("info")
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)


class MemoryEfficientContextReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        context_size: int,
        episode_length: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(next_observation_space, gym.spaces.Box)

        # capacity which is used in the following refers to number of episodes
        capacity = int(capacity / episode_length)

        dataset_dict = dict(
            observations=np.empty((capacity, context_size + episode_length - 1, *observation_space.shape), dtype=observation_space.dtype),
            next_observations=np.empty((capacity, context_size + episode_length - 1, *observation_space.shape), dtype=observation_space.dtype),
            actions=np.empty((capacity, context_size + episode_length - 1, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity, context_size + episode_length - 1,), dtype=np.float32),
            masks=np.empty((capacity, context_size + episode_length - 1), dtype=np.float32),
            dones=np.empty((capacity, context_size + episode_length - 1), dtype=np.float32),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = np.ones(capacity, dtype=np.int32) * (context_size - 1)
        self.dataset_episode_indices = np.empty((capacity,), dtype=np.int32)
        self._episode_index = 0
        self._fill_warm_up_context_window = True
        self._context_size = context_size

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        data_dict = data_dict.copy()
        if "info" in data_dict:
            data_dict.pop("info")

        data_dict_transition = {}
        data_dict_rest = {}
        for k, v in data_dict.items():
            if k in ["masks", "dones"]:
                data_dict_rest[k] = v
            else:
                data_dict_transition[k] = v
        if self._fill_warm_up_context_window:
            _insert_recursively_episodic(
                self.dataset_dict,
                data_dict_transition,
                episode_index=self._episode_index,
                insert_index=self._insert_index[self._episode_index],
                fill_until_insert_index=True
            )
            _insert_recursively_episodic(
                self.dataset_dict,
                data_dict_rest,
                episode_index=self._episode_index,
                insert_index=self._insert_index[self._episode_index],
                fill_until_insert_index=False
            )
            self._fill_warm_up_context_window = False
        else:
            data_dict_transition = {k: v[-1:] for k, v in data_dict_transition.items()}
            _insert_recursively_episodic(
                self.dataset_dict,
                data_dict_transition,
                episode_index=self._episode_index,
                insert_index=self._insert_index[self._episode_index],
                fill_until_insert_index=False
            )
            _insert_recursively_episodic(
                self.dataset_dict,
                data_dict_rest,
                episode_index=self._episode_index,
                insert_index=self._insert_index[self._episode_index],
                fill_until_insert_index=False
            )

        if data_dict["dones"]:
            self._episode_index = (self._episode_index + 1) % self._capacity
            self._insert_index[self._episode_index] = self._context_size - 1
            self._size = min(self._size + 1, self._capacity)
            self._fill_warm_up_context_window = True
        else:
            self._insert_index[self._episode_index] += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        episode_index: Optional[np.ndarray] = None,
    ) -> Dict:
        if episode_index is None:
            episode_index = self.np_random.integers(len(self), size=batch_size)

        within_episode_index = self.np_random.integers(
            low=self._context_size - 1, high=self._insert_index[episode_index] + 1
        )
        episode_index_repeated = np.repeat(episode_index, self._context_size)
        within_episode_index_range = np.hstack(
            [np.arange(v - self._context_size + 1, v + 1) for v in within_episode_index]
        )
        assert (within_episode_index_range >= 0).all()

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if k in ["masks", "dones"]:
                batch[k] = self.dataset_dict[k][episode_index, within_episode_index]
            elif k == "rewards":
                batch[k] = self.dataset_dict[k][episode_index_repeated, within_episode_index_range]
                K_, = batch[k].shape
                assert int(K_ / self._context_size) == batch_size
                batch[k] = batch[k].reshape(batch_size, self._context_size)
            else:
                batch[k] = self.dataset_dict[k][episode_index_repeated, within_episode_index_range]
                K_, D = batch[k].shape
                assert int(K_ / self._context_size) == batch_size
                batch[k] = batch[k].reshape(batch_size, self._context_size, D)

        return batch


class MultiContextReplayBuffer(object):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        context_size: int,
        num_datasets: int,
        memory_efficient: bool = True,
        episode_length: Optional[int] = None,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if memory_efficient:
            assert episode_length is not None
            self.dataset_dicts = [
                MemoryEfficientContextReplayBuffer(
                    observation_space,
                    action_space,
                    capacity,
                    context_size,
                    episode_length
                ) for _ in range(num_datasets)
            ]
        else:
            self.dataset_dicts = [
                ContextReplayBuffer(
                    observation_space,
                    action_space,
                    capacity,
                    context_size,
                ) for _ in range(num_datasets)
            ]

        self.num_datasets = num_datasets

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        dataset_index: Optional[int] = None,
    ) -> Dict:
        if dataset_index is not None:
            return self.dataset_dicts[dataset_index].sample(batch_size, keys, indx)
        else:
            batches = [
                dataset_dict.sample(
                    batch_size // self.num_datasets, keys, indx
                ) for dataset_dict in self.dataset_dicts
            ]
            combined_batch = {key: [] for key in batches[0].keys()}
            for d in batches:
                for key, value in d.items():
                    combined_batch[key].append(value)
            shuffle_indx = self.dataset_dicts[0].np_random.permutation(batch_size // self.num_datasets * self.num_datasets)
            for key in combined_batch:
                combined_batch[key] = np.concatenate(combined_batch[key], axis=0)[shuffle_indx]
            return combined_batch

    def insert(self, data_dict: DatasetDict, dataset_index: int):
        self.dataset_dicts[dataset_index].insert(data_dict)

    def split(self, ratio: float, dataset_index: int) -> Tuple["Dataset", "Dataset"]:
        return self.dataset_dicts[dataset_index].split(ratio)

    def seed(self, seed: Optional[int] = None) -> list[list]:
        seeds = []
        for dataset_dict in self.dataset_dicts:
            _seed = dataset_dict.seed(seed)
        seeds.append(_seed)
        return seeds
