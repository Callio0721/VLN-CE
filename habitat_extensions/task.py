import gzip
import json
import os
from typing import Dict, List, Optional, Union, Any

import attr
from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    """Loads the R2R VLN-CE dataset"""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            # cast integer IDs to strings
            episode["episode_id"] = str(episode["episode_id"])
            episode["trajectory_id"] = str(episode["trajectory_id"])

            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.DATA_PATH.format(split=config.SPLIT, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_LANGUAGES_MASK not in config.LANGUAGES:
            languages_to_load = set(config.LANGUAGES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.SPLIT
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @classmethod
    def extract_roles_from_config(cls, config: Config) -> List[str]:
        if ALL_ROLES_MASK in config.ROLES:
            return cls.annotation_roles
        assert set(config.ROLES).issubset(set(cls.annotation_roles))
        return config.ROLES

    @classmethod
    def check_config_paths_exist(cls, config: Config) -> bool:
        return all(
            os.path.exists(
                config.DATA_PATH.format(split=config.SPLIT, role=role)
            )
            for role in cls.extract_roles_from_config(config)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        """Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language


# =========================================================
# 🔥 新增部分：BERT 专用 Sensor
# 放在 task.py 的 imports 下面即可
# =========================================================
from transformers import BertTokenizer
import numpy as np
from gym import spaces
from habitat.core.simulator import Sensor, SensorTypes

# 全局缓存 Tokenizer，防止多进程重复加载卡死
_BERT_TOKENIZER = None

def get_tokenizer():
    global _BERT_TOKENIZER
    if _BERT_TOKENIZER is None:
        print("Loading BERT Tokenizer...")
        _BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    return _BERT_TOKENIZER

@registry.register_sensor(name="BertInstructionSensor")
class BertInstructionSensor(Sensor):
    def __init__(self, config: Config, dataset: Dataset, *args: Any, **kwargs: Any):
        # 1. 读取配置中的 MAX_LENGTH (比如 200)
        # 如果配置里没写，就默认 128
        self.max_length = getattr(config, "MAX_LENGTH", 128)

        print(f"🚀 Config Max Length: {self.max_length}") # 打印确认

        self._config = config
        self._dataset = dataset
        super().__init__(config=config)
        get_tokenizer()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "instruction"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TOKEN_IDS

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Space:
        # 🔥 修改点 1: 告诉 Habitat，我输出的数组长度永远固定是 self.max_length
        return spaces.Box(
            low=0,
            high=30522,
            shape=(self.max_length,), # <--- 这里必须用变量，不能写死 256
            dtype=np.int64,
        )

    def get_observation(
        self, observations: Dict[str, Any], episode: VLNEpisode, *args: Any, **kwargs: Any
    ):
        tokenizer = get_tokenizer()
        text = episode.instruction.instruction_text
        
        # 🔥 修改点 2: 真正使用 MAX_LENGTH 进行 "削足适履"
        encoded = tokenizer(
            text, 
            add_special_tokens=True,
            padding='max_length',       # <--- 关键：短了就补 0
            truncation=True,            # <--- 关键：长了就截断
            max_length=self.max_length, # <--- 关键：目标长度
            return_tensors="np"
        )
        
        token_ids = encoded['input_ids'][0]
        return token_ids.astype(np.int64)