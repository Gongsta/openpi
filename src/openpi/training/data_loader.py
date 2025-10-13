from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _validate_datasets_compatibility(
    datasets_meta: Sequence[lerobot_dataset.LeRobotDatasetMetadata],
    action_sequence_keys: Sequence[str],
) -> None:
    """Validate that multiple datasets are compatible for training together.

    Args:
        datasets_meta: Metadata for all datasets to be combined.
        action_sequence_keys: Required action sequence keys.

    Raises:
        ValueError: If datasets are incompatible.
    """
    if len(datasets_meta) < 2:
        return  # Nothing to validate

    # Check FPS consistency (allow 1% tolerance)
    base_fps = datasets_meta[0].fps
    for i, meta in enumerate(datasets_meta[1:], start=1):
        if abs(meta.fps - base_fps) / base_fps > 0.01:
            raise ValueError(
                f"Dataset FPS mismatch: dataset 0 has fps={base_fps}, "
                f"dataset {i} has fps={meta.fps}. All datasets must have the same FPS."
            )

    # Check that all datasets have required action_sequence_keys
    for i, meta in enumerate(datasets_meta):
        for key in action_sequence_keys:
            if key not in meta.features:
                raise ValueError(
                    f"Dataset {i} ({meta.repo_id}) is missing required action key '{key}'. "
                    f"Available features: {list(meta.features.keys())}"
                )

    logging.info(f"Validated {len(datasets_meta)} datasets for compatibility:")
    for i, meta in enumerate(datasets_meta):
        logging.info(f"  Dataset {i}: {meta.repo_id} ({meta.total_episodes} episodes, {meta.total_frames} frames)")


def _merge_task_mappings(
    datasets_meta: Sequence[lerobot_dataset.LeRobotDatasetMetadata],
) -> tuple[dict[int, str], list[int]]:
    """Merge task mappings from multiple datasets with offset indices.

    Task indices are offset to prevent collisions between datasets.

    Args:
        datasets_meta: Metadata for all datasets.

    Returns:
        Tuple of (unified task mapping with offset indices, list of offsets for each dataset).
    """
    merged_tasks = {}
    offsets = []
    offset = 0

    for i, meta in enumerate(datasets_meta):
        offsets.append(offset)

        if "task_index" not in meta.tasks:
            logging.warning(f"Dataset {i} ({meta.repo_id}) has no task_index mapping, skipping task merge for this dataset")
            continue

        dataset_tasks = {int(v): str(k) for k, v in meta.tasks["task_index"].items()}

        # Add tasks with offset
        for task_idx, task_name in dataset_tasks.items():
            merged_tasks[task_idx + offset] = task_name

        # Update offset for next dataset
        if dataset_tasks:
            offset += max(dataset_tasks.keys()) + 1

    logging.info(f"Merged task mappings: {len(merged_tasks)} total tasks across {len(datasets_meta)} datasets")
    return merged_tasks, offsets


class _OffsetTaskIndex(_transforms.DataTransformFn):
    """Offset task_index in dataset samples."""

    def __init__(self, offset: int):
        self.offset = offset

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "task_index" in data and self.offset != 0:
            data = dict(data)  # Make a copy
            data["task_index"] = int(data["task_index"]) + self.offset
        return data


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    split: Literal["train", "val"] | None = None,
    validation_split: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """Create a dataset for training or validation.

    Args:
        data_config: Data configuration.
        action_horizon: Number of action steps.
        model_config: Model configuration.
        split: Which split to use ("train" or "val"). If None, uses full dataset.
        validation_split: Fraction of episodes to use for validation (0.0 to 1.0).
        seed: Random seed for deterministic episode splitting.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Check if we have multiple datasets to combine
    if data_config.repo_ids is not None and len(data_config.repo_ids) > 1:
        # Multi-dataset case
        repo_ids = data_config.repo_ids
        logging.info(f"Creating multi-dataset from {len(repo_ids)} repos: {repo_ids}")

        # Load metadata for all datasets
        datasets_meta = [lerobot_dataset.LeRobotDatasetMetadata(rid) for rid in repo_ids]

        # Validate compatibility
        _validate_datasets_compatibility(datasets_meta, data_config.action_sequence_keys)

        # Create individual datasets
        datasets = []
        for i, (rid, meta) in enumerate(zip(repo_ids, datasets_meta)):
            ds = lerobot_dataset.LeRobotDataset(
                rid,
                delta_timestamps={
                    key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
                },
                video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND"),
            )
            datasets.append(ds)

        # Handle task mapping if needed
        if data_config.prompt_from_task:
            merged_tasks, offsets = _merge_task_mappings(datasets_meta)

            # Apply offset transform to each dataset before concatenating
            for i, (ds, offset) in enumerate(zip(datasets, offsets)):
                if offset != 0:
                    datasets[i] = TransformedDataset(ds, [_OffsetTaskIndex(offset)])

            # Concatenate all datasets
            concat_dataset = torch.utils.data.ConcatDataset(datasets)

            # Apply unified task mapping
            return TransformedDataset(concat_dataset, [_transforms.PromptFromLeRobotTask(merged_tasks)])
        else:
            # Just concatenate without task mapping or prompts
            return torch.utils.data.ConcatDataset(datasets)

    # Single dataset case (backward compatible)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND"),
    )

    # Apply episode-based train/val split if requested
    if split is not None and validation_split > 0.0:
        dataset = _split_dataset_by_episodes(dataset, split, validation_split, seed)

    if data_config.prompt_from_task:
        tasks_map = {int(v): str(k) for k, v in dataset_meta.tasks["task_index"].items()}
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks_map)])

    return dataset


def _split_dataset_by_episodes(
    dataset: lerobot_dataset.LeRobotDataset,
    split: Literal["train", "val"],
    validation_split: float,
    seed: int,
) -> Dataset:
    """Split dataset by episodes for train/val.

    Args:
        dataset: LeRobot dataset to split.
        split: Which split to return ("train" or "val").
        validation_split: Fraction of episodes for validation.
        seed: Random seed for reproducibility.

    Returns:
        Subset of the dataset containing only the requested split.
    """
    # Get episode boundaries
    episode_data_index = dataset.episode_data_index
    num_episodes = len(episode_data_index["from"])

    # Deterministically shuffle episode indices
    rng = np.random.RandomState(seed)
    episode_indices = np.arange(num_episodes)
    rng.shuffle(episode_indices)

    # Split episodes
    num_val_episodes = max(1, int(num_episodes * validation_split))
    val_episode_indices = set(episode_indices[:num_val_episodes].tolist())
    train_episode_indices = set(episode_indices[num_val_episodes:].tolist())

    logging.info(
        f"Split dataset: {num_episodes} episodes -> "
        f"{len(train_episode_indices)} train, {len(val_episode_indices)} val"
    )

    # Get frame indices for the selected episodes
    target_episodes = val_episode_indices if split == "val" else train_episode_indices
    frame_indices = []

    for episode_idx in sorted(target_episodes):
        from_idx = episode_data_index["from"][episode_idx].item()
        to_idx = episode_data_index["to"][episode_idx].item()
        frame_indices.extend(range(from_idx, to_idx))

    logging.info(f"Split '{split}' contains {len(frame_indices)} frames from {len(target_episodes)} episodes")

    # Return a Subset
    return torch.utils.data.Subset(dataset, frame_indices)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    split: Literal["train", "val"] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training or validation.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        split: Which split to use ("train" or "val"). If None, uses full dataset.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        if split is not None:
            logging.warning("Train/val split is not supported for RLDS datasets yet")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        split=split,
        validation_split=config.validation_split,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    split: Literal["train", "val"] | None = None,
    validation_split: float = 0.0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        split: Which split to use ("train" or "val"). If None, uses full dataset.
        validation_split: Fraction of episodes to use for validation.
    """
    dataset = create_torch_dataset(
        data_config, action_horizon, model_config,
        split=split, validation_split=validation_split, seed=seed
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
