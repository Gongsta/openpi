import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), chunked_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # Compute action statistics (actions shape: [batch, action_horizon, action_dim])
    # Variance across action dimensions
    action_variance = jnp.mean(jnp.var(actions, axis=-1))

    # Velocity: first derivative (difference between consecutive timesteps)
    action_velocity = jnp.diff(actions, axis=1)  # [batch, action_horizon-1, action_dim]
    action_velocity_norm = jnp.mean(jnp.linalg.norm(action_velocity, axis=-1))

    # Acceleration: second derivative
    action_acceleration = jnp.diff(action_velocity, axis=1)  # [batch, action_horizon-2, action_dim]
    action_acceleration_norm = jnp.mean(jnp.linalg.norm(action_acceleration, axis=-1))

    # Jerk: third derivative (rate of change of acceleration)
    action_jerk = jnp.diff(action_acceleration, axis=1)  # [batch, action_horizon-3, action_dim]
    action_jerk_norm = jnp.mean(jnp.linalg.norm(action_jerk, axis=-1))

    # Compute comprehensive metrics
    info = {
        "train/loss": loss,
        "train/loss_std": jnp.std(chunked_loss),
        "train/loss_min": jnp.min(chunked_loss),
        "train/loss_max": jnp.max(chunked_loss),
        "train/grad_norm": optax.global_norm(grads),
        "train/param_norm": optax.global_norm(kernel_params),
        "train/update_norm": optax.global_norm(updates),
        "train/action_variance": action_variance,
        "train/action_velocity": action_velocity_norm,
        "train/action_acceleration": action_acceleration_norm,
        "train/action_jerk": action_jerk_norm,
    }

    # Add EMA param norm if using EMA
    if state.ema_decay is not None and state.ema_params is not None:
        ema_kernel_params = jax.tree.map(lambda x: x.value if hasattr(x, 'value') else x,
                                         jax.tree_util.tree_leaves(state.ema_params.filter(
                                             nnx.All(
                                                 nnx.Param,
                                                 nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                                                 lambda _, x: x.value.ndim > 1,
                                             )
                                         )))
        info["train/ema_param_norm"] = optax.global_norm(ema_kernel_params)

    return new_state, info


@at.typecheck
def eval_step(
    rng: at.KeyArrayLike,
    model_def: nnx.GraphDef,
    params: at.Params,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Evaluation step without gradients."""
    model = nnx.merge(model_def, params)
    model.eval()

    eval_rng = rng
    observation, actions = batch

    chunked_loss = model.compute_loss(eval_rng, observation, actions, train=False)
    loss = jnp.mean(chunked_loss)

    # Compute action statistics (actions shape: [batch, action_horizon, action_dim])
    action_variance = jnp.mean(jnp.var(actions, axis=-1))
    action_velocity = jnp.diff(actions, axis=1)
    action_velocity_norm = jnp.mean(jnp.linalg.norm(action_velocity, axis=-1))
    action_acceleration = jnp.diff(action_velocity, axis=1)
    action_acceleration_norm = jnp.mean(jnp.linalg.norm(action_acceleration, axis=-1))
    action_jerk = jnp.diff(action_acceleration, axis=1)
    action_jerk_norm = jnp.mean(jnp.linalg.norm(action_jerk, axis=-1))

    info = {
        "val/loss": loss,
        "val/loss_std": jnp.std(chunked_loss),
        "val/loss_min": jnp.min(chunked_loss),
        "val/loss_max": jnp.max(chunked_loss),
        "val/action_variance": action_variance,
        "val/action_velocity": action_velocity_norm,
        "val/action_acceleration": action_acceleration_norm,
        "val/action_jerk": action_jerk_norm,
    }
    return info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Create training data loader (with train split if validation_split > 0)
    split = "train" if config.validation_split > 0 else None
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split=split,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Create validation data loader (separate val split, no shuffling)
    if config.validation_split > 0:
        val_data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            split="val",
        )
        val_data_iter = iter(val_data_loader)
        val_batch = next(val_data_iter)
        logging.info(f"Initialized validation data loader with {config.validation_split:.1%} of episodes")
    else:
        val_data_loader = None
        val_data_iter = None
        val_batch = None
        logging.info("Validation split disabled (validation_split=0)")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    peval_step = jax.jit(
        eval_step,
        in_shardings=(replicated_sharding, replicated_sharding, replicated_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    best_val_loss = float('inf')
    start_time = time.time()
    log_start_time = start_time

    # Compute initial learning rate
    def get_learning_rate(step: int) -> float:
        """Get learning rate from schedule."""
        schedule_fn = _optimizer.create_schedule(config.lr_schedule)
        return float(schedule_fn(step))

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0 and step > 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # Add learning rate
            current_lr = get_learning_rate(step)
            reduced_info["train/learning_rate"] = current_lr

            # Add throughput metrics
            elapsed = time.time() - log_start_time
            steps_per_sec = config.log_interval / elapsed
            samples_per_sec = steps_per_sec * config.batch_size
            reduced_info["perf/steps_per_sec"] = steps_per_sec
            reduced_info["perf/samples_per_sec"] = samples_per_sec
            reduced_info["perf/time_elapsed"] = time.time() - start_time

            # Run validation every eval_interval (if validation is enabled)
            if step % config.eval_interval == 0 and val_data_loader is not None:
                val_infos = []
                for _ in range(config.num_eval_batches):
                    with sharding.set_mesh(mesh):
                        # Use EMA params if available, otherwise use regular params
                        eval_params = train_state.ema_params if train_state.ema_params is not None else train_state.params
                        val_info = peval_step(train_rng, train_state.model_def, eval_params, val_batch)
                    val_infos.append(val_info)
                    val_batch = next(val_data_iter)

                stacked_val_infos = common_utils.stack_forest(val_infos)
                reduced_val_info = jax.device_get(jax.tree.map(jnp.mean, stacked_val_infos))
                reduced_info.update(reduced_val_info)

                # Track best validation loss
                if reduced_val_info["val/loss"] < best_val_loss:
                    best_val_loss = reduced_val_info["val/loss"]
                    reduced_info["val/best_loss"] = best_val_loss
                else:
                    reduced_info["val/best_loss"] = best_val_loss

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            log_start_time = time.time()

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
