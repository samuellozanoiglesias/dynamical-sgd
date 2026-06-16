from __future__ import annotations

from dataclasses import dataclass
from math import prod, sqrt
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp


ParamTree = dict[str, Any]


def _normalize_init_type(init_type: str) -> str:
    normalized = str(init_type).strip().lower().replace("_", "")
    if normalized in {"glorot", "xavier"}:
        return "glorot"
    if normalized == "lecun":
        return "lecun"
    raise ValueError(f"Unsupported init_type '{init_type}'. Use Glorot or LeCun.")


def _linear_init(rng_key, in_dim, out_dim, init_type: str):
    init_name = _normalize_init_type(init_type)
    k_w, k_b = jax.random.split(rng_key)

    if init_name == "glorot":
        # Glorot/Xavier uniform — matches stax.Dense
        bound = sqrt(6.0 / float(in_dim + out_dim))
        kernel = jax.random.uniform(
            k_w,
            shape=(in_dim, out_dim),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        )
        # Zero bias — matches stax.Dense
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        return {"kernel": kernel, "bias": bias}

    # LeCun uniform
    bound = 1.0 / sqrt(float(max(1, in_dim)))
    kernel = jax.random.uniform(
        k_w,
        shape=(in_dim, out_dim),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    bias = jax.random.uniform(
        k_b,
        shape=(out_dim,),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    return {"kernel": kernel, "bias": bias}


def _conv_init(rng_key, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], init_type: str):
    init_name = _normalize_init_type(init_type)
    if isinstance(kernel_size, int):
        kh = kw = int(kernel_size)
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])

    k_w, k_b = jax.random.split(rng_key)
    fan_in = int(in_channels * kh * kw)
    fan_out = int(out_channels * kh * kw)

    if init_name == "glorot":
        bound = sqrt(6.0 / float(fan_in + fan_out))
        kernel = jax.random.uniform(
            k_w,
            shape=(kh, kw, in_channels, out_channels),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        )
        bias = jnp.zeros((out_channels,), dtype=jnp.float32)
        return {"kernel": kernel, "bias": bias}

    bound = 1.0 / sqrt(float(max(1, fan_in)))
    kernel = jax.random.uniform(
        k_w,
        shape=(kh, kw, in_channels, out_channels),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    bias = jax.random.uniform(
        k_b,
        shape=(out_channels,),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    return {"kernel": kernel, "bias": bias}


def _conv2d(
    x: jax.Array,
    kernel: jax.Array,
    bias: jax.Array | None,
    stride: int,
    padding: str,
) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if bias is not None:
        y = y + bias[None, None, None, :]
    return y


def _max_pool2d(x: jax.Array, window_size: int, stride: int) -> jax.Array:
    init_val = jnp.array(-jnp.inf, dtype=x.dtype)
    return jax.lax.reduce_window(
        x,
        init_val,
        jax.lax.max,
        window_dimensions=(1, int(window_size), int(window_size), 1),
        window_strides=(1, int(stride), int(stride), 1),
        padding="SAME",
    )


def _avg_pool2d(x: jax.Array, window_size: int, stride: int) -> jax.Array:
    # Use depthwise conv for average pooling to keep reverse-mode autodiff stable.
    if x.ndim != 4:
        raise ValueError("avg_pool2d expects NHWC inputs with 4 dimensions.")
    kh = int(window_size)
    kw = int(window_size)
    stride = int(stride)
    in_channels = int(x.shape[-1])
    if in_channels <= 0:
        raise ValueError("avg_pool2d expects a positive channel dimension.")
    scale = 1.0 / float(kh * kw)
    kernel = jnp.full((kh, kw, 1, in_channels), scale, dtype=x.dtype)
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=in_channels,
    )


def _global_avg_pool(x: jax.Array, *, data_format: str = "NCHW") -> jax.Array:
    if data_format.upper() == "NHWC":
        return jnp.mean(x, axis=(1, 2))
    return jnp.mean(x, axis=(2, 3))


def _resnet_block_init(
    rng_key: jax.Array,
    in_channels: int,
    out_channels: int,
    stride: int,
    init_type: str,
) -> dict[str, dict[str, jax.Array]]:
    key1, key2, key3 = jax.random.split(rng_key, 3)
    block: dict[str, dict[str, jax.Array]] = {
        "conv1": _conv_init(key1, in_channels, out_channels, 3, init_type),
        "conv2": _conv_init(key2, out_channels, out_channels, 3, init_type),
    }
    if stride != 1 or in_channels != out_channels:
        block["shortcut"] = _conv_init(key3, in_channels, out_channels, 1, init_type)
    return block


def _resnet_mlp_block_init(
    rng_key: jax.Array,
    in_dim: int,
    out_dim: int,
    init_type: str,
) -> dict[str, dict[str, jax.Array]]:
    key1, key2, key3 = jax.random.split(rng_key, 3)
    block: dict[str, dict[str, jax.Array]] = {
        "dense1": _linear_init(key1, in_dim, out_dim, init_type),
        "dense2": _linear_init(key2, out_dim, out_dim, init_type),
    }
    if in_dim != out_dim:
        block["shortcut"] = _linear_init(key3, in_dim, out_dim, init_type)
    return block


def _activation(name: str, x: jax.Array) -> jax.Array:
    if name == "identity":
        return x
    if name == "relu":
        return jax.nn.relu(x)
    if name == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unsupported activation '{name}'.")


@dataclass(frozen=True)
class JAXModel:
    architecture: str
    input_shape: tuple[int, ...]
    num_classes: int
    hidden_dim: int
    num_hidden_layers: int
    use_bias: bool
    activation_name: str
    init_type: str
    resnet_base_width: int = 64
    resnet_block_counts: tuple[int, int, int, int] = (2, 2, 2, 2)

    def init(self, rng_key: jax.Array) -> ParamTree:
        if self.architecture in {"mlp", "mlp_relu", "mlp_tanh"}:
            input_dim = int(prod(self.input_shape))
            hidden_dims = [self.hidden_dim for _ in range(self.num_hidden_layers)]
            layer_dims = [input_dim, *hidden_dims, self.num_classes]

            keys = jax.random.split(rng_key, num=max(1, len(layer_dims) - 1))
            hidden_layers: list[dict[str, jax.Array]] = []
            for layer_idx in range(len(layer_dims) - 2):
                hidden_layers.append(
                    _linear_init(
                        keys[layer_idx],
                        layer_dims[layer_idx],
                        layer_dims[layer_idx + 1],
                        self.init_type,
                    )
                )
            classifier = _linear_init(
                keys[len(layer_dims) - 2],
                layer_dims[-2],
                layer_dims[-1],
                self.init_type,
            )

            return {
                "hidden_layers": hidden_layers,
                "classifier": classifier,
            }

        if self.architecture == "resnet18":
            base_width = int(self.resnet_base_width)
            stage_channels = [base_width, base_width * 2, base_width * 4, base_width * 8]
            stages = []
            key = rng_key

            if len(self.input_shape) == 3:
                in_channels = int(self.input_shape[0])
                key, stem_key = jax.random.split(key)
                stem = {"conv": _conv_init(stem_key, in_channels, base_width, 7, self.init_type)}

                prev_channels = base_width
                for stage_idx, num_blocks in enumerate(self.resnet_block_counts):
                    out_channels = stage_channels[stage_idx]
                    stage_blocks: list[dict[str, dict[str, jax.Array]]] = []
                    for block_idx in range(int(num_blocks)):
                        stride = 1 if stage_idx == 0 or block_idx > 0 else 2
                        key, block_key = jax.random.split(key)
                        block = _resnet_block_init(
                            block_key,
                            prev_channels,
                            out_channels,
                            stride,
                            self.init_type,
                        )
                        stage_blocks.append(block)
                        prev_channels = out_channels
                    stages.append(stage_blocks)

                key, classifier_key = jax.random.split(key)
                classifier = _linear_init(
                    classifier_key,
                    stage_channels[-1],
                    self.num_classes,
                    self.init_type,
                )

                return {
                    "hidden_layers": {
                        "stem": stem,
                        "stages": stages,
                    },
                    "classifier": classifier,
                }

            input_dim = int(prod(self.input_shape))
            key, stem_key = jax.random.split(key)
            stem = {"linear": _linear_init(stem_key, input_dim, base_width, self.init_type)}

            prev_dim = base_width
            for stage_idx, num_blocks in enumerate(self.resnet_block_counts):
                out_dim = stage_channels[stage_idx]
                stage_blocks = []
                for _ in range(int(num_blocks)):
                    key, block_key = jax.random.split(key)
                    block = _resnet_mlp_block_init(
                        block_key,
                        prev_dim,
                        out_dim,
                        self.init_type,
                    )
                    stage_blocks.append(block)
                    prev_dim = out_dim
                stages.append(stage_blocks)

            key, classifier_key = jax.random.split(key)
            classifier = _linear_init(
                classifier_key,
                prev_dim,
                self.num_classes,
                self.init_type,
            )

            return {
                "hidden_layers": {
                    "stem": stem,
                    "stages": stages,
                },
                "classifier": classifier,
            }

        raise ValueError(
            f"Unsupported model architecture '{self.architecture}'. "
            "JAX runner currently supports: mlp, mlp_relu, mlp_tanh, resnet18."
        )

    def apply(
        self,
        params: ParamTree,
        x: jax.Array,
        *,
        return_intermediates: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        if self.architecture in {"mlp", "mlp_relu", "mlp_tanh"}:
            x_arr = jnp.asarray(x, dtype=jnp.float32)
            x_flat = jnp.reshape(x_arr, (x_arr.shape[0], -1))

            h = x_flat
            first_linear_pre: jax.Array | None = None
            first_activation_post: jax.Array | None = None

            for layer_idx, layer in enumerate(params["hidden_layers"]):
                z = h @ layer["kernel"]
                if self.use_bias:
                    z = z + layer["bias"]
                if layer_idx == 0:
                    first_linear_pre = z
                h = _activation(self.activation_name, z)
                if layer_idx == 0:
                    first_activation_post = h

            pre_classifier = h
            logits = pre_classifier @ params["classifier"]["kernel"]
            if self.use_bias:
                logits = logits + params["classifier"]["bias"]

            if not return_intermediates:
                return logits

            intermediates: dict[str, jax.Array] = {
                "input": x_flat,
                "pre_classifier": pre_classifier,
            }
            if first_linear_pre is not None:
                intermediates["first_linear_pre"] = first_linear_pre
            if first_activation_post is not None:
                intermediates["first_activation_post"] = first_activation_post
            return logits, intermediates

        if self.architecture == "resnet18":
            x_arr = jnp.asarray(x, dtype=jnp.float32)
            first_linear_pre: jax.Array | None = None
            first_activation_post: jax.Array | None = None

            hidden_layers = params["hidden_layers"]
            stages = hidden_layers.get("stages", [])

            if len(self.input_shape) == 3:
                if x_arr.ndim != 4:
                    raise ValueError(
                        "resnet18 expects inputs with shape (batch, channels, height, width)."
                    )
                x_arr = jnp.transpose(x_arr, (0, 2, 3, 1))
                stem_conv = hidden_layers["stem"]["conv"]
                z = _conv2d(
                    x_arr,
                    stem_conv["kernel"],
                    stem_conv["bias"] if self.use_bias else None,
                    stride=2,
                    padding="SAME",
                )
                first_linear_pre = z
                h = jax.nn.relu(z)
                first_activation_post = h
                h = _avg_pool2d(h, window_size=3, stride=2)

                for stage_idx, stage_blocks in enumerate(stages):
                    for block_idx, block in enumerate(stage_blocks):
                        stride = 1 if stage_idx == 0 or block_idx > 0 else 2
                        y = _conv2d(
                            h,
                            block["conv1"]["kernel"],
                            block["conv1"]["bias"] if self.use_bias else None,
                            stride=stride,
                            padding="SAME",
                        )
                        y = jax.nn.relu(y)
                        y = _conv2d(
                            y,
                            block["conv2"]["kernel"],
                            block["conv2"]["bias"] if self.use_bias else None,
                            stride=1,
                            padding="SAME",
                        )
                        shortcut = h
                        shortcut_params = block.get("shortcut")
                        if shortcut_params is not None:
                            shortcut = _conv2d(
                                h,
                                shortcut_params["kernel"],
                                shortcut_params["bias"] if self.use_bias else None,
                                stride=stride,
                                padding="SAME",
                            )
                        h = jax.nn.relu(y + shortcut)

                    pre_classifier = _global_avg_pool(h, data_format="NHWC")
            else:
                x_flat = jnp.reshape(x_arr, (x_arr.shape[0], -1))
                stem_linear = hidden_layers["stem"]["linear"]
                z = x_flat @ stem_linear["kernel"]
                if self.use_bias:
                    z = z + stem_linear["bias"]
                first_linear_pre = z
                h = jax.nn.relu(z)
                first_activation_post = h

                for stage_blocks in stages:
                    for block in stage_blocks:
                        y = h @ block["dense1"]["kernel"]
                        if self.use_bias:
                            y = y + block["dense1"]["bias"]
                        y = jax.nn.relu(y)
                        y = y @ block["dense2"]["kernel"]
                        if self.use_bias:
                            y = y + block["dense2"]["bias"]
                        shortcut = h
                        shortcut_params = block.get("shortcut")
                        if shortcut_params is not None:
                            shortcut = shortcut @ shortcut_params["kernel"]
                            if self.use_bias:
                                shortcut = shortcut + shortcut_params["bias"]
                        h = jax.nn.relu(y + shortcut)

                pre_classifier = h

            logits = pre_classifier @ params["classifier"]["kernel"]
            if self.use_bias:
                logits = logits + params["classifier"]["bias"]

            if not return_intermediates:
                return logits

            intermediates = {
                "input": x_arr if len(self.input_shape) == 3 else x_flat,
                "pre_classifier": pre_classifier,
            }
            if first_linear_pre is not None:
                intermediates["first_linear_pre"] = first_linear_pre
            if first_activation_post is not None:
                intermediates["first_activation_post"] = first_activation_post
            return logits, intermediates

        raise ValueError(f"Unsupported model architecture '{self.architecture}'.")

    def iter_named_parameters(self, params: ParamTree) -> Iterator[tuple[str, jax.Array]]:
        if self.architecture in {"mlp", "mlp_relu", "mlp_tanh"}:
            for layer_idx, layer in enumerate(params["hidden_layers"]):
                yield (f"feature_extractor.{layer_idx}.weight", layer["kernel"])
                if self.use_bias:
                    yield (f"feature_extractor.{layer_idx}.bias", layer["bias"])
        elif self.architecture == "resnet18":
            hidden_layers = params["hidden_layers"]
            stem = hidden_layers.get("stem", {})
            stages = hidden_layers.get("stages", [])
            if len(self.input_shape) == 3:
                stem_conv = stem.get("conv")
                if stem_conv is not None:
                    yield ("feature_extractor.stem.conv.weight", stem_conv["kernel"])
                    if self.use_bias:
                        yield ("feature_extractor.stem.conv.bias", stem_conv["bias"])

                for stage_idx, stage_blocks in enumerate(stages):
                    for block_idx, block in enumerate(stage_blocks):
                        conv1 = block["conv1"]
                        conv2 = block["conv2"]
                        yield (
                            f"feature_extractor.stage{stage_idx}.block{block_idx}.conv1.weight",
                            conv1["kernel"],
                        )
                        if self.use_bias:
                            yield (
                                f"feature_extractor.stage{stage_idx}.block{block_idx}.conv1.bias",
                                conv1["bias"],
                            )
                        yield (
                            f"feature_extractor.stage{stage_idx}.block{block_idx}.conv2.weight",
                            conv2["kernel"],
                        )
                        if self.use_bias:
                            yield (
                                f"feature_extractor.stage{stage_idx}.block{block_idx}.conv2.bias",
                                conv2["bias"],
                            )
                        shortcut = block.get("shortcut")
                        if shortcut is not None:
                            yield (
                                f"feature_extractor.stage{stage_idx}.block{block_idx}.shortcut.weight",
                                shortcut["kernel"],
                            )
                            if self.use_bias:
                                yield (
                                    f"feature_extractor.stage{stage_idx}.block{block_idx}.shortcut.bias",
                                    shortcut["bias"],
                                )
            else:
                stem_linear = stem.get("linear")
                if stem_linear is not None:
                    yield ("feature_extractor.stem.linear.weight", stem_linear["kernel"])
                    if self.use_bias:
                        yield ("feature_extractor.stem.linear.bias", stem_linear["bias"])

                for stage_idx, stage_blocks in enumerate(stages):
                    for block_idx, block in enumerate(stage_blocks):
                        dense1 = block["dense1"]
                        dense2 = block["dense2"]
                        yield (
                            f"feature_extractor.stage{stage_idx}.block{block_idx}.dense1.weight",
                            dense1["kernel"],
                        )
                        if self.use_bias:
                            yield (
                                f"feature_extractor.stage{stage_idx}.block{block_idx}.dense1.bias",
                                dense1["bias"],
                            )
                        yield (
                            f"feature_extractor.stage{stage_idx}.block{block_idx}.dense2.weight",
                            dense2["kernel"],
                        )
                        if self.use_bias:
                            yield (
                                f"feature_extractor.stage{stage_idx}.block{block_idx}.dense2.bias",
                                dense2["bias"],
                            )
                        shortcut = block.get("shortcut")
                        if shortcut is not None:
                            yield (
                                f"feature_extractor.stage{stage_idx}.block{block_idx}.shortcut.weight",
                                shortcut["kernel"],
                            )
                            if self.use_bias:
                                yield (
                                    f"feature_extractor.stage{stage_idx}.block{block_idx}.shortcut.bias",
                                    shortcut["bias"],
                                )
        else:
            raise ValueError(f"Unsupported model architecture '{self.architecture}'.")

        yield ("fc.weight", params["classifier"]["kernel"])
        if self.use_bias:
            yield ("fc.bias", params["classifier"]["bias"])

    def classifier_weight_matrix(self, params: ParamTree) -> jax.Array:
        # Store kernels as [in_dim, out_dim], but NC geometry expects rows by class.
        return jnp.transpose(params["classifier"]["kernel"], (1, 0))

    def classifier_bias_vector(self, params: ParamTree) -> jax.Array | None:
        if not self.use_bias:
            return None
        return params["classifier"]["bias"]


@dataclass(frozen=True)
class BuiltModel:
    model: JAXModel
    params: ParamTree


def build_model(
    model_cfg: dict,
    input_shape: Sequence[int],
    num_classes: int,
    random_seed: int,
) -> BuiltModel:
    architecture = str(model_cfg.get("architecture", "mlp")).strip().lower()
    use_bias = bool(model_cfg.get("use_bias", True))
    init_type = _normalize_init_type(model_cfg.get("init_type", "glorot"))
    resnet_base_width = int(model_cfg.get("resnet_base_width", 64))

    if architecture == "mlp":
        activation_name = "identity"
    elif architecture == "mlp_relu":
        activation_name = "relu"
    elif architecture == "mlp_tanh":
        activation_name = "tanh"
    elif architecture == "resnet18":
        activation_name = "relu"
    else:
        raise ValueError(f"Unsupported model architecture '{architecture}'.")

    model = JAXModel(
        architecture=architecture,
        input_shape=tuple(int(v) for v in input_shape),
        num_classes=int(num_classes),
        hidden_dim=int(model_cfg.get("nn_width", 10)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
        use_bias=use_bias,
        activation_name=activation_name,
        init_type=init_type,
        resnet_base_width=resnet_base_width,
    )

    params = model.init(jax.random.PRNGKey(int(random_seed)))
    return BuiltModel(model=model, params=params)