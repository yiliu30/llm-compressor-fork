import contextlib
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["GPTQModifier"]


from collections import defaultdict
import os

FALLBACK_CHANGE = os.environ.get("FALLBACK_CHANGE", "0").lower() in ("1", "true", "yes")
_DEBUG = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")

all_module_input = defaultdict(list)
all_module_output = defaultdict(list)


def input_capture_hook(module, *args, **kwargs):
    all_module_input[module._tmp_name].append((args, kwargs))


def output_capture_hook(module, *args, **kwargs):
    all_module_output[module._tmp_name].append((args, kwargs))




def _is_decoding_layer(module, name):
    return "decoderlayer" in module.__class__.__name__.lower()


class _LLModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()

    def forward(self, *args, **kwargs):
        for layer in self.layers:
            res = layer(*args, **kwargs)
        return res


class _PretrainModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LLModelWrapper()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class GPTQModifier(Modifier, QuantizationMixin):
    """
    Implements the GPTQ algorithm from https://arxiv.org/abs/2210.17323. This modifier
    uses activations to calibrate a hessian matrix, which is then used to determine
    optimal quantizion values and orderings for the model weights.

    | Sample yaml:
    | test_stage:
    |    obcq_modifiers:
    |      GPTQModifier:
    |          block_size: 128
    |          dampening_frac: 0.001
    |          offload_hessians: False
    |          actorder: static
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 8
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: group
    |                    group_size: 128

    Lifecycle:
        - on_initialize
            - apply config to model
        - on_start
            - add activation calibration hooks
            - add gptq weight calibration hooks
        - on_sequential_epoch_end
            - quantize_weight
        - on_finalize
            - remove_hooks()
            - model.apply(freeze_module_quantization)

    :param sequential_targets: list of layer names to compress during GPTQ, or
        '__ALL__' to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param actorder: order in which weight columns are quantized. Defaults to "static"
        activation ordering, which achieves best accuracy recovery with no runtime cost.
        For more information, see https://github.com/vllm-project/vllm/pull/8135
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param scheme: a single quantization scheme to apply to the model. This is a
        dictionary that supports all keys from QuantizationScheme except targets, which
        will be set to the targets parameter set at the modifier level. Can also be set
        to a dictionary of the format `preset_scheme_name: targets` for example:
        `W8A8: ['Linear']` for weight and activation 8-bit.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aformentioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    """

    # gptq modifier arguments
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    # TODO: this does not serialize / will be incorrectly written
    actorder: Optional[Union[ActivationOrdering, Sentinel]] = Sentinel("static")
    offload_hessians: bool = False

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    
    _cur_layer_idx = PrivateAttr(default=0)
    

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

        def resolve_actorder(existing):
            # sentinel default only overrides if existing is None
            if self.actorder == Sentinel("static"):
                return ActivationOrdering.STATIC if existing is None else existing

            # user-provided value always attempts to override
            if existing is None or self.actorder == existing:
                return self.actorder

            # if existing provided and conflicts
            raise ValueError(
                "Cannot resolve activation ordering when both "
                "`GPTQModifier.actorder` and `QuantizationScheme.actorder` "
                f"are provided and differ ({self.actorder}, {existing}). "
                "Either unset `GPTQModifier.actorder` or "
                "remove `actorder` from config groups."
            )

        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if (
                getattr_chain(scheme, "weights.strategy", None)
                == QuantizationStrategy.GROUP
            ):
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }
        # add tmp name for each module for logging
        for name, mod in state.model.named_modules():
            mod._tmp_name = name
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)
        # breakpoint()
        # register gptq hooks
        added_hook = False
        # for name, module in match_named_modules(state.model, self.resolved_targets, self.ignore):
        #     if getattr_chain(module, "quantization_scheme.weights", None) is not None:
        #         # HACK: previously, embeddings were not quantized because they were not
        #         # accessible by the layer compressor. For now, we manually ignore it,
        #         # but in the FUTURE this should be ignored by the user
        #         if not isinstance(module, torch.nn.Embedding):
        #             # logger.warning(f">> Registering GPTQ hook for module {getattr(module, '_tmp_name', '')} || {name}")
        #             # self.register_hook(module, self.calibrate_module, "forward")
        #             added_hook = True
        # breakpoint()
        for name, module in state.model.named_modules():
            if _is_decoding_layer(module, name):
                # register input/output capture hooks for decoding layers
                logger.warning(f">> Registering input/output capture hooks for decoding layer {getattr(module, '_tmp_name', '')} || {name}")
                module.register_forward_pre_hook(input_capture_hook, with_kwargs=True)
                module.register_forward_hook(output_capture_hook, with_kwargs=True)

        # if not added_hook:
        #     raise ValueError(
        #         "GPTQModifier requires a weight quantization config be specified by "
        #         "this modifier or a modifier preceding it"
        #     )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            if FALLBACK_CHANGE:
                self.compress_modules(state=state)
            else:
                self.autoround(state)


        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if FALLBACK_CHANGE:
                self.compress_modules(state=state)
            else:
                pass
            if not self.ended_:
                self.on_end(state, None)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Calibration hook used to accumulate the hessian of the input to the module

        :param module: module being calibrated
        :param args: inputs to the module, the first element of which is the
            cannonical input
        :param _output: uncompressed module output, unused
        """
        # Assume that first argument is the input
        # breakpoint()
        logger.info(f">>||>> Strating GPTQ calibration for module {getattr(module, '_tmp_name', '')}")
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def autoround(self, state):
        cur_layer_idx = self._cur_layer_idx
        self._cur_layer_idx += 1
        logger.info(f">>||>> AutoRound for decoding layer index {cur_layer_idx}")
        if cur_layer_idx >= len(state.model.model.layers):
            logger.info(f">>||>> All decoding layers have been processed for AutoRound.")
            self.compress_modules(return_directly=False)
            return
        decoding_layer = state.model.model.layers[cur_layer_idx]
        new_model = _PretrainModelWrapper()
        new_model.model.layers.append(decoding_layer)
        logger.warning(f">>||>> Strating AutoRound for decoding layer {getattr(decoding_layer, '_tmp_name', '')}")
        param = next(decoding_layer.parameters())
        new_model.dtype = param.dtype
        with align_module_device(decoding_layer):
            if _DEBUG:
                iters = 4
            else:
                iters = 200
            import auto_round
            rounder = auto_round.AutoRound(
                model=new_model,
                tokenizer="",
                scheme="W4A16",
                # iters = 128,
                iters = iters,
                enable_quanted_input=False,
                # FIXME: batch size 1 causes error, looks like related to the input_others prepare
                # batch_size=1 
            )
            # block: torch.nn.Module,
            # input_ids: list[torch.Tensor],
            # input_others: dict,
            # q_input: Union[None, torch.Tensor] = None,
            # device: Union[str, torch.device] = "cpu",


            from auto_round.compressors.utils import set_layer_config
            def _preprocess(self):
                self.layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
                    self.model,
                    self.layer_config,
                    self.scheme,
                    self.scale_dtype,
                    self.supported_types,
                    self.inner_supported_types,
                    self.quant_block_list,
                    self.fp_layers,
                    self.quant_lm_head,
                    # enable_gguf_official_mixed=enable_gguf_official_mixed,
                    is_mllm=self.mllm,
                )
                self.batch_dim = 0
            _preprocess(rounder)

            input_name = f"model.layers.{cur_layer_idx}"
            cur_inputs = all_module_input[input_name]
            input_ids = []
            input_others = {}
            positional_inputs = []
            attention_mask = None
            position_ids = None
            position_embeddings = (None, None)
            for cur_inp in cur_inputs:
                _input_ids = cur_inp[0][0][0]
                input_ids.append(cur_inp[0][0][0])
                for key, val in cur_inp[0][1].items():
                    if key == "position_ids":
                        position_ids = val
                    elif key == "position_embeddings":
                        position_embeddings = val
            input_others["position_ids"] = position_ids
            input_others["positional_inputs"] = positional_inputs
            input_others["attention_mask"] = attention_mask
            input_others["position_embeddings"] = position_embeddings
            decoding_layer.tuning_device = torch.device("cuda")
            with torch.enable_grad():
                rounder._quantize_block(
                    block=decoding_layer,
                    input_ids=input_ids,
                    input_others=input_others,
                    q_input=None,
                    device="cuda",
                )
            logger.info(f"AutoRound completed for decoding layer {getattr(decoding_layer, '_tmp_name', '')}")
            # decoding_layer.mlp.up_proj.weight_scale.shape
            # decoding_layer.mlp.up_proj.weight_zero_point.shape
            logger.info(f"Weight scale shape: {decoding_layer.mlp.up_proj.weight_scale.shape}, zero point shape: {decoding_layer.mlp.up_proj.weight_zero_point.shape}")
            logger.info(f"Weight scale shape: {decoding_layer.mlp.down_proj.weight_scale.shape}, zero point shape: {decoding_layer.mlp.down_proj.weight_zero_point.shape}")
            # decoding_layer.to("cuda")
            print(f" decoding_layer.mlp.down_proj{ decoding_layer.mlp.down_proj}")

            for name, module in decoding_layer.named_modules():
                if hasattr(module, "weight_scale") and hasattr(module, "weight_zero_point"):
                    logger.info(f"Updating offload parameters for module {getattr(module, '_tmp_name', '')} || {name}")
                    # weight = module.weight
                    scale = module.scale
                    del module.scale
                    # zero_point = module.weight_zero_point
                    # offload_device = torch.device("cpu")
                    # module.weight.data.copy_(weight.data)
                    # module.weight_scale.data.copy_(scale.data)
                    # module.weight_zero_point.data.copy_(zero_point.data)
                    # update_offload_parameter(module, "weight", weight)
                    # FIXME: yi quantize the weight based on the scale and zero_point from AutoRound
                    update_offload_parameter(module, "weight_scale", scale)
                    # update_offload_parameter(module, "weight_zero_point", zero_point)
            # state.model.model.layers[0].mlp.gate_proj.weightbt
            for module in list(self._num_samples.keys()):
                name = self._module_names[module]
                del self._num_samples[module]
            decoding_layer.eval()
            all_module_input.clear()
            all_module_output.clear()
        # logger.info(f"state.model.model.layers[0].mlp.gate_proj.weight: {state.model.model.layers[0].mlp.gate_proj.weight.device}")
    def compress_modules(self, return_directly=True, state=None):
        """
        Quantize modules which have been calibrated
        """
        if not FALLBACK_CHANGE:
            if return_directly:
                return 
        # logger.info(f">> Quantizing {name} using {num_samples} samples")
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            with torch.no_grad(), align_module_device(
                module
            ), self._maybe_onload_hessian(module), CompressionLogger(
                module
            ) as comp_logger:
                loss, quantized_weight, scale, zero_point, g_idx = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessians_dict=self._hessians,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                )
                breakpoint()
                comp_logger.set_loss(loss)

            update_offload_parameter(module, "weight", quantized_weight)
            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)
            if g_idx is not None:
                update_offload_parameter(module, "weight_g_idx", g_idx)

            # self._hessians[module] already deleted by quantize_weight
            del self._num_samples[module]

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()

        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")
