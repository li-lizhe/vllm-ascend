# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Monkey-patch vLLM's process_weights_after_loading to add OTP wo_b row-slicing."""

import os
import torch
import torch.nn as nn

import vllm.model_executor.model_loader.utils as loader_utils

_orig_process_weights_after_loading = loader_utils.process_weights_after_loading


def _ascend_process_weights_after_loading(
    model: nn.Module,
    model_config,
    target_device: torch.device,
) -> None:
    pid = os.getpid()
    print(f"[OTP_PATCH] _ascend_process_weights_after_loading ENTERED, pid={pid}", flush=True)

    # First, run the original vLLM logic
    _orig_process_weights_after_loading(model, model_config, target_device)

    print(f"[OTP_PATCH] original function DONE, pid={pid}", flush=True)

    # Then, OTP wo_b row-slicing
    try:
        from vllm_ascend.utils import oproj_tp_enable
        otp_enabled = oproj_tp_enable()
        print(f"[OTP_PATCH] oproj_tp_enable={otp_enabled}, pid={pid}", flush=True)
        if otp_enabled:
            from vllm_ascend.distributed.parallel_state import get_otp_group
            from vllm.model_executor.layers.linear import RowParallelLinear

            otp_group = get_otp_group()
            otp_rank = otp_group.rank_in_group
            otp_size = otp_group.world_size
            print(f"[OTP_PATCH] otp_rank={otp_rank}, otp_size={otp_size}, pid={pid}", flush=True)

            wo_b_count = 0
            for name, module in model.named_modules():
                if isinstance(module, RowParallelLinear) and name.endswith('.wo_b'):
                    wo_b_count += 1
                    old_shape = module.weight.data.shape
                    print(f"[OTP_PATCH] wo_b '{name}' BEFORE shape={old_shape}, pid={pid}", flush=True)
                    wo_b_weight = module.weight.data
                    rows_per_shard = wo_b_weight.shape[0] // otp_size
                    wo_b_shard = wo_b_weight[
                        otp_rank * rows_per_shard:(otp_rank + 1) * rows_per_shard, :
                    ].clone()
                    del wo_b_weight
                    module.weight = nn.Parameter(wo_b_shard, requires_grad=False)
                    print(f"[OTP_PATCH] wo_b '{name}' AFTER shape={module.weight.data.shape}, pid={pid}", flush=True)

                    if hasattr(module, 'weight_scale'):
                        ws = module.weight_scale.data
                        ws_shard = ws[
                            otp_rank * rows_per_shard:(otp_rank + 1) * rows_per_shard
                        ].clone()
                        del ws
                        module.weight_scale = nn.Parameter(ws_shard, requires_grad=False)
                    if hasattr(module, 'weight_offset'):
                        wo = module.weight_offset.data
                        wo_shard = wo[
                            otp_rank * rows_per_shard:(otp_rank + 1) * rows_per_shard
                        ].clone()
                        del wo
                        module.weight_offset = nn.Parameter(wo_shard, requires_grad=False)

                    torch.npu.synchronize()
            print(f"[OTP_PATCH] wo_b_count={wo_b_count}, pid={pid}", flush=True)
    except ImportError as e:
        print(f"[OTP_PATCH] ImportError: {e}, pid={pid}", flush=True)
    except Exception as e:
        print(f"[OTP_PATCH] Exception: {e}, pid={pid}", flush=True)


# Apply monkey-patch
loader_utils.process_weights_after_loading = _ascend_process_weights_after_loading

import vllm.model_executor.model_loader.base_loader as base_loader
base_loader.process_weights_after_loading = _ascend_process_weights_after_loading

print(f"[OTP_PATCH] Monkey-patch applied, pid={os.getpid()}, loader_utils={loader_utils.process_weights_after_loading.__name__}, base_loader={base_loader.process_weights_after_loading.__name__}", flush=True)

# Monkey-patch LogitsProcessor._gather_logits to skip all_gather in OTP mode
import vllm.model_executor.layers.logits_processor as lp_module
_orig_gather_logits = lp_module.LogitsProcessor._gather_logits

def _otp_safe_gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
    from vllm_ascend.utils import oproj_tp_enable
    if oproj_tp_enable():
        # Skip all_gather in OTP mode - each rank already has full logits
        return logits
    return _orig_gather_logits(self, logits)

lp_module.LogitsProcessor._gather_logits = _otp_safe_gather_logits
print(f"[OTP_PATCH] LogitsProcessor._gather_logits patched for OTP mode, pid={os.getpid()}", flush=True)
