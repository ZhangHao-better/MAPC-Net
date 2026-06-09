"""
自定义Hook: 在加载checkpoint后强制设置use_contrastive=True
解决load_from加载旧checkpoint时meta配置覆盖新配置的问题
"""
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class ForceContrastiveHook(Hook):
    """强制启用对比学习的Hook"""
    
    def __init__(self, enable_contrastive=True):
        self.enable_contrastive = enable_contrastive
    
    def after_load_checkpoint(self, runner, checkpoint):
        """在加载checkpoint后立即执行"""
        if hasattr(runner.model, 'module'):
            model = runner.model.module  # DDP
        else:
            model = runner.model
        
        # 强制设置decode_head的use_contrastive
        if hasattr(model, 'decode_head'):
            if hasattr(model.decode_head, 'use_contrastive'):
                old_value = model.decode_head.use_contrastive
                model.decode_head.use_contrastive = self.enable_contrastive
                runner.logger.info(
                    f"ForceContrastiveHook: "
                    f"use_contrastive changed from {old_value} to {self.enable_contrastive}"
                )
            else:
                runner.logger.warning(
                    "ForceContrastiveHook: decode_head has no use_contrastive attribute"
                )
        else:
            runner.logger.warning(
                "ForceContrastiveHook: model has no decode_head"
            )
