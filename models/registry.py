from timm.models.registry import register_model
from .fasternetfire import FasterNetFire


@register_model
def fasternetfire(**kwargs):
    model = FasterNetFire(**kwargs)
    return model
