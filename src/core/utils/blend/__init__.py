import importlib
import pyximport

pyximport.install()

blend_module = importlib.import_module("core.utils.blend.blend")
blend_images_cy = blend_module.blend_images_cy

__all__ = ["blend_images_cy"]
