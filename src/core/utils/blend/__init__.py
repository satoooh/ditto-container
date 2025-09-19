"""Helper to expose the Cython-powered blend function without eager imports."""

__all__ = ["blend_images_cy"]


def blend_images_cy(*args, **kwargs):
    """Install pyximport on demand and forward to the compiled helper."""
    import pyximport

    pyximport.install()
    from .blend import blend_images_cy as _blend_images_cy

    return _blend_images_cy(*args, **kwargs)
