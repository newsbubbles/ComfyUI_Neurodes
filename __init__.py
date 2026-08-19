"""neurodes — neural network architecture as a ComfyUI node graph.

ComfyUI loads this file. It deliberately does almost nothing: the whole pack lives in the
``neurodes`` package next to it, so it can be imported and tested without ComfyUI present.

Note that ComfyUI checks for ``NODE_CLASS_MAPPINGS`` *before* ``comfy_entrypoint`` and takes
the first one it finds, so this module must not define both.
"""

from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./web"


class NeurodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .neurodes.nodes import ALL_NODES
        return ALL_NODES


async def comfy_entrypoint() -> ComfyExtension:
    return NeurodesExtension()
