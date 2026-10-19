"""Narrow import bridge for pinned ImageReward on Transformers 4.57.6.

These are exact upstream function aliases, not reimplemented scoring operations.
No files inside either third-party package are modified.
"""


def imagereward_import_bridge():
    import transformers.modeling_utils as modeling
    import transformers.pytorch_utils as pytorch
    names = ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer")
    installed = []
    for name in names:
        if not hasattr(modeling, name):
            setattr(modeling, name, getattr(pytorch, name))
            installed.append(name)
    return {"bridge": "imagereward-transformers-4.57.6-imports-v1", "installed_aliases": installed}
