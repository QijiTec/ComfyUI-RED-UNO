from .uno_nodes.comfy_nodes import REDUNOModelLoader, REDUNOGenerate


# 注册节点
NODE_CLASS_MAPPINGS = {
    "REDUNOModelLoader": REDUNOModelLoader,
    "REDUNOGenerate": REDUNOGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "REDUNOModelLoader": "UNO Model Loader @REDAIGC",
    "REDUNOGenerate": "UNO Generate @REDAIGC",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']