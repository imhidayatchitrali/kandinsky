'''
RunPod | Kandinsky | Schemas
'''

INPUT_SCHEMA = {
    
    'prompt': {
        'type': str,
        'required': True,
    },
    'init_image': {
        'type': str,
        'required': False,
        'default': ""
    }
}