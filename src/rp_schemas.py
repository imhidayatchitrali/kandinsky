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
    },
    'num_images_per_prompt': {
        'type': int,
        'required': False,
        'default': 1  # Default value for num_images_per_prompt
    }
}
