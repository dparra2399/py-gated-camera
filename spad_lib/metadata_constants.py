metadata_keys = [
    'Author', 'System', 'Date taken', 'Time taken', 'Mode',
    'Integration time', 'Laser frequency', 'Overlap', 'Frame', 
    'Frames', 'Gate step', 'Gate steps', 'Gate step arbitrary', 
    'Gate step size', 'Gate width', 'Gate offset', 'Gate increment', 
    'External frame trigger', 'External gate trigger', 'Software version'
]

def get_dict_metadata(im):
    metadata_dict = {}
    for key in metadata_keys:
        value = im.info.get(key)  # Use .get() to safely access the key
        if value == None:  # Only print if the key exists
            continue
        if key in ['Frame', 'Frames', 'Gate steps', 'Gate step']:
            value = int(value)
        elif key in ['Laser frequency']:
            if 'MHz' in value:
                value = float(value.split()[0]) * 1e6
                metadata_dict['Laser time'] = 1/value * 1e12
            else:
                value = float(value.split()[0])
                metadata_dict['Laser time'] = 1/value * 1e12
        elif key in ['Gate offset', 'Gate step size']:
            value = float(value.split()[0])
        elif key in ['Gate width']:
            value = float(value.split()[0]) * 1e3
        metadata_dict[key] = value
    return metadata_dict

