import os


def get_my_directory_name(file: str = __file__):
    """
    @param: you must send __file__
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(file)))


def get_my_file_name(file: str = __file__, includeExt: bool = False):
    """
    @param: you must send __file__
    @param: you wanna get extension name or not 
    """
    filename = os.path.basename(file)
    return filename if includeExt else filename.split('.')[0]


def generate_filename_to_parent_directory(file: str = __file__, ext: str = None):
    """
    Parameters
      file - you must send __file__ 

      ext - you must 
    """
    print(file)
    file_name = get_my_file_name(
        file,
        includeExt=True if ext == None else False)
    print(file_name)
    target_dir = get_my_directory_name(file)
    target = target_dir + '/' + file_name + ('' if ext == None else ext)
    return target
