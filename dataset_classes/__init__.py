from .SubdirImg import SubdirImg_dataset
from .TextFileImg import TextFileImg_dataset
from Params import Params


class_lookup = {
    'subdir': SubdirImg_dataset,
    'textfile': TextFileImg_dataset
}


def create_dataset(phase):
    if Params.DatasetType.casefold() not in class_lookup:
        print( f"Error: No dataset type of name {Params.DatasetType} exists." )
        print( "Options are:" )
        print( class_lookup.keys() )

    return class_lookup[Params.DatasetType.casefold()](phase)
