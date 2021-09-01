from .VGG_Model import VGG16_Model
from .GoogleNet_Model import GoogleNet_Model
from .Inceptionv3_Model import Inceptionv3_Model


def create_model(class_name):
    lookup = {
        'GoogleNet': GoogleNet_Model,
        'Inceptionv3': Inceptionv3_Model,
        'VGG16': VGG16_Model
    }
    if class_name not in lookup:
        print( f"Error: No model of name {class_name} exists." )
        print( "Options are:" )
        print( lookup.keys() )

    return lookup[class_name]()
