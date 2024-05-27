""" module for model specific concretisation of abstract interfaces """
import os

from tools.converter import Converter, InputConverter
from tools.inference import Inferencer


class ConcreteConverter(Converter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        pass


class ConcreteInferencer(Inferencer):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        command = f"python predict.py {filepath} {output_filename}"
        print(command)
        os.system(command)


class ConcreteInputConverter(InputConverter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        pass
