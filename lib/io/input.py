import configparser
import argparse
import sys
from pathlib import Path
import random


class Config():
    def __init__(self):
        pass

    def parse_config(self):
        self._original_config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._original_config.optionxform = str
        if self.input is not None:
            if (isinstance(self.input, str)):
                self.infile = self.input
            elif (isinstance(self.input, dict)):
                self.indict = self.input
        else:
            if not Path(self.args.infile).is_file():
                raise FileNotFoundError("Input file not found")
            self.infile = self.args.infile

        if (self.infile is not None):
            # We have an input file.
            self._original_config.read(self.infile)

            # continue setting up self._original_config
            infile_folder = str(Path(self.infile).parent.absolute())
            file_name = self.infile.split('/')[-1]
            if not Path(infile_folder+'/'+file_name).is_file():
                raise RuntimeError("Input file {} not found in {}", file_name, infile_folder)

            #vprint = output.screen if self.args.verbose else lambda *arguments, **kwargs: None
            # This adds keyword replacements to the config.
            if self.args.keyword_replacements:
                for kwg, kwn, kwv in self.args.keyword_replacements:
                    if kwg not in self._original_config:
                        raise ValueError(f"{kwg} is not a valid keyword group")
                    self._original_config[kwg][kwn] = kwv


    def convert_to_dict(self, original_input=False):
        """
        Convert the current config (settings) object to a dictionary. Note that datatypes may not be preserved.

        Args:
            original_input: optional, set to True to return the original input 

        Returns:
            config_dict: Python dictionary containing the same elements as the original or current config (settings) object.
        """
        if original_input:
            config_dict = {s:dict(self._original_config.items(s)) for s in self._original_config.sections()}
        else:
            config_dict = {s:vars(self.sections[s]) for s in self.sections}
            return config_dict
