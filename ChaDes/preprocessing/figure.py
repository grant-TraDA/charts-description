import pandas as pd 
import numpy as np 
import os
import shutil
import json
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

from PIL import Image


PLOT_CATEGORIES = ["vbar_categorical", "hbar_categorical",  "pie", "line", "dot_line"]


class DatasetDivider():
    def __init__(
        self,
        path_to: str,
        model_dataset_path: str
    ):
        """Class initization.

        Arguments:
            path_to {str} -- path where files are extracted into directories named by category name
            model_dataset_path {str} -- path with plots 
        """
        self.path_to = path_to
        self.model_dataset_path = model_dataset_path    

    def divide_files(
        self, 
        categories: list, 
        annotation_path: str
    ):
        """Divide files into directories based on chart type.

        Arguments:
            categories {list} -- list of chart types
            annotation_path {str} -- path to file with annotations
        """
        # create folders
        for cat in categories:
            directory = str(self.path_to + cat)
            if not os.path.exists(directory):
                os.mkdir(directory)        

        with open(annotation_path) as json_file:
            chart_types = json.load(json_file)

        for chart_type in chart_types:
            shutil.copy(self.model_dataset_path + str(chart_type['image_index']) + ".png", 
                        self.path_to + chart_type['type'])
        print("Sucess! Files have been extracted.")


class RevisionDownloader():
    def __init__(
        self,
        config_path: str, 
        output_path: str
    ):
        """Initialize class which download Revision dataset.

        Arguments:
            config_path {str} -- path to file with url of images
            output_path {str} -- path where images will be saved
        """

        self.config_path = config_path
        self.output_path = output_path

    def download_dataset(self):
        """Save images in chosen directory.
        """
        df = pd.read_csv(self.config_path, sep="	", header=None)
        df.columns = ["type", "path"]
        categories = list(df.type.unique())
         # create folders
        for cat in categories:
            directory = str(self.output_path + cat)
            if not os.path.exists(directory):
                os.mkdir(directory) 

        # save files
        for path, image_type in zip(df.path, df.type):
            try:
                path_to = self.output_path + str(image_type) + "/" + path.split('/')[-1]
                urllib.request.urlretrieve(str(path), path_to)
            except:
                print("Failed to download " + str(path))
