import json 
import os
import pytesseract
import random
import cv2 
import numpy as np
import re
import pandas as pd
import torch
import pickle
import tqdm

from fvcore.common.file_io import PathManager
from PIL import Image


class VectorCreator:
    """
    Mapping for categories
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12}
    """
    def __init__(self, prediction_path, images_path, classification_prediction_path):
        with open(prediction_path, "rb") as f:
            self.prediction = json.load(f)
        self.classification_prediction_path = classification_prediction_path
        self.mapp_types = {
            "dot_line":"dot line plot",
            "hbar_categorical": "horizontal bar plot",
            "line": "line plot" ,
            "vbar_categorical": "vertical bar plot"
        }
        self.map_categories = {
            1: 'bars',
            2: 'dot lines',
            3: 'legend',
            4: 'line', 
            5: 'title',
            6: 'x label', 
            7: 'x tick label',
            8: 'y label', 
            9: 'y tick label', 
            10: 'x axis',
            11: 'y axis', 
            13: 'legend element',
            14: 'legend element desccription'
        }
        self.images_path = images_path

    def create_vectors(self, save=True, save_path="./"):
        # load chart types
        predicted_types = self.read_classification_prediction()
        # create template list
        vector_pred = [[('image_id', id, 1), ("chart type", value, 2)] for id, value in enumerate(predicted_types)]
        print("Template created!")
        pred_categories = [[] for i in range(len(vector_pred))]
        # add textual vectors
        for pred in tqdm.tqdm(self.prediction):
            #if pred['image_id'] < 75000:
            pred_categories[pred['image_id']].append(pred['category_id'])
            if pred['category_id'] in [5, 6, 7, 8, 9, 14]:
                l = len(vector_pred[pred["image_id"]]) + 1
                text_ocr = self.extract_textual_data(pred, l)
                if text_ocr[1] != "":
                    vector_pred[pred["image_id"]].append(text_ocr)
            # add legend info
            if pred["category_id"] == 3:
                max_id = len(vector_pred[pred["image_id"]])
                vect = (
                    "legend exists",
                    "yes",
                    max_id + 1
                )
                vector_pred[pred["image_id"]].append(vect)
                # add info if legend is horizontal/vertical
                img_path = self.images_path + "/" + str(pred["image_id"]) + ".png"
                im = Image.open(img_path)
                box = pred['bbox']
                # calculate coordinates
                x1, x2 = box[0], box[0] + box[2]
                y1, y2 = box[1], box[1] + box[3]
                im  = im.crop((x1, y1, x2, y2))
                # extract_text
                text_ocr = pytesseract.image_to_string(im, config='')
                vect = (
                    "legend position",
                    "vertical" if "\n" in text_ocr else "horizontal",
                    max_id + 2
                )
                vector_pred[pred["image_id"]].append(vect)
        # add number of chosen detected objects
        vector_pred = self.add_n_category(pred_categories, vector_pred)
        # add max value of x and y tick (if it is possible)
        vector_pred = self.add_max_tick_value(vector_pred)
        if save:
            self.save(save_path, vector_pred)
            print("Saved!")
        return vector_pred

    def add_max_tick_value(self, vector_pred):
        for pred in vector_pred:
            x_tick, y_tick = [], []
            for vect in pred:
                if vect[0] == self.map_categories[7]:
                    value = re.findall(r'[-+]?\d*\.\d+|\d+', vect[1])
                    if value != []:
                        x_tick.append(value)
                if vect[0] == self.map_categories[9]:
                    value = re.findall(r'[-+]?\d*\.\d+|\d+', vect[1])
                    if value != []:
                        y_tick.append(value)
            if len(x_tick) > 1:
                x_tick = [float(x[0]) for x in x_tick if x is not None]
            if len(y_tick) > 1:
                y_tick = [float(x[0]) for x in y_tick if x is not None]
            if len(y_tick) > 0:
                max_id = len(pred)
                vect = (
                    "max value of y",
                    max(y_tick),
                    max_id + 1
                )
                pred.append(vect)
            if len(x_tick) > 0:
                max_id = len(pred)
                vect = (
                    "max value of x",
                    max(x_tick),
                    max_id + 1
                )
                pred.append(vect)
        return vector_pred

    def add_n_category(self, pred_categories, vector_pred):
        """Add number of elements of dots, bars, element in legend."""
        for id, annot in enumerate(pred_categories):
            for cat in [1, 2, 7, 9, 13, 14]:
                n_elem = annot.count(cat)
                max_id = len(vector_pred[id])
                vect = (
                    "number of {}".format(self.map_categories[cat]),
                    n_elem,
                    max_id + 1
                )
                vector_pred[id].append(vect)
        return vector_pred

    def save(self, save_path, vect):
        with open(save_path + "pred_vect.pkl", "wb") as f:
            pickle.dump(vect, f)

    def read_classification_prediction(self):
        probs = pd.read_csv(self.classification_prediction_path).drop("Unnamed: 0",1)
        return [self.mapp_types[type_] for type_ in probs.idxmax(axis=1)]

    def extract_textual_data(self, prediction, length):
        """Uses OCR to extract textual data

        It is used for title, xlabel, xtickabel, ylabel, ytickabel, legend_element_desc.
        """
        if prediction['category_id'] not in [5, 6, 7, 8, 9, 14]:
            raise Exception("Incorrect category.")
        img_path = self.images_path + "/" + str(prediction["image_id"]) + ".png"
        im = Image.open(img_path)
        box = prediction['bbox']
        # calculate coordinates
        x1, x2 = box[0], box[0] + box[2]
        y1, y2 = box[1], box[1] + box[3]
        im = im.crop((x1, y1, x2, y2))
        # extract_text
        text_ocr = pytesseract.image_to_string(im, config='')
        text_ocr = text_ocr.replace("\x0c", "").replace("\n", "")
        return (self.map_categories[prediction['category_id']], text_ocr, length)


def open_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path,"w") as f:
        json.dump(data, f)

def open_txt(path):
  file_ = open(path,'r')
  file_ = file_.readlines()
  return file_

def save_txt(data, path):
    file_ = open(path, "w")
    file_.writelines(data)
    file_.close()
