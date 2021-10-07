from keras.models import load_model
from lime import lime_image
import os
from skimage import io, transform
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class_map = {0: "dot_line", 1: "hbar_categorical", 2:"line", 3:"pie", 4:"vbar_categorical"}

class Explainer:

    def __init__(
        self,
        path_to_model: str,
    ):
        """Class initization.

        Arguments:
            path_to_model {str} -- path to model
        """
        self.model = load_model(path_to_model)
        self.explainer = lime_image.LimeImageExplainer()

    def explain_single(
        self,
        path_to_figure: str,
        explanation_path = "results" 
    ):
        """Create explanation for single figure and save to png file.

        Arguments:
            path_to_figure {str} --path to figuretra

        """
        ##TODO: SHOW GROUND TRUTH

        exp_path = os.path.join(explanation_path, path_to_figure.split("/")[-1]).split(".")[0]
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        input_shape = self.model.input_shape
        figure = io.imread(path_to_figure)[:,:,:3]


        figure = transform.resize(figure, (input_shape[1],input_shape[2])) 
        print("\nTransformed to (%s,%s)"%(input_shape[1],input_shape[2]))
        
        preds = self.model.predict(figure[np.newaxis,:,:,:3])
      
        print("\nTop prediction classes: ", [class_map[prediction] for prediction in preds[0].argsort()[::-1]])
        print("with probabilities: ", self.model.predict(figure[np.newaxis,:,:,:3])) 
        print("\nPrediction class:", class_map[preds[0].argsort()[::-1][0]],"\n")

        explanation = self.explainer.explain_instance(figure, self.model.predict, top_labels=5, hide_color=0, num_samples=1000)
       

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        plt.imsave(os.path.join(exp_path,"only_positive.png"),mark_boundaries(temp / 2 + 0.5, mask))
  
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        plt.imsave(os.path.join(exp_path,"pos_rest_division.png"),mark_boundaries(temp / 2 + 0.5, mask))

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        plt.imsave(os.path.join(exp_path,"pros_cons.png"),mark_boundaries(temp / 2 + 0.5, mask))

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
        plt.imsave(os.path.join(exp_path,"pros_cons_high_weight.png"), mark_boundaries(temp / 2 + 0.5, mask))
