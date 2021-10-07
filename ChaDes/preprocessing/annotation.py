import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import os
import random
import json
import tqdm

from argparse import ArgumentParser
from PIL import Image


class AnnotationUpdater():
    """Class which changes annotations of axis in order to be similar to ChaTa annotations."""
    def __init__(
        self,
        annotation_path: str,
        save_annotation: bool = False,
        out_annotation_path: str = None
    ):
        """Initialization of class.

        Args:
            annotation_path (str): Path to image annotations.
            save_annotation (bool, optional): If updated annotations should
                be saved. Defaults to False.
            out_annotation_path (str, optional): Path to directory where
                updated annotation will be saved. Defaults to None.
        """
        self.save_annotation = save_annotation
        self.annotation_path = annotation_path
        self.out_annotation_path = out_annotation_path
        with open(annotation_path) as json_file:
            image_annotation = json.load(json_file)
        self.image_annotation = copy.deepcopy(image_annotation)
        self.update_status = False

    def update_annotations(self):
        """Update annotations. Change bounding box of y_axis and x_axis."""
        for i, annotation in tqdm.tqdm(enumerate(self.image_annotation)):
            title, img_type, y_axis, y_title, x_axis, x_title, y_label, x_label = self.read_annotation(annotation)
            # pie chart does not have those elements
            if img_type != "pie":
                # change annotation of axis boxes
                h, w, x = [], [], []
                for a in x_label:
                    h.append(a['h'])
                for a in y_label:
                    w.append(a['w'])
                    x.append(a['x'])
                max_w = max(w)
                max_h, min_x = max(h), min(x)
                self.image_annotation[i]['general_figure_info']['y_axis']['rule']['bbox']["w"] += max_w + 10
                self.image_annotation[i]['general_figure_info']['y_axis']['rule']['bbox']["x"] = min_x
                self.image_annotation[i]['general_figure_info']['x_axis']['rule']['bbox']["h"] += max_h + 10

                yaxis_label = self.image_annotation[i]['general_figure_info']['y_axis']['label']['text']
                xaxis_label = self.image_annotation[i]['general_figure_info']['x_axis']['label']['text']
                self.image_annotation[i].update({
                    "plot_desc":
                    "The figure presents the dependence of {} on {}.".format(
                            yaxis_label, xaxis_label
                        )
                })
            else:
                # empty description if it is pie chart
                self.image_annotation[i].update({"plot_desc": ""})

        # change status
        self.update_status = True

        # save
        if self.save_annotation:
            self._save()

        return self.image_annotation

    def _save(self):
        """Save annotation to chosen file."""        
        if not self.update_status:
            self.update_annotations()
        with open(self.out_annotation_path if self.out_annotation_path is not None else self.annotation_path, 'w') as outfile:
            json.dump(self.image_annotation, outfile)

    def read_annotation(self, annotation: dict):
        """Extract chosen parts of dictionary with annotations.

        Args:
            annotation (dict): Individual annotation.
        """
        title = annotation['general_figure_info']['title']['bbox']
        img_type = annotation['type']
        if img_type == "pie":
            y_axis, y_title, x_axis, x_title, y_label, x_label = None, None, None, None, None, None

        else:
            y_axis = annotation['general_figure_info']['y_axis']['rule']['bbox']
            y_title = annotation['general_figure_info']['y_axis']['label']['bbox']
            x_axis = annotation['general_figure_info']['x_axis']['rule']['bbox']
            x_title = annotation['general_figure_info']['x_axis']['label']['bbox']
            y_label = annotation['general_figure_info']['y_axis']['major_labels']['bboxes']
            x_label = annotation['general_figure_info']['x_axis']['major_labels']['bboxes']

        return title, img_type, y_axis, y_title, x_axis, x_title, y_label, x_label

    def plot_boundary_boxes(
        self,
        image_id: int,
        images_path: str,
        change_axis_box: bool = True,
        save_fig: bool = False
    ):
        """Plot with bounding boxes.

        Args:
            image_id (int): Identificator of image.
            images_path (str): Path to directory with images. Assumption:
                chosen image can be found in {images_path}{image_id}.png.
            change_axis_box (bool, optional): If annotation of axis bounding
                boxes should be changed. Defaults to True.
            save_fig (bool, optional): Whether save chart to png.
                Defaults to False.
        """
        if not self.update_status and change_axis_box:
            self.update_annotations()

        annotation = self.image_annotation[image_id]

        title, img_type, y_axis, y_title, x_axis, x_title, y_label, x_label = self.read_annotation(annotation)

        def add_rectangular(element):
            rect = patches.Rectangle(
                (element['x'], element['y']),
                element['w'],
                element['h'],
                linewidth=1, edgecolor='r', facecolor='none'
                )
            ax.add_patch(rect)

        # load image
        img = Image.open(images_path + str(image_id) + ".png")
        im = np.array(img, dtype=np.uint8)
        # create figure and axes
        fig, ax = plt.subplots(1, figsize=(15, 10))
        plt.axis('off')
        # display the image
        ax.imshow(im)
        # create a Rectangle patch for chart title
        add_rectangular(title)
        # create a Rectangle patch for chart y title
        add_rectangular(y_title)
        # create a Rectangle patch for chart x title
        add_rectangular(x_title)
        # create a Rectangle patch for chart axis
        add_rectangular(x_axis)
        add_rectangular(y_axis)
        # plot ticks if label is not changed
        if not change_axis_box:
            for y_tick in y_label:
                add_rectangular(y_tick)
            for x_tick in x_label:
                add_rectangular(x_tick)
        # save figureannotation_path
        if save_fig:
            plt.savefig(
                str(image_id) + 'vis.png',
                bbox_inches='tight',
                pad_inches=0
            )
        plt.show()


class AnnotationCOCO():

    def __init__(self, annotation_path, img_path, update_annotations=True):
        self.annotation_path = annotation_path
        self.img_path = img_path
        if update_annotations:
            # call class AnnotationUpdater
            self.annotations = AnnotationUpdater(annotation_path).update_annotations()
        else:
            with open(annotation_path) as json_file:
                self.annotations = json.load(json_file)

    def convert_to_coco_format(self, save=False, output_path=None):
        """Convert annotations which are formatted in PlotQA or FigureQA way to COCO format.
        
        Template of format:
        {
            'images' : [],
            'annotations: [],
            'categories': []
        }

        For more details please visit https://cocodataset.org/#format-data.
        """
        i = 0
        # categories of objects
        categories = self.create_categories()
        images = []
        annotations = []
        # loop over images
        for idx, annotation in enumerate(self.annotations):
            single_image = self.create_image_info(annotation)
            # append prepared dictionary
            images.append(single_image)
            # prepare dictionaries for 'annotations'
            annot, i = self.get_image_annotations(annotation, i)
            annotations.extend(annot)
        
        coco_annotations = {
            'images': images,
            'annotations': annotations, 
            'categories': categories
        }

        if save:
            self._save(coco_annotations, output_path)

        return coco_annotations   

    def create_categories(self):
        """Create dictionary of categories used in object detection."""        
        
        categories = [
            {'name': 'bar', 'id': 1, 'supercategory': 'visual'},
            {'name': 'dot_line', 'id': 2, 'supercategory': 'visual'},
            {'name': 'legend', 'id': 3, 'supercategory': 'visual'},
            {'name': 'line', 'id': 4, 'supercategory': 'visual'},
            {'name': 'title', 'id': 5, 'supercategory': 'textual'},
            {'name': 'xlabel', 'id': 6, 'supercategory': 'textual'},
            {'name': 'xticklabel', 'id': 7, 'supercategory': 'textual'},
            {'name': 'ylabel', 'id': 8, 'supercategory': 'textual'},
            {'name': 'yticklabel', 'id': 9, 'supercategory': 'textual'},
            {'name': 'x_axis', 'id': 10, 'supercategory': 'visual'},
            {'name': 'y_axis', 'id': 11, 'supercategory': 'visual'},
            {'name': 'legend_element', 'id': 13, 'supercategory': 'visual'},
            {'name': 'legend_element_desc', 'id': 14, 'supercategory': 'textual'}
        ]
        return categories

    def create_image_info(
        self, 
        annotation: dict
    ):
        """Create COCO format dictionary of image info.

        Args:
            annotation (dict): Annotation of single image.
        """        
        single_image = {}

        # prepare dictionary for 'images'
        filename = os.path.join(self.img_path, str(annotation["image_index"]) + ".png")
        img = Image.open(filename)        
        width, height = img.size
        single_image["file_name"] = str(annotation["image_index"]) + ".png"
        single_image["id"] = annotation["image_index"]
        single_image["height"] = height
        single_image["width"] = width

        return single_image

    @staticmethod
    def get_image_annotations(
        annotation,
        i
    ):
        def get_single_annotations(
            single_annotation_box: dict, 
            image_id: int, 
            category: int,
            annotation_id: int
        ):
            """Get single annotation and change it into COCO format.

            Note that box is [x, y, w, h]
            where:
            x, y: the upper-left coordinates of the bounding box
            width, height: the dimensions of your bounding box

            Args:
                single_annotation_box (dict): Dictionary with coordinates of bounding box.
                image_id (int): Image id.
                category (int): The id of category.
                annotation_id (int): The id of annotation.
            """
            x, y, w, h = single_annotation_box['x'], single_annotation_box['y'], single_annotation_box['w'], single_annotation_box['h']
            annotation = {}
            annotation['image_id'] = image_id
            annotation['bbox'] = [x, y, w, h]
            annotation['category_id'] = category
            annotation['iscrowd'] = 0
            annotation['id'] = annotation_id
            annotation['segmentation'] = [[x, y, x + w, y, x + w, y - h, x, y - h]] # left top, right top, right bottom, left bottom
            annotations['area'] = w * h
            return annotation
        annotations = []
        image_id = annotation['image_index']
        # bar
        if annotation["type"] in ["vbar_categorical", "hbar_categorical"]:
            for x in annotation['models']:
                for box in x["bboxes"]:
                    i += 1
                    coco_anotation = get_single_annotations(
                        single_annotation_box=box, 
                        image_id=image_id, 
                        category=1,
                        annotation_id=i
                    )
                    annotations.append(coco_anotation)
        # dot line
        if annotation["type"] == "dot_line":
            for x in annotation['models']:
                for box in x["bboxes"]:
                    i += 1
                    coco_anotation = get_single_annotations(
                        single_annotation_box=box, 
                        image_id=image_id, 
                        category=2,
                        annotation_id=i
                    )
                    annotations.append(coco_anotation)
        # legend 
        if "legend" in annotation["general_figure_info"].keys():
            i += 1
            coco_anotation = get_single_annotations(
                single_annotation_box=annotation['general_figure_info']["legend"]["bbox"], 
                image_id=image_id, 
                category=3,
                annotation_id=i
            )
            annotations.append(coco_anotation)
        # line
        if annotation["type"] == "line":
            for x in annotation['models']:
                for box in x["bboxes"]:
                    i += 1
                    coco_anotation = get_single_annotations(
                        single_annotation_box=box, 
                        image_id=image_id, 
                        category=4,
                        annotation_id=i
                    )
                    annotations.append(coco_anotation)
        # title
        i += 1
        coco_anotation = get_single_annotations(
            single_annotation_box=annotation['general_figure_info']['title']['bbox'],
            image_id=image_id, 
            category=5,
            annotation_id=i
        )
        annotations.append(coco_anotation)
        # x label
        if 'x_axis' in annotation['general_figure_info']:
            i += 1
            coco_anotation = get_single_annotations(
                single_annotation_box=annotation['general_figure_info']['x_axis']['label']['bbox'],
                image_id=image_id, 
                category=6,
                annotation_id=i
            )
            annotations.append(coco_anotation)
        # x tick labels
        if 'x_axis' in annotation['general_figure_info']:
            for box in annotation['general_figure_info']['x_axis']['major_labels']['bboxes']:
                i += 1
                coco_anotation = get_single_annotations(
                    single_annotation_box=box, 
                    image_id=image_id, 
                    category=7,
                    annotation_id=i
                )
                annotations.append(coco_anotation)
        # y label
        if 'y_axis' in annotation['general_figure_info']:
            i += 1
            coco_anotation = get_single_annotations(
                single_annotation_box=annotation['general_figure_info']['y_axis']['label']['bbox'], 
                image_id=image_id, 
                category=8,
                annotation_id=i
            )
            annotations.append(coco_anotation)
        # y tick label
        if 'y_axis' in annotation['general_figure_info']:
            for box in annotation['general_figure_info']['y_axis']['major_labels']['bboxes']:
                i += 1
                coco_anotation = get_single_annotations(
                    single_annotation_box=box, 
                    image_id=image_id, 
                    category=9,
                    annotation_id=i
                )
                annotations.append(coco_anotation)
        # x axis
        if 'x_axis' in annotation['general_figure_info']:
            i += 1
            coco_anotation = get_single_annotations(
                single_annotation_box=annotation['general_figure_info']['x_axis']['rule']['bbox'], 
                image_id=image_id, 
                category=10,
                annotation_id=i
            )
            annotations.append(coco_anotation)
        # y axis
        if 'y_axis' in annotation['general_figure_info']:
            i += 1
            coco_anotation = get_single_annotations(
                single_annotation_box=annotation['general_figure_info']['y_axis']['rule']['bbox'], 
                image_id=image_id, 
                category=11,
                annotation_id=i
            )
            annotations.append(coco_anotation)
        # legend element + legend element description
        if 'legend' in annotation['general_figure_info']:
            for elem in annotation['general_figure_info']['legend']['items']:
                i += 1
                coco_anotation = get_single_annotations(
                    single_annotation_box=elem['preview']['bbox'], 
                    image_id=image_id, 
                    category=13,
                    annotation_id=i
                )
                annotations.append(coco_anotation)
                i += 1
                coco_anotation = get_single_annotations(
                    single_annotation_box=elem['label']['bbox'], 
                    image_id=image_id, 
                    category=14,
                    annotation_id=i
                )
                annotations.append(coco_anotation)

        return annotations, i

    @staticmethod
    def _save(annotations, output_path):
        assert output_path is not None, "Please give an argument `output_path`."
        with open(output_path, 'w') as outfile:
            json.dump(annotations, outfile)


def open_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path,"w") as f:
        json.dump(data, f)


def convert_questions_to_answers(v1):

    """Create new key "answer_string" question converted to answer.

    Args:
        v1 (dict): Dictionary with Question-Answer pairs annotations 

    Function supports 9 types of questions from PlotQA corpora (Where ..., What ..., How ..., Is ..., Across ...,
    In ..., Does ..., Are ..., Do ...)

    """
    #1) Where
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Where"):
            answer_string = "The legend appears in the %s of the graph." % (annotation["answer"])
            annotation.update({'answer_string': answer_string})
    #2) What
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("What is the") and " per " in annotation["question_string"]:
            per_who  = annotation["question_string"].split(" per ")[1].split("?")[0]
            question =  "T" + annotation["question_string"].split("What is t")[1].split(" per ")[0]
            answer_string = question + " is %s per %s." % (annotation["answer"], per_who)
            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("What is the") and " in " in annotation["question_string"] and not " per " in annotation["question_string"]:
            answer_string = "The%s is %s."%(annotation["question_string"].split("What is the")[1].split(" ?")[0], annotation["answer"])
            if "?" in answer_string:
                answer_string = answer_string.replace("?","")
            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("What is the") and not " in " in annotation["question_string"] and not " per " in annotation["question_string"]:
            if "title" in annotation["question_string"]:
                answer = '"'+annotation["answer"]+'"'
            else:
                answer = annotation["answer"]
            answer_string = "T"+annotation["question_string"].split("What is t")[1].split("?")[0]+"is "+str(answer)+"."

            
            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("What does"):
            answer_string =  "T" + annotation["question_string"].split("What does t")[1].split("?")[0] + "%s."%(annotation["answer"].lower()) 

            annotation.update({'answer_string': answer_string})

    #3) How
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("How are"):
            answer_string = "Legend labels are stacked %sly."%(annotation["answer"])
            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("How many"):
            if "countries"  in annotation["question_string"]:
                if annotation["answer"]==0:
                    #If answer = 0
                    answer_string = "There are no countries in the graph."
            
                if annotation["answer"]==1:
                    #If answer = 1
                    answer_string = "There is one country in the graph."
                    
                
                if annotation["answer"]>1:
                    #If answer > 1
                    answer_string = "There are %s countries in the graph."%(annotation["answer"])
                
                    
                
            if "years"  in annotation["question_string"]:
                if annotation["answer"]==0:
                    #If answer = 0
                    answer_string = "There are no years in the graph."
                    
                if annotation["answer"]==1:
                    #If answer = 1
                    answer_string = "There is one year in the graph."
                    
                if annotation["answer"]>1:
                    #If answer > 1
                    answer_string = "There are %s years in the graph."%(annotation["answer"])
                    

            else:
                question = annotation["question_string"].split("How many ")[1].split(" are ther")[0]
                if "tick" in annotation["question_string"]:
                    end = annotation["question_string"].split(" are there")[1].split(" ?")[0]
                
                else:
                    end = ""
                if annotation["answer"]==0:
                    #If answer = 0
                    answer_string = "There are no %s%s."%(question, end)
                    
                    
                if annotation["answer"]==1:
                    #If answer = 1
                    answer_string = "There is one %s%s."%(question[:-1], end)
            
                
                if annotation["answer"]>1:
                    #If answer > 1
                    answer_string = "There are %s %s%s."%(annotation["answer"],question, end)
            
            annotation.update({'answer_string': answer_string})
                

    #4) Is
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Is") and "equal to the number of legend labels " in annotation["question_string"]:
            answer_string = "The " + annotation["question_string"].split("Is the ")[1].split(" equal to")[0]
            if annotation["answer"] == "Yes":
                answer_string += " is equal to the number of legend labels."
            else:
                answer_string += " is not equal to the number of legend labels."
            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Is the"):
            if "equal to the number of legend labels " not in annotation["question_string"]:
                if "less" in annotation["question_string"]:
                    answer_string = "The" +annotation["question_string"].split("Is the")[1].split(" less ")[0]
                    if annotation["answer"]=="Yes":
                        answer_string += " is less %s."%(annotation["question_string"].split(" less ")[1].split(" ?")[0])
                    else:
                        answer_string += " is greater %s."%(annotation["question_string"].split(" less ")[1].split(" ?")[0])
                
                        
                if "greater" in annotation["question_string"]:
                    answer_string = "The" +annotation["question_string"].split("Is the")[1].split(" greater ")[0]
                    if annotation["answer"]=="Yes":
                        answer_string += " is greater %s."%(annotation["question_string"].split(" greater ")[1].split(" ?")[0])
                    
                    else:
                        answer_string += " is less %s."%(annotation["question_string"].split(" greater ")[1].split(" ?")[0])
                    
                if "?" in answer_string:
                    answer_string = answer_string.replace("?","")
                annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Is it"):
        

            if "greater" in annotation["question_string"]:
                
        
                every_what = annotation["question_string"].split("the case that in every ")[1].split(",")[0]
            
        
                answer_string = "Every %s%s."%(every_what, 
                                                annotation["question_string"].split(",")[1].split("?")[0])
                    
                
                if annotation["answer"]=="No":
                    answer_string = answer_string.replace("is greater", "is not greater")
                

            annotation.update({'answer_string': answer_string})
    

    #5) Across
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Across"):
            answer_string = "%s is %s."%(annotation["question_string"].replace("what is ","").replace(" ?",""),
                                        annotation["answer"])

            annotation.update({'answer_string': answer_string})
    #6) In
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("In the"):
            year = annotation["question_string"].split("In the year ")[1].split(",")[0]
            answer_string = "The%sis %s in %s."%(annotation["question_string"].split("what is the")[1].split("?")[0], 
                                                        annotation["answer"],
                                                        year)
            annotation.update({'answer_string': answer_string})
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("In which"):
            minimum_or_maximum = "minimum" if "minimum" in annotation["question_string"] else "maximum"
            if "In which year" in annotation["question_string"]:
                answer_string = "The %s%swas in %s."%(minimum_or_maximum,
                                                annotation["question_string"].replace(minimum_or_maximum, "").split("In which year was the")[1].split("?")[0],
                                                annotation["answer"])
            else:
                answer_string = "The %s%swas in %s."%(minimum_or_maximum,
                                                annotation["question_string"].replace(minimum_or_maximum, "").split("In which country was the")[1].split("?")[0],
                                                annotation["answer"])
            
            
            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("In how many"):
            cases_years_countries = annotation["question_string"].split("In how many ")[1].split(",")[0]
            greater_not_equal = "greater" if "greater" in annotation["question_string"] else "not equal"


            answer_string = "The %sis %s across %s %s."%(annotation["question_string"].split(", is the ")[1].split(greater_not_equal)[0],
                                            greater_not_equal,   
                                            annotation["answer"],
                                            cases_years_countries)
    

            annotation.update({'answer_string': answer_string})


    #7) Does
    verbs = ["contain","appear","increase","intersect","spent"]
    it = 0
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Does"):
            cur_verb = [word for word in verbs if word in annotation["question_string"]][-1]
            subject = annotation["question_string"].split("Does")[1].split(cur_verb)[0]
            
            if not subject.startswith('"'):  
                subject = subject[1].upper() +subject[2:]
        
            object_ = annotation["question_string"].split(cur_verb)[1].split(" ?")[0]
            if annotation["answer"]=="Yes":
                answer_string = "%s%ss %s."%(subject, cur_verb, object_)
            else:
                answer_string = "%s doesn't %s%s."%(subject, cur_verb, object_)

        

            annotation.update({'answer_string': answer_string})


    #8) Are

    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Are"):
            word = ["horizontal","written","equal"]
            cur_verb = [w for w in word if w in annotation["question_string"]][-1]
        
            subject = annotation["question_string"].split("Are")[1].split(cur_verb)[0]
            if not subject.startswith('"'):  
                subject = subject[1].upper() +subject[2:]
            object_ = annotation["question_string"].split(cur_verb)[1].split("?")[0]
            if annotation["answer"]=="Yes":
                answer_string = "%sare %s%s."%(subject, cur_verb, object_)
            else:
                answer_string = "%sare not %s%s."%(subject, cur_verb, object_[:-1])
    
            annotation.update({'answer_string': answer_string})

    #9) Do
    verbs = ["contain","appear","increase","intersect","spent","have"]
    for annotation in v1["qa_pairs"]:
        if annotation["question_string"].startswith("Do") and not annotation["question_string"].startswith("Does"):
            cur_verb = [word for word in verbs if word in annotation["question_string"]][-1]
            subject = annotation["question_string"].split("Do")[1].split(cur_verb)[0]
            subject = subject[0].upper() + subject[1:]
    
            object_ =  annotation["question_string"].split(cur_verb)[1].split(" ?")[0]
            #print(annotation["answer"])
            if annotation["answer"]=="Yes":
                answer_string = "%s%ss%s."%(subject,cur_verb, object_)
            else:
                answer_string = "%sdoesn't %s%s."%(subject, cur_verb, object_)


            annotation.update({'answer_string': answer_string})

    for annotation in v1["qa_pairs"]:
        if "?" in annotation["answer_string"]:
            print(annotation["answer_string"])


    return v1

def divide_annotation_dictionary(image_annotations, single_annotation_path):

    if not os.path.exists(single_annotation_path):
        os.makedirs(single_annotation_path)

    sentence_counts = np.random.binomial(8, 0.5, len(image_annotations))

    for idx, annotation in tqdm.tqdm(enumerate(image_annotations.items())):
        split_annotations = annotation[1].split(". ")
        if sentence_counts[idx]> len(split_annotations):
            sentence_counts[idx]= len(split_annotations)
        if sentence_counts[idx]==0:
            sentence_counts[idx]= 3
        short_annotation = ". ".join(random.sample(split_annotations, sentence_counts[idx]))+"."

        file_ = open(os.path.join(single_annotation_path,str(annotation[0])+".txt"),"w")
        file_.writelines(short_annotation)
        file_.close()


def join_annotations(annotations_path, question_answer_pair_path, all_annotation_path_out, single_annotations_path):

    """Create one dictionary with image_index, dependency sentence and all answer_strings.

    Arguments:
        annotations_path (str): path to annotations.json
        question_answer_pair_path (str): path to qa_pairs.json (list of the dictionaries where each dictionary represents a question)
        all_annotation_path_out (str): path of the new dictionary
    """


    updater = AnnotationUpdater(annotations_path, False)
    print("Updater ready")
    annotations = updater.update_annotations()
    print("Image annotations ready")

    plot_desc = {}
    for ann in annotations:
        plot_desc[ann["image_index"]]=ann["plot_desc"]


    image_annotations = {}
    qa_pairs = open_json(question_answer_pair_path)
    qa_pairs = convert_questions_to_answers(qa_pairs)
    print("QA Pairs ready")

    for annotation in tqdm.tqdm(qa_pairs["qa_pairs"]):
    
        if annotation["image_index"] not in image_annotations:

            image_annotations[annotation["image_index"]]=str(plot_desc[annotation["image_index"]])+" "

            image_annotations[annotation["image_index"]]+=str(annotation["answer_string"])+" "
    
        else:
    
            image_annotations[annotation["image_index"]]+=annotation["answer_string"]+" "

    divide_annotation_dictionary(image_annotations, single_annotations_path)

    save_json(image_annotations, all_annotation_path_out)
