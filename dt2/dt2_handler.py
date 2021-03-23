# Some basic setup:
import detectron2
import os.path
import sys, io, json, time, random
import numpy as np
import cv2
import base64

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from os import path
from json import JSONEncoder

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.predictor = None
        self.model_file = "model_final.pth"
        self.config_file = "config.yaml"
        self.cfg = None  
        self.img_2 = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        print("initializing starting")

        print("File {} exists {}".format(self.model_file, str(path.exists(self.model_file))))
        print("File {} exists {}".format(self.config_file, str(path.exists(self.config_file))))

        try:
            cfg = get_cfg()
            cfg.MODEL.WEIGHTS = self.model_file
            print('model_file completed.')
            cfg.merge_from_file(self.config_file)
            print('config_file completed.')

            print(__file__)
            self.cfg = cfg

            # set the testing threshold for this model
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            
            self.predictor = DefaultPredictor(cfg)

            print("predictor built on initialize")
        except AssertionError as error:
            # Output expected AssertionErrors.
            print(error)
        except FileNotFoundError as error:
            print(error)
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            print("Error: {}".format(e))

        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True
        print("initialized")

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))

        # Take the input data and pre-process it make it inference ready
        print("pre-processing started for a batch of {}".format(len(batch)))

        images = []

        # batch is a list of requests
        for request in batch:
            for request_item in request:
                print(request_item)
                
            # each item in the list is a dictionary with a single body key, get the body of the request
            request_body = request.get("body")

            # read the bytes of the image
            input = io.BytesIO(request_body)

            # get our image
            img = cv2.imdecode(np.fromstring(input.read(), np.uint8), 1)

            # add the image to our list
            images.append(img)
            #self.img_2 = img

        print("pre-processing finished for a batch of {}".format(len(batch)))

        return images

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        # Do some inference call to engine here and return output
        print("inference started for a batch of {}".format(len(model_input)))

        outputs = []

        for image in model_input:
            # run our predictions
            output = self.predictor(image)

            outputs.append(output)

        print("inference finished for a batch of {}".format(len(model_input)))

        return outputs

    def postprocess(self, inference_output):

        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        start_time = time.time()
        
        print("post-processing started at {} for a batch of {}".format(start_time, len(inference_output)))
        
        responses = []

        for output in inference_output:

            # process predictions
            predictions = output["instances"].to("cpu")

            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
            # masks = (predictions.pred_masks > 0.5).squeeze().numpy() if predictions.has("pred_masks") else None        

            responses_json={'classes': classes.tolist(), 'scores': scores.tolist(), "boxes": [b.tolist() for b in boxes]} #, "masks": masks }
            responses.append(json.dumps(responses_json))

        elapsed_time = time.time() - start_time

        #from detectron2.utils.visualizer import Visualizer
        #from detectron2.data import MetadataCatalog

        #v = Visualizer(self.img_2[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        #v = v.draw_instance_predictions(output["instances"].to("cpu"))
        #v.save('output.jpg')
            
        print("post-processing finished for a batch of {} in {}".format(len(inference_output), elapsed_time))

        return responses

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print("handling started")

        # process the data through our inference pipeline
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        output = self.postprocess(model_out)

        print("handling finished")

        return output  

_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
