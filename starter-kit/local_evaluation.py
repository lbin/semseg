import os
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from my_submission.aicrowd_wrapper import AIcrowdWrapper


def check_data(datafolder):
    """
    Checks if the data is downloaded and placed correctly
    """
    imagefolder = os.path.join(datafolder, 'inputs')
    annotationsfolder = os.path.join(datafolder, 'semantic_annotations')
    dl_text = ("Please download the public data from"
               "\n https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/semantic-segmentation/dataset_files"
               "\n And unzip it with ==> unzip <zip_name> -d public_dataset")
    if not os.path.exists(imagefolder):
        raise NameError(f'No folder named {imagefolder} \n {dl_text}')
    if not os.path.exists(annotationsfolder):
        raise NameError(f'No folder named {annotationsfolder} \n {dl_text}')

def dice(target, prediction):
    class_dice = []
    class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for class_val in class_list:
        class_annotation = target == class_val
        class_prediction = prediction == class_val
        numer = 2.0 * np.sum(class_annotation & class_prediction)
        denom = np.sum(class_annotation) + np.sum(class_prediction)
        if denom > 0:
            class_dice.append(numer/denom)
    dice_score = np.mean(class_dice)
    return dice_score

def mean_iou(target, prediction):
    class_ious = []
    class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for class_val in class_list:
        class_annotation = target == class_val
        class_prediction = prediction == class_val
        numer = np.sum(class_annotation & class_prediction)
        denom = np.sum(class_annotation | class_prediction)
        if denom > 0:
            class_ious.append(numer/denom)
    mean_iou_score = np.mean(class_ious)
    return mean_iou_score

def calculate_metrics(semantic_annotation, semantic_prediction):
    mean_iou_score  = mean_iou(semantic_annotation, semantic_prediction)
    dice_score = dice(semantic_annotation, semantic_prediction)

    metrics = {
                "segmentation_mIOU": mean_iou_score,
                "segmentation_dice": dice_score,
                }
    return  metrics

def read_image(path):
        image = np.array(Image.open(path))
        return image

def evaluate(LocalEvalConfig):
    """
    Runs local evaluation for the model
    Final evaluation code is the same as the evaluator
    """
    datafolder = LocalEvalConfig.DATA_FOLDER
    
    check_data(datafolder)

    imagefolder = os.path.join(datafolder, 'inputs')
    preds_folder = LocalEvalConfig.OUTPUTS_FOLDER

    model = AIcrowdWrapper(predictions_dir=preds_folder, dataset_dir=imagefolder)
    file_names = os.listdir(imagefolder)

    # Predict on all images
    for fname in tqdm(file_names, desc="Predicting Segmentation Masks"):
        model.segment_single_image(fname)

    # Evalaute metrics
    all_metrics = {}
    annotationsfolder = os.path.join(datafolder, 'semantic_annotations')
    for fname in tqdm(file_names, desc="Evaluating results"):
        semantic_annotation = read_image(os.path.join(annotationsfolder, fname))
        semantic_prediction = read_image(os.path.join(preds_folder, fname))
        all_metrics[fname] = calculate_metrics(semantic_annotation, semantic_prediction)
    
    metric_keys = list(all_metrics.values())[0].keys()
    metrics_lists = {mk: [] for mk in metric_keys}
    for metrics in all_metrics.values():
        for mk in metrics:
            metrics_lists[mk].append(metrics[mk])
    
    print("Evaluation Results")
    results = {key: np.mean(metric_list) for key, metric_list in metrics_lists.items()}
    for k,v in results.items():
        print(k,v)


if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        DATA_FOLDER = './public_dataset'
        OUTPUTS_FOLDER = './evaluator_outputs'

    outfolder=  LocalEvalConfig.OUTPUTS_FOLDER
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    
    evaluate(LocalEvalConfig)
