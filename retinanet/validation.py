import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import csv_eval
#added:
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)
    #added below:
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser = parser.parse_args(args)

    ##dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              ##transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    #if torch.cuda.is_available():
       # retinanet.load_state_dict(torch.load(parser.model_path))
       # retinanet = torch.nn.DataParallel(retinanet).cuda()
    #else:
        #retinanet.load_state_dict(torch.load(parser.model_path))
        #retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()

    csv_eval.evaluate(dataset_val, retinanet)

if __name__ == '__main__':
    main()
