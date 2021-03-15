import argparse
import collections
import os.path
import os

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--model', help='Path to model (.pt) file.')

    round = 1
    epoch_completed = 12

    parser = parser.parse_args(args)

    retinanet = torch.load(parser.model)

    save_path = '/content/drive/My Drive/NRP 2020/retinanet/loss_record/'
    full_loss_record = 'full_loss_record_{}.txt'.format(round)
    loss_record = 'loss_record_{}.txt'.format(round)

    file_full = open(os.path.join(save_path, full_loss_record), "w")
    file_full.write("Epoch_completed = " + str(epoch_completed) + '\n')
    file_full.close()

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train() 
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    loss = []
    file_full = open(os.path.join(save_path, full_loss_record), "w")
    file = open(os.path.join(save_path, loss_record), "w")

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        loss_average = 0
        classification_loss_average = 0
        regression_loss_average = 0

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num+epoch_completed, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                loss_average += loss
                classification_loss_average += classification_loss
                regression_loss_average += regression_loss
                
                if epoch_num == 0:
                  file_full_write = open(os.path.join(save_path, loss_record), "w")
                  file_full_write.write("{} {} {} {} {}".format(epoch_num,iter_num,classification_loss,regression_loss,loss) + '\n')
                else:
                  file_full_read = open(os.path.join(save_path, loss_record), "r")
                  file_full_r = file_full_read.read()
                  file_full_read.close()
                  file_full_write = open(os.path.join(save_path, loss_record), "w")
                  file_full_r += "{} {} {} {} {}".format(epoch_num,iter_num,classification_loss,regression_loss,loss) + '\n'
                  file_full_write.write(file_r)
                file_full_write.close()

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        loss_average = loss_average / iter_num
        classification_loss_average = classification_loss_average / iter_num
        regression_loss_average = regression_loss_average / iter_num

        loss.append([epoch_num, loss_average])

        if epoch_num == 0:
          file_write = open(os.path.join(save_path, loss_record), "w")
          file_write.write('{} {} {} {}'.format(epoch_num,classification_loss_average,regression_loss_average,loss_average) + '\n')
        else:
          file_read = open(os.path.join(save_path, loss_record), "r")
          file_r = file_read.read()
          file_read.close()
          file_write = open(os.path.join(save_path, loss_record), "w")
          file_r += '{} {} {} {}'.format(epoch_num,classification_loss_average,regression_loss_average,loss_average) + '\n'
          file_write.write(file_r)
        file_write.close()

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num+epoch_completed+1))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
