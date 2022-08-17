import argparse
import os

import cv2
import numpy as np
import setproctitle
import torch
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from torch import optim
from torch.autograd import Variable
from torchvision import transforms,models
import torch.nn.functional as F

import densenetAll

finalWidth = 224
finalHeight = 224

proc_train_dir = "fv_proc/train"
proc_test_dir = "fv_proc/test"

image_extentions = ['.png','.PNG','.JPG','jpg','BMP','bmp']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         normTransform])
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True} if args.cuda else {}
    train_data = ImageDataset(images_folder=proc_train_dir,  transform=transform)
    test_data = ImageDataset(images_folder=proc_test_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batchSz, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batchSz,shuffle=False, **kwargs)
    dataset_sizes = train_data.__len__()

    net = densenetAll.DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                               bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=106)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    # if args.cuda:
    #     net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)


    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    if (os.path.exists("latest.pth.pth")):
        Denset_proc_model = torch.load("latest.pth.pth")
    else:
        for epoch in range(1, args.nEpochs + 1):
            adjust_opt(args.opt, optimizer, epoch)
            train(args, epoch, net, train_loader, optimizer, trainF)
            test(args, epoch, net, test_loader, optimizer, testF)
            torch.save(net, os.path.join(args.save, 'latest.pth'))
            os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()



class ImageDataset(torch.utils.data.Dataset):

    def __init__(self,images_folder,transform=None):
        images_rgb = []
        # images_g = []
        labels = []
        for dirname in os.listdir(images_folder):
            for filename in os.listdir(images_folder+'/'+dirname):
                images_rgb.append((dirname + '/' + filename, int(dirname)))
                # if any(filename.endswith(extension) for extension in image_extentions):
                #     images_rgb.append((dirname+'/'+filename,int(dirname)))

        self.images_folder = images_folder
        self.transforms = transform
        self.images_rgb = images_rgb
        # self.images_g = images_g


    def __len__(self):
        return len(self.images_rgb)

    def __getitem__(self, index):
        filename, label = self.images_rgb[index]
        img = Image.open(os.path.join(self.images_folder,filename)).convert('RGB')
        # img_g = Image.open(os.path.join(self.images_folder, filename)).convert('L')
        img = self.transforms(img)
        # img_g = self.transforms(img_g)
        return img,label
        # return img, img_g, label

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
        print(batch_idx,data,target)
        print(target)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__=='__main__':
    main()




# def prepImageDir(dirPath):
#     images_rgb = []
#     images_g = []
#     labels = []
#     for root, dirs, files in os.walk(dirPath):
#         for name in files:
#             dir_names = root.split("\\")
#             image_rgb = cv2.imread (os.path.join(root, name))
#             image_g = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
#             images_rgb.append(image_rgb)
#             images_g.append(image_g)
#             labels.append("u_%s" % dir_names[2])
#     images_rgb_np = np.array(images_rgb)
#     images_g_np = np.array(images_g)
#     nsamples, nx, ny = images_g_np.shape
#     images_g_np = images_g_np.reshape((nsamples, nx*ny))
#     labels_np = np.array(labels)
#     return images_rgb_np, images_g_np, labels_np
#
# X_train_rgb_proc, X_train_g_proc, Y_train_label_proc = prepImageDir(proc_train_dir)
# X_test_rgb_proc, X_test_g_proc, Y_test_label_proc = prepImageDir(proc_test_dir)
#
# rf_proc = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# rf_proc.fit(X_train_g_proc, Y_train_label_proc)
# y_rf_proc_pred = rf_proc.predict(X_test_g_proc)
#
# nb_proc = GaussianNB()
# nb_proc.fit(X_train_g_proc, Y_train_label_proc)
# y_nb_proc_pred = nb_proc.predict(X_test_g_proc)
#
# encoder = LabelEncoder()
# encoder = encoder.fit(Y_train_label_proc)
#
# Y_train_proc_encoded = encoder.transform(Y_train_label_proc)
# Y_train_proc_encoded_dummy = np_utils.to_categorical(Y_train_proc_encoded)

