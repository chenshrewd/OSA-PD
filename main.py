import argparse
import datetime
import time
import matplotlib

matplotlib.use('Agg')
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import ResNet18, ResNet50
from resnet_cifar import ResNet18_CIFAR
import Sampling

import datasets
from utils import *

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
# parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--max-query', type=int, default=10)
# parser.add_argument('--max-query', type=int, default=5)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--query-strategy', type=str, default='OSA-PRD',
                    choices=['random', 'uncertainty', 'BADGE', 'CoreSet', 'MostConfidence', 'LeastConfidence', 'LfOSA', 'OSA-PRD', 'Full-Sampling', 'OSA-PDD'])
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='resnet_cifar')
# misc
parser.add_argument('--eval-freq', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--init-percent', type=int, default=8)
parser.add_argument('--known-T', type=float, default=0.5)
parser.add_argument('--unknown-T', type=float, default=0.5)
parser.add_argument('--modelB-T', type=float, default=1)
# DKL
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=10.0)
parser.add_argument('--T', type=float, default=6)

args = parser.parse_args()


def main():
    print("seed:", args.seed)
    full_data = False
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, args.query_strategy + '_' + args.dataset + 'seed_' + str(args.seed) + 'class' + str(args.known_class) + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))

    trainset, testset = datasets.get_dataset(args.dataset)

    labeled_ind_train, unlabeled_ind_train, unlabeled_ind_test, known_nums = datasets.get_openset(trainset, testset, args.known_class, args.init_percent)

    invalidList = []

    print("Creating model: {}".format(args.model))

    start_time = time.time()

    Acc = []
    Err = []
    Precision = []
    Recall = []
    Query_num = []
    Entropy = []

    for query in range(args.max_query):
        # Model initialization
        if args.model == "resnet18":
            detector = ResNet18(num_classes=args.known_class + 1)
            classifier = ResNet18(num_classes=args.known_class)
        elif args.model == "resnet_cifar":
            detector = ResNet18_CIFAR(num_classes=args.known_class + 1)
            classifier = ResNet18_CIFAR(num_classes=args.known_class)
        elif args.model == "resnet50":
            detector = ResNet50(num_classes=args.known_class + 1)
            classifier = ResNet50(num_classes=args.known_class)

        if use_gpu:
            detector = nn.DataParallel(detector).cuda()
            classifier = nn.DataParallel(classifier).cuda()

        criterion_xent = nn.CrossEntropyLoss()
        optimizer_detector = torch.optim.SGD(detector.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

        if args.stepsize > 0:
            scheduler_D = lr_scheduler.StepLR(optimizer_detector, step_size=args.stepsize, gamma=args.gamma)
            scheduler_C = lr_scheduler.StepLR(optimizer_classifier, step_size=args.stepsize, gamma=args.gamma)

        trainloader_D = datasets.get_loader(trainset, args.batch_size, args.workers, labeled_ind_train + invalidList, use_gpu)
        trainloader_C = datasets.get_loader(trainset, args.batch_size, args.workers, labeled_ind_train, use_gpu)

        # Model training
        for epoch in tqdm(range(args.max_epoch)):
            # Train model A for detecting unknown classes
            train_epoch_D(detector, criterion_xent, optimizer_detector, trainloader_D, use_gpu)
            # Train model B for classifying known classes
            train_epoch(classifier, criterion_xent, optimizer_classifier, trainloader_C, use_gpu)

            if args.stepsize > 0:
                scheduler_D.step()
                scheduler_C.step()

        print("==> Test")
        testloader = datasets.get_loader(testset, args.batch_size, args.workers, unlabeled_ind_test, use_gpu)
        acc_C, err_C = test(classifier, testloader, use_gpu)
        del testloader
        print("classifier | Accuracy (%): {}\t Error rate (%): {}".format(acc_C, err_C))

        # Record results
        Acc.append(float(acc_C))
        Err.append(float(err_C))
        # Query samples and calculate precision and recall
        unlabeledloader = datasets.get_myloader(trainset, args.batch_size, args.workers, unlabeled_ind_train, use_gpu)
        queryIndex = []
        if args.query_strategy == "random":
            queryIndex, invalidIndex, precision, recall = Sampling.random_sampling(args, unlabeledloader, len(labeled_ind_train), classifier, use_gpu)
        elif args.query_strategy[-10:] == "Confidence":
            queryIndex, invalidIndex, precision, recall = Sampling.confidence_sampling(args, unlabeledloader, len(labeled_ind_train), classifier, use_gpu, args.query_strategy)
        elif args.query_strategy == "uncertainty":
            queryIndex, invalidIndex, precision, recall = Sampling.uncertainty_sampling(args, unlabeledloader, len(labeled_ind_train), classifier, use_gpu)
        elif args.query_strategy == "BADGE":
            queryIndex, invalidIndex, precision, recall = Sampling.badge_sampling(args, unlabeledloader, len(labeled_ind_train), classifier, use_gpu)
        elif args.query_strategy == "CoreSet":
            queryIndex, invalidIndex, precision, recall = Sampling.coreset_sampling(args, trainloader_C, unlabeledloader, len(labeled_ind_train), classifier, use_gpu)
        elif args.query_strategy == "LfOSA":
            queryIndex, invalidIndex, precision, recall = Sampling.lfpsa_sampling(args, unlabeledloader, len(labeled_ind_train), detector, use_gpu)
        elif args.query_strategy == "OSA-PRD":
            queryIndex, invalidIndex, precision, recall = Sampling.osa_prd_sampling(args, trainloader_D, unlabeledloader, len(labeled_ind_train), detector, classifier, use_gpu)
        elif args.query_strategy == "Full-Sampling":
            queryIndex, invalidIndex, precision, recall = Sampling.supervised_learning(args, unlabeledloader, full_data)
        elif args.query_strategy == "OSA-PDD":
            queryIndex, invalidIndex, precision, recall = Sampling.osa_pdd_sampling(args, trainloader_D, unlabeledloader, len(labeled_ind_train), detector, classifier, use_gpu)
        else:
            sys.exit()

        if args.query_strategy == "test":
            break

        # Update labeled, unlabeled and invalid set

        del trainloader_D
        del trainloader_C
        del unlabeledloader
        Precision.append(precision)
        Recall.append(recall)
        unlabeled_ind_train = list(set(unlabeled_ind_train) - set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        invalidList = list(invalidList) + list(invalidIndex)

        print("Query Strategy: " + args.query_strategy + " | Query Batch: " + str(
            args.query_batch) + " | Valid Query Nums: " + str(len(queryIndex)) + " | Query Precision: " + str(
            precision) + " | Query Recall: " + str(recall) + " | Training Nums: " + str(
            len(labeled_ind_train)) + " | Unalebled Nums: " + str(len(unlabeled_ind_train)))
        Query_num.append(len(queryIndex))

        # 求当前训练data的平均entropy
        trainloader_C = datasets.get_loader(trainset, args.batch_size, args.workers, labeled_ind_train, use_gpu)
        info = get_dataentropy(args, trainloader_C, classifier, use_gpu)
        del trainloader_C
        Entropy.append(info)

        if len(labeled_ind_train) == known_nums:
            full_data = True
            print("full data train finish!")
            break


    if full_data:
        unlabeledloader = datasets.get_myloader(trainset, args.batch_size, args.workers, unlabeled_ind_train, use_gpu)
        queryIndex, invalidIndex, precision, recall = Sampling.supervised_learning(args, unlabeledloader, full_data)
        del unlabeledloader
        Precision.append(precision)
        Recall.append(recall)
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)

        # Model initialization
        if args.model == "resnet18":
            # 多出的一类用来预测为unknown
            classifier = ResNet18(num_classes=args.known_class)
        elif args.model == "resnet_cifar":
            classifier = ResNet18_CIFAR(num_classes=args.known_class)
        elif args.model == "resnet50":
            classifier = ResNet50(num_classes=args.known_class)

        if use_gpu:
            classifier = nn.DataParallel(classifier).cuda()

        criterion_xent = nn.CrossEntropyLoss()
        optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

        if args.stepsize > 0:
            scheduler_C = lr_scheduler.StepLR(optimizer_classifier, step_size=args.stepsize, gamma=args.gamma)

        # 创建dataloader
        trainloader_C = datasets.get_loader(trainset, args.batch_size, args.workers, labeled_ind_train, use_gpu)

        for epoch in tqdm(range(args.max_epoch)):
            # Train model B for classifying known classes
            train_epoch(classifier, criterion_xent, optimizer_classifier, trainloader_C, use_gpu)

            if args.stepsize > 0:
                scheduler_D.step()
                scheduler_C.step()

        print("==> Test")
        testloader = datasets.get_loader(testset, args.batch_size, args.workers, unlabeled_ind_test, use_gpu)
        acc_C, err_C = test(classifier, testloader, use_gpu)
        print("Full Data Training classifier | Accuracy (%): {}\t Error rate (%): {}".format(acc_C, err_C))


    print("Acc:", Acc)
    print("Err:", Err)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("Query Number:", Query_num)
    print("Entropy:", Entropy)

    ## Save model
    if args.is_mini:
        torch.save(classifier, "save_model/" + args.query_strategy + "_" + args.dataset + "_mini_query_" + str(args.query_batch) + "_seed" + str(args.seed) + "_class" + str(args.known_class) + ".pt")
    else:
        torch.save(classifier, "save_model/" + args.query_strategy + "_" + args.dataset + "_query_" + str(args.query_batch) + "_seed" + str(args.seed) + "_class" + str(args.known_class) + ".pt")

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train_epoch(model, criterion_xent, optimizer_model, trainloader, use_gpu):
    model.train()

    for _, (data, labels) in enumerate(trainloader):

        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)

        loss = criterion_xent(outputs, labels)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


def train_epoch_D(model, criterion_xent, optimizer_model, trainloader, use_gpu):
    model.train()

    known_T = args.known_T
    unknown_T = args.unknown_T
    invalid_class = args.known_class
    for _, (data, labels) in enumerate(trainloader):
        # Reduce temperature
        T = torch.tensor([known_T] * labels.shape[0], dtype=float)
        for i in range(len(labels)):
            # Annotate "unknown"
            if labels[i] >= invalid_class:
                labels[i] = invalid_class
                T[i] = unknown_T
        if use_gpu:
            data, labels, T = data.cuda(), labels.cuda(), T.cuda()
        features, outputs = model(data)
        outputs = outputs / T.unsqueeze(1)
        loss_xent = criterion_xent(outputs, labels)

        loss = loss_xent
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


def test(model, testloader, use_gpu):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for _, (data, labels) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':

    args.seed = 1
    main()
