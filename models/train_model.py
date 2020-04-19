import torch, time, copy, sys, os
from sklearn import metrics
import glob
from tqdm import tqdm
from meters import AverageMeter
from sklearn.metrics import roc_auc_score
# import keras.backend as K


# def binary_crossentropy_with_ranking(preds, y_true, labels_onehot, outputs, criterion):
def binary_crossentropy_with_ranking(outputs, labels_onehot):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    y_pred = outputs
    y_true = labels_onehot
    # logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

    # next, build a rank loss

    # #########################################################################clip the probabilities to keep stability
    # y_pred_clipped = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    epsilon = 10e-8
    y_pred_clipped = y_pred.clamp(epsilon, 1-epsilon)
    y_pred_clipped = y_pred_clipped.cpu().detach()
    # print("y_pred_clipped", y_pred_clipped)
    # ###################################################################translate into the raw scores before the logit
    # y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))
    # y_pred_score = torch.log(y_pred_clipped / (1 - y_pred_clipped))
    y_pred_score = y_pred_clipped
    # print("y_pred_score", y_pred_score)

    # ###########################################################determine what the maximum score for a zero outcome is
    # print("y_true", y_true)
    # print("y_true < 1", y_true < 1)
    y_true_1 = (y_true < 1)
    y_true_1 = y_true_1.type(torch.FloatTensor)
    mul = y_pred_score * y_true_1
    # print("mul", mul)
    # y_pred_score_zerooutcome_max = K.max(mul)
    y_pred_score_zerooutcome_max = torch.max(mul)
    # print("y_pred_score_zerooutcome_max", y_pred_score_zerooutcome_max)

    # ###############################################################determine how much each score is above or below it
    rankloss = y_pred_score.cpu() - y_pred_score_zerooutcome_max

    # ###########################################################################only keep losses for positive outcomes
    rankloss = rankloss * y_true.cpu()

    # ################################################################only keep losses where the score is below the max
    # rankloss = K.square(K.clip(rankloss, -100, 0))
    rankloss = rankloss.clamp(-100, 0) ** 2

    # ##################################################################average the loss for just the positive outcomes
    # rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)
    rankloss = torch.sum(rankloss.cpu()) / (torch.sum(y_true > 0) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss


def train_model(output_path, model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=5, scheduler=None):
    if not os.path.exists('iterations/' + str(output_path) + '/saved'):
        os.makedirs('iterations/' + str(output_path) + '/saved')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    val_auc_all = []
    val_acc_all = []
    test_auc_all = []
    test_acc_all = []
    TPFPFN0_all = []
    TPFPFN1_all = []
    best_val_auc = 0.0
    best_epoch = 0
    for epoch in range(1, num_epochs+1):
        print('-' * 50)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # tqdm_loader = tqdm(dataloaders[phase])
            # for data in tqdm_loader:
            #     inputs, labels = data
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2)
                labels_onehot = labels_onehot.type(torch.FloatTensor)

                # BCEloss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.cpu(), labels_onehot, torch.FloatTensor([1.0, 1.0]))
                BCEloss = criterion(outputs.cpu(), labels_onehot)
                # print("BCEloss", BCEloss)
                BCEloss_rank = binary_crossentropy_with_ranking(outputs, labels_onehot)
                # print("BCEloss_rank", BCEloss_rank)
                # BCEloss_rank.requires_grad = True
                loss = BCEloss + 0 * BCEloss_rank
                # print("BCEloss, BCEloss_rank", BCEloss, BCEloss_rank)
                # loss = (BCEloss_rank + 1) * BCEloss

                loss.backward()
                optimizer.step()

                losses.update(loss.item(), inputs.size(0))
                acc = float(torch.sum(preds == labels.data)) / preds.shape[0]
                accuracies.update(acc)
                all_preds += list(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().data.numpy())
                all_labels += list(labels.cpu().data.numpy())
                # tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

            auc = roc_auc_score(all_labels, all_preds)

            if phase == 'train':
                auc_t = auc
                loss_t = losses.avg
                acc_t = accuracies.avg
            if phase == 'val':
                auc_v = auc
                loss_v = losses.avg
                acc_v = accuracies.avg
                val_acc_all.append(acc_v)
                val_auc_all.append(auc_v)

        print('Train AUC: {:.8f} Loss: {:.8f} ACC: {:.8f} ' .format(auc_t, loss_t, acc_t))
        print('Val AUC: {:.8f} Loss: {:.8f} ACC: {:.8f} ' .format(auc_v, loss_v, acc_v))
        if auc_v > best_val_auc:
            best_val_auc = auc_v
            best_epoch = epoch
            # print(auc_v, best_val_auc)
            # print(best_epoch)
            best_model = copy.deepcopy(model)

        torch.save(model.module, './iterations/' + str(output_path) + '/saved/model_{}_epoch.pt'.format(epoch))
    # ############################################################################################################# Test
        for phase in ['test']:
            model.eval()  # Set model to evaluate mode

            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)

                acc = float(torch.sum(preds == labels.data)) / preds.shape[0]
                accuracies.update(acc)
                all_preds += list(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().data.numpy())
                all_labels += list(labels.cpu().data.numpy())
                # tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

            auc = roc_auc_score(all_labels, all_preds)

            auc_test = auc
            loss_test = losses.avg
            acc_test = accuracies.avg
            test_acc_all.append(acc_test)
            test_auc_all.append(auc_test)

        print('Test AUC: {:.8f} Loss: {:.8f} ACC: {:.8f} ' .format(auc_test, loss_test, acc_test))

        nb_classes = 2
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            TrueP0 = 0
            FalseP0 = 0
            FalseN0 = 0
            TrueP1 = 0
            FalseP1 = 0
            FalseN1 = 0
            for i, (inputs, classes) in enumerate(dataloaders[phase]):
                confusion_matrix = torch.zeros(nb_classes, nb_classes)
                input = inputs.to(device)
                target = classes.to(device)
                outputs = model(input)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(target.view(-1), preds.view(-1)):
                    confusion_matrix[t, p] += 1
                this_class = 0
                col = confusion_matrix[:, this_class]
                row = confusion_matrix[this_class, :]
                TP = row[this_class]
                FN = sum(row) - TP
                FP = sum(col) - TP
                # print("TP, FP, FN: ", TP, FP, FN)
                TrueP0 = TrueP0 + TP
                FalseP0 = FalseP0 + FP
                FalseN0 = FalseN0+ FN

                this_class = 1
                col = confusion_matrix[:, this_class]
                row = confusion_matrix[this_class, :]
                TP = row[this_class]
                FN = sum(row) - TP
                FP = sum(col) - TP
                # print("TP, FP, FN: ", TP, FP, FN)
                TrueP1 = TrueP1 + TP
                FalseP1 = FalseP1 + FP
                FalseN1 = FalseN1 + FN
            TPFPFN0 = [TrueP0, FalseP0, FalseN0]
            TPFPFN1 = [TrueP1, FalseP1, FalseN1]
            TPFPFN0_all.append(TPFPFN0)
            TPFPFN1_all.append(TPFPFN1)
            print("overall_TP, FP, FN for 0: ",  TrueP0, FalseP0, FalseN0)
            print("overall_TP, FP, FN for 1: ",  TrueP1, FalseP1, FalseN1)


    print("best_ValidationEpoch:", best_epoch)
    # print(TPFPFN0_all, val_auc_all, test_auc_all)
    TPFPFN0_best = TPFPFN0_all[best_epoch-1][0]
    TPFPFN1_best = TPFPFN1_all[best_epoch-1][0]
    val_auc_best = val_auc_all[best_epoch-1]
    val_acc_best = val_acc_all[best_epoch-1]
    test_auc_best = test_auc_all[best_epoch-1]
    test_acc_best = test_acc_all[best_epoch-1]

    # #################### save only the best, delete others
    file_path = './iterations/' + str(output_path) + '/saved/model_' + str(best_epoch) + '_epoch.pt'
    if os.path.isfile(file_path):
        for CleanUp in glob.glob('./iterations/' + str(output_path) + '/saved/*.pt'):
            if 'model_' + str(best_epoch) + '_epoch.pt' not in CleanUp:
                os.remove(CleanUp)
    # # ######################################################

    return best_epoch, best_model, TPFPFN0_all[best_epoch-1], TPFPFN1_all[best_epoch-1], test_acc_best, test_auc_best









# def binary_crossentropy_with_ranking(y_true, y_pred):
#     """ Trying to combine ranking loss with numeric precision"""
#     # first get the log loss like normal
#     logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
#
#     # next, build a rank loss
#
#     # clip the probabilities to keep stability
#     y_pred_clipped = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#
#     # translate into the raw scores before the logit
#     y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))
#
#     # determine what the maximum score for a zero outcome is
#     y_pred_score_zerooutcome_max = K.max(y_pred_score * (y_true < 1))
#
#     # determine how much each score is above or below it
#     rankloss = y_pred_score - y_pred_score_zerooutcome_max
#
#     # only keep losses for positive outcomes
#     rankloss = rankloss * y_true
#
#     # only keep losses where the score is below the max
#     rankloss = K.square(K.clip(rankloss, -100, 0))
#
#     # average the loss for just the positive outcomes
#     rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)
#
#     # return (rankloss + 1) * logloss - an alternative to try
#     return rankloss + logloss