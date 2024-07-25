import os
from model import MMVH
import torch.optim as optim
import eval as Eval
import time
from loss import InfoNCE
import torch


def train_model(
        train_loader=None
        , q_image_loader=None
        , r_image_loader=None
        , bert_config='./BertConfig'
        , cls_path='./cls.pt'
        , hashembeding_dim=64
        , device='cuda:6'
        , margin=1.0
        , max_iter=100
        , lr=1e-4
        , fileindex='0'
        , result_log_dir = None
        , result_weight_dir = None
        , weight_path = None
):
    with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
        f.write('margin = {}\n'.format(margin))
        f.write('hashembeding_dim = {}\n'.format(hashembeding_dim))
        f.write('lr = {}\n'.format(lr))
        f.close()

    # Device
    use_device = torch.device(device)

    model = MMVH(model_config_path=bert_config, cls_path=cls_path,hashcode_size=hashembeding_dim).to(device)
    for name, param in model.named_parameters():
        print(name)

    if weight_path:
        state_dict = torch.load(weight_path,map_location=device)
        model.load_state_dict(state_dict['berthash'],strict=True)
        print("加载成功！")

    # Create criterion
    criterion = InfoNCE(negative_mode='paired')

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-3)

    best_acc = 0.
    since_time = time.time()
    for epoch in range(0, max_iter):
        # Logger
        print('Epoch {}/{}'.format(epoch + 1, max_iter))
        print('-' * 20)
        with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
            f.write('Epoch {}/{}\n'.format(epoch + 1, max_iter))
            f.write('-' * 20)
            f.write('\n')
            f.close()

        # Trianing
        model.train()

        t_loss = 0.

        time_start = time.time()

        for index, data in enumerate(train_loader):
            # print(f'{epoch} {index}')
            if index % 50 == 0:
                print(f'{epoch} {index}')

            anchorI, anchorA, posI, posA, negI, negA = data

            anchorI = anchorI.to(use_device)
            anchorA = anchorA.to(use_device)
            posI = posI.to(use_device)
            posA = posA.to(use_device)
            negI = negI.to(use_device)
            negA = negA.to(use_device)

            optimizer.zero_grad()

            batch, negs = negI.shape[0], negI.shape[1]

            anchor_A_enc_pooler_output, anchor_I_enc_pooler_output, anchor_vh = model(anchorI, anchorA)

            pos_A_enc_pooler_output, pos_I_enc_pooler_output, pos_vh = model(posI, posA)

            negI_ = negI.view(-1,25,768)
            negA_ = negA.view(-1,25,768)
            neg_A_enc_pooler_output, neg_I_enc_pooler_output, neg_vh = model(negI_, negA_)
            neg_A_enc_pooler_output = neg_A_enc_pooler_output.view(batch,negs,-1)
            neg_I_enc_pooler_output = neg_I_enc_pooler_output.view(batch,negs,-1)
            neg_vh = neg_vh.view(batch,negs,-1)

            # A 类内部
            intra_A = criterion(anchor_A_enc_pooler_output, pos_A_enc_pooler_output, neg_A_enc_pooler_output)
            # I 类内部
            intra_I = criterion(anchor_I_enc_pooler_output, pos_I_enc_pooler_output, neg_I_enc_pooler_output)
            # 类间 1
            inter_AI = criterion(anchor_A_enc_pooler_output, pos_I_enc_pooler_output, neg_I_enc_pooler_output)
            # 类间 2
            inter_IA = criterion(anchor_I_enc_pooler_output, pos_A_enc_pooler_output, neg_A_enc_pooler_output)
            
            # 视频
            lv = criterion(anchor_vh, pos_vh, neg_vh)

            loss = lv + 50*(intra_A + intra_I + inter_IA + inter_AI)


            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        epoch_loss = t_loss / (len(train_loader.dataset) / 128)

        print('Train Loss: {:.6f}'.format(epoch_loss))
        time_end = time.time() - time_start
        print('Epoch training in {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60))

        with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
            f.write('Train Loss: {:.6f}\n'.format(epoch_loss))
            f.close()

        # Val or Test:
        model.eval()
        with torch.no_grad():

            q_image_code, q_image_targets = generate_code(model, q_image_loader, hashembeding_dim, use_device)
            r_image_code, r_image_targets = generate_code(model, r_image_loader, hashembeding_dim, use_device)

        mAP = Eval.mean_average_precision(
            q_image_code.to(device),
            r_image_code.to(device),
            q_image_targets.to(device),
            r_image_targets.to(device),
            use_device
        )

        result = 'mAP: {:.4f}'.format(mAP)
        print(result)

        with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
            f.write(result + '\n')
            f.close()

        if mAP > best_acc:
            best_acc = mAP
            model_state_dict = model.state_dict()
            dir = os.path.join(result_weight_dir, fileindex + '_' + str(hashembeding_dim) + '.pth')
            state = {
                'berthash': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, dir)
            print(str(epoch + 1) + 'saved')

    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))


def generate_code(model, dataloader, code_length, device):
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        numclass = dataloader.dataset.num_classes
        code = torch.zeros([N, code_length])
        target = torch.zeros([N, numclass])
        for image_features, audio_features,  tar, index in dataloader:
            image_features = image_features.to(device)
            audio_features = audio_features.to(device)

            _,_,hash_code = model(image_features, audio_features)
            code[index, :] = hash_code.sign().cpu()
            target[index, :] = tar.clone().cpu()
    torch.cuda.empty_cache()
    return code, target
