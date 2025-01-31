import torch
from torch import nn
import matplotlib.pyplot as plt
from overwritten_layers import *
from utils import *
from einops import rearrange
import imageio

# from ours.Utils.utils import *   # Georgios
# from ours.Networks.overwritten_layers import *   # Georgios

class ViT_model(nn.Module):
    def __init__(self, n_classes=1000, img_size=(224, 224), patch_size=16, in_ch=3, embed_size=768,
                 n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0., n_blocks=12, mlp_hidden_ratio=4.,
                 device="cuda", max_epochs=10):
        super(ViT_model, self).__init__()

        self.train_history = {"loss": []}
        self.val_history = {"loss": []}
        self.min_val = np.inf

        self.current_epoch = 0
        self.max_epochs = max_epochs

        self.device = device
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.QKV_bias = QKV_bias

        self.add = Add()
        self.patch_embed = Img_to_patch(img_size, patch_size, in_ch, embed_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        set_seeds(0)
        no_grad_trunc_normal_(self.pos_embed, std=.02)
        no_grad_trunc_normal_(self.cls_token, std=.02)

        self.input_grad = None

        self.blocks = nn.ModuleList([Block(embed_size=self.embed_size, n_heads=self.n_heads,
                           QKV_bias=self.QKV_bias, att_dropout=att_dropout, out_dropout=out_dropout, mlp_hidden_ratio=4)
                                       for _ in range(self.n_blocks)])

        self.norm = LayerNorm(embed_size)
        self.head = Linear(self.embed_size, self.n_classes)

        self.pool = ClsSelect()

        self.to(self.device)


    def forward(self, x):

        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)# from Phil Wang
        x = torch.cat((cls_token, x), dim=1)
        x = self.add([x, self.pos_embed])# x+= self.positional_embed

        if x.requires_grad:
            x.register_hook(self.store_input_grad) ## When computing the grad wrt to the input x, store that grad to the model.input_grad

        for current_block in self.blocks:
            x = current_block(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, index=torch.tensor(0, device=x.device)) ## retrieve the cls
        x = x.squeeze(1)
        x = self.head(x)

        return x

    def store_input_grad(self, grad):
        print("")
        self.input_grad = grad

    def compute_att_rollout(self, all_relevances):
        b = 1 # only batch size of one is supported
        n = all_relevances[-1].shape[0]
        I = torch.eye(n).to(self.device)
        ## eq 13
        A = [all_relevances[index]+I for index in range(len(all_relevances))]

        att_rollout = A[0]
        for index in range(1,len(all_relevances)):
            att_rollout = A[index].matmul(att_rollout)
        return att_rollout


    def relevance_propagation(self, one_hot_label):

        ## from top to bottom
        relevance = self.head.relevance_propagation(one_hot_label)
        relevance = self.pool.relevance_propagation(relevance)
        relevance = self.norm.relevance_propagation(relevance)

        for current_block in reversed(self.blocks):
            relevance = current_block.relevance_propagation(relevance)

        all_relevances = []
        ## transformer_attribution
        for current_block in self.blocks:
            current_grad = current_block.attn.get_att_grad()
            current_relevance = current_block.attn.get_att_relevance()
            current_grad = current_grad.squeeze(0)
            current_relevance = current_relevance.squeeze(0)
            current_relevance *= current_grad

            ## considering only (+)
            current_relevance = current_relevance.clamp(min=0)

            ## averaging across the head dimension in accordance to Eq. 13
            current_relevance = current_relevance.mean(dim=0)

            all_relevances.append(current_relevance)

        att_rollout = self.compute_att_rollout(all_relevances)

        ## cls token
        att_rollout = att_rollout[0,1:]

        return att_rollout


    def extract_LRP(self, input, class_indices = None, sqrt=False):

        pred = self(input)

        if class_indices is None:
            class_indices = torch.argmax(pred, dim=1).data.cpu().numpy().tolist()


        one_hot = np.zeros((1, pred.shape[-1]), dtype=np.float32)
        one_hot[0, class_indices] = 1

        one_hot_label = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input.device) * pred)

        self.zero_grad()
        one_hot.backward(retain_graph=True) ## Register_hooks are excecuted in here

        att_rollout = self.relevance_propagation(torch.tensor(one_hot_label).to(input.device))


        ## reshaping att_rollout
        cue_size = int(self.patch_embed.n_patches ** (0.5))
        explainability_cue = att_rollout.reshape(1, 1,cue_size, cue_size)

        ## scaling to input's dimensions
        input_size = self.img_size[0]
        explainability_cue = torch.nn.functional.interpolate(
            explainability_cue, scale_factor=input_size//cue_size, mode='bilinear')[0,0]


        explainability_cue = explainability_cue.data.cpu().numpy()

        if sqrt == True:
            explainability_cue = np.sqrt(explainability_cue)

        explainability_cue = min_max_normalize(explainability_cue)

        explainability_cue = torch.from_numpy(explainability_cue)

        return explainability_cue.to(self.device), pred

    def train_epoch(self, dataloader, optimizer):


        train_loss = 0
        self.train()
        self.current_epoch += 1

        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]

            x = self(img)
            # explainability_cue, preds = self.extract_LRP(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            ### Printing epoch results
            print('Train Epoch: {}/{}\n'
                  'Step: {}/{}\n'
                  'Batch ~ Loss: {:.4f}\n'
                  .format(self.current_epoch, self.max_epochs,
                          index + 1, len(dataloader),
                          train_loss / (index + 1)))

        self.train_history["loss"].append(train_loss / len(dataloader))
        return

    def val_epoch(self, dataloader):

        val_loss = 0
        self.eval()
        with torch.no_grad():

            for index, data in enumerate(dataloader):

                img = data[1]
                label = data[2]

                x = self(img)

                loss = F.multilabel_soft_margin_loss(x, label)

                ### adding batch loss into the overall loss
                val_loss += loss.item()

                ### Printing epoch results
                print('Val Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      .format(self.current_epoch, self.max_epochs,
                              index + 1, len(dataloader),
                              val_loss/(index+1)))

            self.val_history["loss"].append(val_loss / len(dataloader))
        return

    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name + "/loss.png")
        plt.close()
        return

    def load_pretrained(self, weights_path):

        ## loading weights
        weights_dict = torch.load(weights_path)

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}

        no_pretrained_dict = {k: v for k, v in model_dict.items() if
                           not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def extract_metrics(self, dataloader, vis_class_top=True):
        ## stolen from https://github.com/hila-chefer/Transformer-Explainability
        ## Thanks Hila Chefer

        predictions, targets = [], []
        total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        total_ap = []

        for index, data in enumerate(dataloader):

            print(index/len(dataloader))

            img = data[0]
            label = data[1]

            vis_class = None if vis_class_top else label[0,0].data.cpu().numpy().tolist()
            explainability_cue, preds = self.extract_LRP(img, class_indices=vis_class)


            correct, labeled, inter, union, ap, pred, target = eval_batch(explainability_cue, label)

            predictions.append(pred)
            targets.append(target)

            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            total_ap += [ap]
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            mAp = np.mean(total_ap)

        return pixAcc, mIoU, mAp

    def extract_AUC(self, dataloader, transform, positive=False, vis_class_top=True):

        ## positive : True = when doing the positive perturbation and
        ##            False = when doing the negative
        ## vis_class_target: True = when extracting the explainability cue wrt the target class
        ##                   False  = when extracting the explainability cue wrt the predicted class

        pred_accuracy = np.zeros(10)

        for index, data in enumerate(dataloader):

            img = data[0]
            img_ = transform(img)
            label = data[1]

            vis_class = None if vis_class_top else label[0,0].data.cpu().numpy().tolist()
            explainability_cue, preds = self.extract_LRP(img_, class_indices=vis_class)

            pred_c = torch.argmax(preds).data.cpu().numpy()
            target_c = label[0,0].data.cpu().numpy()
            ####
            pred_accuracy[0] += pred_c == target_c

            if not positive:
                ## negative
                explainability_cue = - explainability_cue

            explainability_cue = explainability_cue.flatten()
            N_pixels = len(explainability_cue)

            for current_step in range(1, 10):

                current_img = img.clone() ## copying img
                _, indices_perturb = torch.topk(explainability_cue, int(N_pixels * current_step/10))
                indices_perturb = indices_perturb.repeat((3, 1)).unsqueeze(0)
                current_img = current_img.flatten(start_dim=-2, end_dim=-1)
                current_img = current_img.scatter_(-1, indices_perturb, 0)
                current_img = current_img.reshape(img.size())

                current_img = transform(current_img)

                current_preds = self(current_img)
                current_pred_c = torch.argmax(current_preds).data.cpu().numpy()
                pred_accuracy[current_step] += current_pred_c == target_c

        pred_accuracy /= len(dataloader)
        AUC = np.trapz(pred_accuracy, dx=0.1)

        return AUC


    def extract_LRP_for_affinity(self, dataloader, alpha_low=4, alpha_high=32,
                                 alpha_low_folder = "crf_lows/", alpha_high_folder = "crf_highs/",
                                 cam_folder = "cams/", pred_folder = "preds/"):

        if not os.path.exists(alpha_low_folder):
            os.makedirs(alpha_low_folder)

        if not os.path.exists(alpha_high_folder):
            os.makedirs(alpha_high_folder)

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)

        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)


        for index, data in enumerate(dataloader):

            print(index/len(dataloader))

            img_key = data[0][0].split("/")[-1][:-4]
            img = data[1]
            label = data[2]

            img_orig = plt.imread(data[0][0])
            img_orig_size = [i[0].data.cpu().numpy() for i in data[-1]]

            explainability_pred = np.zeros((self.n_classes, img_orig_size[0],
                                          img_orig_size[1]))

            explainability_bg = np.ones((img_orig_size[0],
                                          img_orig_size[1]))*0.2

            explainability_LRPs = {}
            vis_class = np.nonzero(label[0].data.cpu().numpy())[0]
            for current_vis_class in vis_class:

                for flip_flag in [0,1]:

                    if flip_flag:
                        current_explainability_cue_flipped, preds = self.extract_LRP(torch.flip(img, (0,3)) , class_indices=current_vis_class, sqrt=False)

                        current_explainability_cue_flipped = torch.nn.Upsample((img_orig_size[0], img_orig_size[1]),
                                                                       mode='bilinear') \
                            (current_explainability_cue_flipped.view(1, 1, *current_explainability_cue_flipped.shape))

                        current_explainability_cue_flipped = torch.flip(current_explainability_cue_flipped, (0,3))

                        current_explainability_cue+=current_explainability_cue_flipped
                        current_explainability_cue /=2
                    else:
                        current_explainability_cue, preds = self.extract_LRP(img,
                                                                             class_indices=current_vis_class, sqrt=False)

                        current_explainability_cue = torch.nn.Upsample((img_orig_size[0], img_orig_size[1]),
                                                                        mode='bilinear') \
                            (current_explainability_cue.view(1, 1, *current_explainability_cue.shape))

                explainability_LRPs[current_vis_class] = current_explainability_cue.data.cpu().numpy()[0,0]
                explainability_pred[current_vis_class] = current_explainability_cue.data.cpu().numpy()[0,0]

            explainability_pred = np.concatenate((explainability_bg[None,...], explainability_pred))
            explainability_pred = np.argmax(explainability_pred,axis=0)

            ## save cam
            np.save(cam_folder + img_key + ".npy", explainability_LRPs)

            ## pred
            imageio.imwrite(pred_folder + img_key + ".png", explainability_pred.astype(np.uint8))

            ### confident foreground
            LRP_v = np.array(tuple(explainability_LRPs.values()))
            bg_v = (1 - np.max(LRP_v,axis=0))**alpha_low

            v = np.concatenate((bg_v[None,...],LRP_v),axis=0)
            crf_low = crf_inference(img_orig, v, labels=LRP_v.shape[0]+1)

            crf_low_dict = {}
            crf_low_dict[0] = crf_low[0]
            for index, current_class in enumerate(vis_class):
                crf_low_dict[current_class+1] = crf_low[index+1]

            np.save(alpha_low_folder+img_key+".npy", crf_low_dict)

            ### confident background
            LRP_v = np.array(tuple(explainability_LRPs.values()))
            bg_v = (1 - np.max(LRP_v, axis=0)) ** alpha_high

            v = np.concatenate((bg_v[None, ...], LRP_v), axis=0)
            crf_high = crf_inference(img_orig, v, labels=LRP_v.shape[0] + 1)

            crf_high_dict = {}
            crf_high_dict[0] = crf_high[0]
            for index, current_class in enumerate(vis_class):
                crf_high_dict[current_class + 1] = crf_high[index + 1]

            np.save(alpha_high_folder + img_key + ".npy", crf_high_dict)



class Img_to_patch(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, input_ch=3, embed_size=768):
        super(Img_to_patch, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_ch = input_ch
        self.embed_size = embed_size
        ## TODO architecture
        self.proj = Conv2d(self.input_ch, self.embed_size, kernel_size=(self.patch_size, self.patch_size),
                             stride=(self.patch_size, self.patch_size))

        self.n_patches = (img_size[1] // self.patch_size) * (img_size[0] // self.patch_size)

    def forward(self, x):

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention_layer(nn.Module):

    def __init__(self, embed_size=768, n_heads=12, QKV_bias=False, att_dropout=0., out_dropout=0.):
        super().__init__()

        self.n_heads = n_heads
        self.QKV_bias = QKV_bias

        head_dim = embed_size // n_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = Linear(embed_size, embed_size * 3, bias=self.QKV_bias)
        self.proj = Linear(embed_size, embed_size)

        # A = Q*K^T
        self.matmul1 = Matmul(transpose=True)
        # att = A*V
        self.matmul2 = Matmul(transpose=False)
        self.att_softmax = Softmax(dim=-1)


        self.att_dropout = Dropout(att_dropout)
        self.out_dropout = Dropout(out_dropout)

        self.v = None
        self.att = None
        self.att_grad = None

        self.v_relevance = None
        self.att_relevance = None

    # TODO eliminated those not being used

    def store_v(self, v):
        self.v = v

    def store_att(self, att):
        self.att = att

    def store_att_grad(self, grad):
        self.att_grad = grad

    def store_v_relevance(self, relevance):
        self.v_relevance = relevance

    def store_att_relevance(self, relevance):
        self.att_relevance = relevance

    def get_v(self):
        return self.v

    def get_att(self):
        return self.att

    def get_att_grad(self):
        return self.att_grad

    def get_v_relevance(self):
        return self.v_relevance

    def get_att_relevance(self):
        return self.att_relevance


    def forward(self, x):
        batch, n, embed_size = x.shape
        qkv = self.qkv(x)

        ## ours
        qkv = torch.reshape(qkv, (batch, n, 3, self.n_heads, embed_size//self.n_heads))
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        self.store_v(v)

        # A = Q * K.T
        scaled_products = self.matmul1([q, k]) * self.scale

        att = self.att_softmax(scaled_products)
        att = self.att_dropout(att)

        self.store_att(att)

        if att.requires_grad:
            att.register_hook(self.store_att_grad)

        # att = A*V
        x = self.matmul2([att, v])
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (batch, n, embed_size))


        x = self.proj(x)
        x = self.out_dropout(x)

        return x

    def relevance_propagation(self, relevance):
        batch, n, embed_size = relevance.shape

        relevance = self.out_dropout.relevance_propagation(relevance)
        relevance = self.proj.relevance_propagation(relevance)

        relevance = torch.reshape(relevance,
                            (batch, n, self.n_heads, embed_size//self.n_heads))
        relevance = relevance.permute(0, 2, 1, 3)


        relevance, relevance_v = self.matmul2.relevance_propagation(relevance)
        ## TODO why? /2
        relevance /=2
        relevance_v /=2

        self.store_v_relevance(relevance_v)
        self.store_att_relevance(relevance)

        relevance = self.att_dropout.relevance_propagation(relevance)
        relevance = self.att_softmax.relevance_propagation(relevance)

        relevance_q, relevance_k = self.matmul1.relevance_propagation(relevance)

        ## TODO why? /2
        relevance_q /=2
        relevance_k /=2

        relevance_qkv = torch.stack([relevance_q,
                                   relevance_k,
                                   relevance_v])

        relevance_qkv = relevance_qkv.permute(1, 3, 0, 2, 4)
        relevance_qkv = torch.reshape(relevance_qkv, (batch, n, 3*embed_size))

        relevance_qkv = self.qkv.relevance_propagation(relevance_qkv)

        return relevance_qkv

class Mlp(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, dropout=0.):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, in_dim)
        self.dropout = Dropout(dropout)
        self.gelu = GELU()



    def forward(self, x):
        ## FC1
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        ## FC2
        x = self.fc2(x)
        x = self.dropout(x)

        return x

    def relevance_propagation(self, relevance):
        ## FC2
        relevance = self.dropout.relevance_propagation(relevance)
        relevance = self.fc2.relevance_propagation(relevance)

        ## FC1
        relevance = self.dropout.relevance_propagation(relevance)
        relevance = self.gelu.relevance_propagation(relevance)
        relevance = self.fc1.relevance_propagation(relevance)

        return relevance

class Block(nn.Module):

    def __init__(self, embed_size=768, n_heads=12, QKV_bias=True, att_dropout=0., out_dropout=0.,
                 mlp_hidden_ratio=4):
        super().__init__()

        self.embed_size = embed_size
        self.n_heads = n_heads
        self.QKV_bias = QKV_bias
        self.mlp_hidden_ratio = mlp_hidden_ratio


        ## MLP layer
        self.mlp = Mlp(embed_size, hidden_dim=int(self.mlp_hidden_ratio*self.embed_size), dropout=out_dropout)

        ## Attention layer
        self.attn = Attention_layer(embed_size=self.embed_size, n_heads=self.n_heads, QKV_bias=self.QKV_bias,
                                   att_dropout=att_dropout, out_dropout=out_dropout)

        ## Normalization layers
        self.norm1 = LayerNorm(self.embed_size, eps=1e-6)
        self.norm2 = LayerNorm(self.embed_size, eps=1e-6)

        self.add1 = Add()
        self.add2 = Add()

        self.clone1 = Clone()
        self.clone2 = Clone()

    ###### GM NEW ###### todo --> remove comment after explaining
    def relevance_propagation(self, relevance):
        (relevance, relevance_dupl) = self.add2.relevance_propagation(relevance)
        relevance_dupl = self.mlp.relevance_propagation(relevance_dupl)
        relevance_dupl = self.norm2.relevance_propagation(relevance_dupl)
        relevance = self.clone2.relevance_propagation((relevance, relevance_dupl))

        (relevance, relevance_dupl) = self.add1.relevance_propagation(relevance)
        relevance_dupl = self.attn.relevance_propagation(relevance_dupl)
        relevance_dupl = self.norm1.relevance_propagation(relevance_dupl)
        relevance = self.clone1.relevance_propagation((relevance, relevance_dupl))

        return relevance



    def forward(self,x):

        x1, x2 = self.clone1(x, 2)
        x2 = self.norm1(x2)
        x2 = self.attn(x2)
        x = self.add1([x1, x2])

        x1, x2 = self.clone2(x, 2)
        x2 = self.norm2(x2)
        x2 = self.mlp(x2)
        x = self.add2([x1, x2])

        return x

