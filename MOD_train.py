import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
import wandb 
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# from pytorch_lightning.utilities.seed import seed_everything
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from torch.nn import functional as F




from src.genie.data_module import RAMSDataModule
from src.genie.ACE_data_module import ACEDataModule
from src.genie.KAIROS_data_module import KAIROSDataModule 
from src.genie.model import GenIEModel 


logger = logging.getLogger(__name__)


def cross_entropy_loss(logits, labels):
  return F.nll_loss(logits, labels)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    device = torch.device('cuda')

    # Required parameters
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['gen','constrained-gen']
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['RAMS', 'ACE', 'KAIROS', 'KAIROS0']
    )
    parser.add_argument('--tmp_dir', type=str)
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--load_ckpt",
        default=None,
        type=str, 
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
    )
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--coref_dir', type=str, default='data/kairos/coref_outputs')
    parser.add_argument('--use_info', action='store_true', default=False, help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--mark_trigger', action='store_true')
    parser.add_argument('--sample-gen', action='store_true', help='Do sampling when generation.')
    parser.add_argument('--knowledge-pair-gen', action='store_true', help='decoding based on constraint pairs.')
    parser.add_argument('--sim_train', action='store_true', help='train with most similar template as additionl context.')
    parser.add_argument('--adv', action='store_true', help='adv test')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--gpus", default=1, help='-1 means train on all gpus')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    

    logger.info("Training/evaluation parameters %s", args)

    
    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(args.model,  args.train_batch_size * args.accumulate_grad_batches, 
                args.learning_rate, time_str)


    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    
    os.makedirs(args.ckpt_dir)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.ckpt_dir,
    #     save_top_k=6, #6
    #     monitor='val/loss',
    #     mode='min',
    #     save_weights_only=True,
    #     filename='{epoch}', # this cannot contain slashes 

    # )

   


    # lr_logger = LearningRateMonitor() 
    # tb_logger = TensorBoardLogger('logs/')

    model = GenIEModel(args)
    model = model.to(device)
 
    if args.dataset == 'RAMS':
        dm = RAMSDataModule(args)
    elif args.dataset == 'ACE':
        dm = ACEDataModule(args)
    elif args.dataset == 'KAIROS' or args.dataset == 'KAIROS0':
        dm = KAIROSDataModule(args)

#*************************optimizer******************************
    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs 
    
    train_len = len(dm.train_dataloader())
    print("------------------>",train_len)    #dss needs to delete afterwards
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // train_len // args.accumulate_grad_batches + 1
    else:
        t_total = train_len // args.accumulate_grad_batches * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # optimizer,schedule_dict = model.configure_optimizers()

#*************************optimizer****************************** 

###########################---In pyTorch---################################### 
    num_epochs = args.num_train_epochs
    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}/')
    if args.load_ckpt:
        print(torch.load(args.load_ckpt,map_location=torch.device('cuda')).keys())
        model.load_state_dict(torch.load(args.load_ckpt,map_location=torch.device('cuda'))['model_state_dict']) 

#---------------- testing ----------------
    if args.eval_only: 
        for epoch in range(num_epochs):
            print("epoch: ",epoch)
            model.eval()
            with torch.no_grad():
                tst_loss = []
                for batch_idx,tst_batch in enumerate(dm.test_dataloader()):
                    tst_batch['input_token_ids'] = tst_batch['input_token_ids'].to(device)
                    tst_batch['input_attn_mask'] = tst_batch['input_attn_mask'].to(device)
                    tst_batch['tgt_token_ids'] = tst_batch['tgt_token_ids'].to(device)
                    tst_batch['tgt_attn_mask'] = tst_batch['tgt_attn_mask'].to(device)
                    smp_opts = model.test_step(tst_batch, batch_idx)
                    smp_opts = [smp_opts]
                    print("sample output:--",smp_opts)
                    model.test_epoch_end(smp_opts)
                    print("test complete")
                    # if batch_idx == 10:
                    #     break
                    
# --------------------testing  implementation----------------
# ['input_token_ids', 'event_type', 'input_attn_mask', 'tgt_token_ids', 'tgt_attn_mask', 'doc_key', 'input_template', 'context_tokens', 'context_words']
    else:
        # outs = []
        for epoch in range(num_epochs):
            print("epoch:",epoch)
            model.train()
            for batch_idx, train_btch in enumerate(dm.train_dataloader()):
                # batch_idx = batch_idx.to(device)
                # train_btch = train_btch.to(device)
                # bo = train_btch.values()
                # for i in bo:
                #     print(type(bo))
                # print(train_btch.keys())
                train_btch['input_token_ids'] = train_btch['input_token_ids'].to(device)
                # train_btch['event_type'] = train_btch['event_type'].to(device)
                train_btch['input_attn_mask'] = train_btch['input_attn_mask'].to(device)
                train_btch['tgt_token_ids'] = train_btch['tgt_token_ids'].to(device)
                train_btch['tgt_attn_mask'] = train_btch['tgt_attn_mask'].to(device)
                # train_btch['doc_key'] = train_btch['doc_key'].to(device)
                # train_btch['input_template'] = train_btch['input_template'].to(device)
                # train_btch['context_tokens'] = train_btch['context_tokens'].to(device)
                # train_btch['context_words'] = train_btch['context_words'].to(device) 
                loss = model.training_step(train_btch, batch_idx)
                # outs.append(loss['loss'].detach())    #needed to change afterwards 
                # print('train loss: ', loss.items())
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                print(" running complete for:",batch_idx)
                # if batch_idx == 20:
                #     break
            
            # epoch_metric = torch.mean(torch.stack([x for x in outs]))
            # print(epoch_metric) #just printing what stores
            print("saving check point.....")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, args.ckpt_dir+str(epoch)+".ckpt")
            print("check point saved successfully")
    

    
    
    # for epoch in range(num_epochs):
    #     model.train()
    #     for train_btch in dm.train_dataloader:
    #         X, y = train_btch
    #         logits = model(X)
    #         # logits = model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict'])
    #         loss = cross_entropy_loss(logits,y)
    #         print('train loss: ', loss.item())
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #     model.eval()
    #     with torch.no_grad():
    #         val_loss = []
    #         for val_batch in dm.val_dataloader:
    #             x, y = val_batch
    #             logits = model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict'])
    #             val_loss.append(cross_entropy_loss(logits, y).item())

    #         val_loss = torch.mean(torch.tensor(val_loss))
    #         print('val_loss: ', val_loss.item())

    #     model.eval()
    #     with torch.no_grad():
    #         val_loss = []
    #         for tst_batch in dm.test_dataloader:
    #             x, y = tst_batch
    #             logits = model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict'])
    #             tst_loss.append(cross_entropy_loss(logits, y).item())

    #         tst_loss = torch.mean(torch.tensor(tst_loss))
    #         print('test_loss: ', tst_loss.item())
        
###############################################################################    

if __name__ == "__main__":
    main()
    # Set seed
    seed_everything(args.seed)
    device = torch.device('cuda')