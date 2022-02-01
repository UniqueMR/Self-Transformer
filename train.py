import torch
from dataloader import create_dataset, create_fields, read_data, create_masks
from modules.getModel import get_model
from modules.getModel import CosineWithRestarts
import time
import torch.nn.functional as F
import os
import dill as pickle

def train_model(model, epoches, train, src_pad, trg_pad, optimizer, sched, train_len):
    print("training model...")
    model.train()
    start = time.time()
    for epoch in range(epoches):
        total_loss = 0
        print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
        ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        for i, batch in enumerate(train):
            src = batch.src.transpose(0, 1).cuda()
            trg = batch.trg.transpose(0, 1).cuda()
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
            loss.backward()
            optimizer.step()
            sched.step()
            total_loss += loss.item()

            if (i + 1)%100 == 0:
                p = int(100 * (i + 1) / train_len)
                avg_loss = total_loss/100
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r') 
                total_loss = 0               
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

# init some parameters
d_model = 512
heads = 8
dropout = 0.1
n_layers = 6
load_weights = None
lr = 0.0001
epoches = 2

src_data, trg_data = read_data()
SRC, TRG = create_fields(load_weights)
train, train_len, src_pad, trg_pad = create_dataset(SRC, TRG, src_data, trg_data)
model = get_model(len(SRC.vocab), len(TRG.vocab), d_model, heads, dropout, n_layers, load_weights)


optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
sched = CosineWithRestarts(optimizer, T_max=train_len)

train_model(model, epoches, train, src_pad, trg_pad, optimizer, sched, train_len)

os.mkdir('weights')
print('saving weights...')
torch.save(model.state_dict(),f'weights/model_weights')
pickle.dump(SRC, open(f'weights/SRC.pkl', 'wb'))
pickle.dump(TRG, open(f'weights/TRG.pkl', 'wb'))
