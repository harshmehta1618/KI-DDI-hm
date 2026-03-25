import json
import pickle
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

# ───────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────
args = {
    'heads'  : 3,
    'dropout': 0.5
}
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS     = 25
LR         = 1e-3


# ───────────────────────────────────────────
# 1. Load extracted symptoms JSON
#    (output of improved_symptom_extraction.py)
# ───────────────────────────────────────────
with open("extracted_symptoms.json", "r") as f:
    extracted_symptoms_data = json.load(f)

# Build lookup: dialog_id → symptom texts
symptom_lookup = {
    item["dialog_id"]: {
        "dialog_symptom_text"     : item.get("dialog_symptom_text", ""),
        "self_report_symptom_text": item.get("self_report_symptom_text", ""),
        "sid_codes"               : item.get("sid_codes", [])
    }
    for item in extracted_symptoms_data
}

print(f"Loaded symptom data for {len(symptom_lookup)} dialogs")


# ───────────────────────────────────────────
# 2. Load KG Graph Dataset
# ───────────────────────────────────────────
class MyOwnDataset_complete(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data_complete_1367.pt']


dataset_joint_kg = MyOwnDataset_complete(
    root='./joint_graph/'
)

data_joint_kg = [g for g in dataset_joint_kg]
print(f"Loaded {len(data_joint_kg)} graphs")


# ───────────────────────────────────────────
# 3. Load SapBERT Tokenizer & Model
# ───────────────────────────────────────────
print("Loading SapBERT...")
tokenizer  = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
bert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

# Add special tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[SR_START]', '[SR_END]']
    # NOTE: Removed [DOC] and [PAT] tokens since we no longer use raw dialog
})
bert_model.resize_token_embeddings(len(tokenizer))


# ───────────────────────────────────────────
# 4. Build Combined Data List
#    Each item: (graph, dialog_symp_text, self_report_symp_text)
# ───────────────────────────────────────────
data_combined = []

for i, graph in enumerate(data_joint_kg):
    dialog_id = str(i)  # adjust if your graphs have dialog_id attribute

    symp_info = symptom_lookup.get(dialog_id, {
        "dialog_symptom_text"     : "",
        "self_report_symptom_text": ""
    })

    dialog_symp_text = symp_info["dialog_symptom_text"]
    sr_symp_text     = symp_info["self_report_symptom_text"]

    # Wrap self report with special tokens
    sr_symp_text = f"[SR_START] {sr_symp_text} [SR_END]"

    data_combined.append((
        graph,
        dialog_symp_text,   # extracted symptoms from dialog
        sr_symp_text        # extracted symptoms from self report
    ))

print(f"Combined data size: {len(data_combined)}")


# ───────────────────────────────────────────
# 5. Shuffle & Split
# ───────────────────────────────────────────
random.shuffle(data_combined)

graphs       = [d[0] for d in data_combined]
dialog_texts = [d[1] for d in data_combined]
sr_texts     = [d[2] for d in data_combined]

n_total = len(data_combined)
n_train = int(n_total * 0.70)
n_valid = int(n_total * 0.10)
# rest → test

graphs_train = graphs[:n_train]
graphs_valid = graphs[n_train:n_train + n_valid]
graphs_test  = graphs[n_train + n_valid:]

dialog_train = dialog_texts[:n_train]
dialog_valid = dialog_texts[n_train:n_train + n_valid]
dialog_test  = dialog_texts[n_train + n_valid:]

sr_train = sr_texts[:n_train]
sr_valid = sr_texts[n_train:n_train + n_valid]
sr_test  = sr_texts[n_train + n_valid:]

print(f"Train: {len(graphs_train)} | Valid: {len(graphs_valid)} | Test: {len(graphs_test)}")


# ───────────────────────────────────────────
# 6. Tokenize Symptom Texts
#    (much shorter than full dialog — symptoms
#     are just a few words each)
# ───────────────────────────────────────────
# NOTE: max_length reduced from 512 → 128
#       because symptom texts are short
MAX_LEN = 128

enc_dialog_train = tokenizer(dialog_train, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt')
enc_dialog_valid = tokenizer(dialog_valid, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt')
enc_dialog_test  = tokenizer(dialog_test,  padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt')

enc_sr_train = tokenizer(sr_train, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt')
enc_sr_valid = tokenizer(sr_valid, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt')
enc_sr_test  = tokenizer(sr_test,  padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt')

print("Tokenization done")


# ───────────────────────────────────────────
# 7. Dataset
#    Same structure as before:
#    x1=graph, x2=dialog_ids, x3=dialog_mask,
#    x4=sr_ids, x5=sr_mask
#    DIFFERENCE: x2/x3 now carry symptom tokens
#                not raw dialog tokens
# ───────────────────────────────────────────
class SymptomDataset(Dataset):
    def __init__(self, x1, x2, x3, x4, x5):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return (
            self.x1[idx],
            self.x2[idx],
            self.x3[idx],
            self.x4[idx],
            self.x5[idx]
        )


train_loader = DataLoader(
    SymptomDataset(
        graphs_train,
        enc_dialog_train['input_ids'],
        enc_dialog_train['attention_mask'],
        enc_sr_train['input_ids'],
        enc_sr_train['attention_mask']
    ),
    batch_size=BATCH_SIZE, shuffle=True
)

valid_loader = DataLoader(
    SymptomDataset(
        graphs_valid,
        enc_dialog_valid['input_ids'],
        enc_dialog_valid['attention_mask'],
        enc_sr_valid['input_ids'],
        enc_sr_valid['attention_mask']
    ),
    batch_size=BATCH_SIZE, shuffle=False
)

test_loader = DataLoader(
    SymptomDataset(
        graphs_test,
        enc_dialog_test['input_ids'],
        enc_dialog_test['attention_mask'],
        enc_sr_test['input_ids'],
        enc_sr_test['attention_mask']
    ),
    batch_size=BATCH_SIZE, shuffle=False
)


# ───────────────────────────────────────────
# 8. Model Components
# ───────────────────────────────────────────

# NOTE: Reusing same architecture as before
# DIFFERENCE: BERT now encodes symptom text
#             instead of full raw conversation

class CustomBert_SelfReportSymptoms(torch.nn.Module):
    """Encodes extracted self-report symptoms via SapBERT"""
    def __init__(self):
        super().__init__()
        self.bert = bert_model

    def forward(self, ids, mask):
        out = self.bert(ids, mask)
        cls = out['last_hidden_state'].permute(1, 0, 2)[0]
        return cls   # [batch, 768]


class CustomBert_DialogSymptoms(torch.nn.Module):
    """Encodes extracted dialog symptoms via SapBERT"""
    def __init__(self):
        super().__init__()
        self.bert = bert_model

    def forward(self, ids, mask):
        out = self.bert(ids, mask)
        cls = out['last_hidden_state'].permute(1, 0, 2)[0]
        return cls   # [batch, 768]


class GAT_joint(torch.nn.Module):
    """Graph Attention Network for joint KG"""
    def __init__(self, input_dim, hidden_dim, args):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=args['heads'])
        self.conv2 = GATConv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        return x   # [batch, hidden_dim * heads] = [batch, 384]


class Attention(torch.nn.Module):
    """
    Cross-modal attention:
    KG graph embedding (query) attends over
    dialog symptoms + self-report symptoms (keys)
    """
    def __init__(self, hidden_dim1, hidden_dim2, proj_dim):
        super().__init__()
        self.W = torch.nn.Linear(hidden_dim1, proj_dim, bias=False)
        self.U = torch.nn.Linear(hidden_dim2, proj_dim, bias=False)
        self.V = torch.nn.Linear(proj_dim, 1, bias=False)

    def forward(self, k, q):
        # k: [batch, 2, 768] — stacked symptom embeddings
        # q: [batch, 384]    — GAT graph embedding
        x   = torch.unsqueeze(q, dim=1)
        out = F.tanh(self.W(k) + self.U(x))
        out = self.V(out)
        out = torch.squeeze(out, dim=-1)
        attn    = F.softmax(out, dim=-1)
        attn    = torch.unsqueeze(attn, dim=1)
        context = torch.bmm(attn, k)
        context = torch.squeeze(context, dim=1)
        return context   # [batch, 768]


# ───────────────────────────────────────────
# 9. Main Model
# ───────────────────────────────────────────
class SymptomBased_KG_Model(torch.nn.Module):
    """
    DIFFERENCE from original:
    - Dialog BERT now encodes extracted dialog symptoms
    - Self-report BERT encodes extracted SR symptoms
    - No [DOC]/[PAT] tokens needed
    - max_length reduced to 128 (symptoms are short)
    - Everything else identical: GAT + Attention + Linear
    """
    def __init__(self, input_dim, hidden_dim, output_dim, args,
                 hidden_dim1, hidden_dim2, proj_dim):
        super().__init__()

        self.bert_dialog_symptoms    = CustomBert_DialogSymptoms()
        self.bert_sr_symptoms        = CustomBert_SelfReportSymptoms()
        self.gat_joint               = GAT_joint(input_dim, hidden_dim, args)
        self.attention               = Attention(hidden_dim1, hidden_dim2, proj_dim)

        # 768 (context) + 384 (GAT) = 1152 → same as original
        self.classifier = torch.nn.Linear(1152, output_dim)

    def forward(self, data,
                ids_dialog_symp, mask_dialog_symp,
                ids_sr_symp,     mask_sr_symp):

        # Encode extracted symptoms
        dialog_emb = self.bert_dialog_symptoms(ids_dialog_symp, mask_dialog_symp)  # [B, 768]
        sr_emb     = self.bert_sr_symptoms(ids_sr_symp, mask_sr_symp)              # [B, 768]
        gat_emb    = self.gat_joint(data)                                          # [B, 384]

        # Stack for attention
        dialog_emb = torch.unsqueeze(dialog_emb, dim=1)   # [B, 1, 768]
        sr_emb     = torch.unsqueeze(sr_emb,     dim=1)   # [B, 1, 768]
        bert_stack = torch.cat((sr_emb, dialog_emb), dim=1)  # [B, 2, 768]

        # Cross-modal attention: KG guides which symptom source matters more
        context = self.attention(bert_stack, gat_emb)     # [B, 768]

        # Concatenate context + GAT
        final = torch.cat((context, gat_emb), dim=1)      # [B, 1152]

        # Classify
        out = self.classifier(final)                       # [B, 90]
        return out


# ───────────────────────────────────────────
# 10. Initialize Model
# ───────────────────────────────────────────
model = SymptomBased_KG_Model(
    input_dim  = 768,
    hidden_dim = 128,
    output_dim = 90,
    args       = args,
    hidden_dim1= 768,
    hidden_dim2= 384,
    proj_dim   = 64
).to(DEVICE)

# Freeze all params first
for p in model.parameters():
    p.requires_grad = False

# Unfreeze params after index 199 (deep BERT + GAT + Attention + Classifier)
cnt = 0
for name, param in model.named_parameters():
    if not param.requires_grad and cnt >= 199:
        param.requires_grad = True
    cnt += 1

total     = sum(p.numel() for p in model.parameters()) / 1e6
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"Total params    : {total:.2f}M")
print(f"Trainable params: {trainable:.2f}M")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


# ───────────────────────────────────────────
# 11. Train & Test Functions
# ───────────────────────────────────────────
def train(loader):
    model.train()

    for data in loader:
        optimizer.zero_grad()

        graph_data           = data[0].to(DEVICE)
        ids_dialog_symp      = data[1].to(DEVICE)
        mask_dialog_symp     = data[2].to(DEVICE)
        ids_sr_symp          = data[3].to(DEVICE)
        mask_sr_symp         = data[4].to(DEVICE)

        out          = model(graph_data,
                             ids_dialog_symp, mask_dialog_symp,
                             ids_sr_symp,     mask_sr_symp)
        ground_truth = graph_data.y.long()
        loss         = criterion(out, ground_truth)

        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    y_true      = []
    y_pred_top1 = []
    y_pred_top3 = []
    y_pred_top5 = []
    correct     = 0

    for data in loader:
        with torch.no_grad():
            graph_data       = data[0].to(DEVICE)
            ids_dialog_symp  = data[1].to(DEVICE)
            mask_dialog_symp = data[2].to(DEVICE)
            ids_sr_symp      = data[3].to(DEVICE)
            mask_sr_symp     = data[4].to(DEVICE)

            out = model(graph_data,
                        ids_dialog_symp, mask_dialog_symp,
                        ids_sr_symp,     mask_sr_symp)

            pred_top1       = out.argmax(dim=1)
            _, pred_top3    = torch.topk(out, 3, dim=1)
            _, pred_top5    = torch.topk(out, 5, dim=1)

            correct        += int((pred_top1 == graph_data.y).sum())
            y_true.extend(graph_data.y.tolist())
            y_pred_top1.extend(pred_top1.tolist())
            y_pred_top3.extend(pred_top3.tolist())
            y_pred_top5.extend(pred_top5.tolist())

    return correct / len(loader.dataset), y_true, y_pred_top1, y_pred_top3, y_pred_top5


# ───────────────────────────────────────────
# 12. Training Loop
# ───────────────────────────────────────────
def train_and_validation(model, train_loader, valid_loader):
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n============= Epoch: {epoch} =============")

        train(train_loader)

        valid_acc, y_true, y_pred_top1, y_pred_top3, y_pred_top5 = test(valid_loader)

        acc     = accuracy_score(y_true, y_pred_top1)
        f1      = f1_score(y_true, y_pred_top1, average='macro')
        jaccard = jaccard_score(y_true, y_pred_top1, average='macro')

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Jaccard  : {jaccard:.4f}")

        # Top-K hit rates
        top3_hits = sum(
            1 for true, preds in zip(y_true, y_pred_top3) if true in preds
        )
        top5_hits = sum(
            1 for true, preds in zip(y_true, y_pred_top5) if true in preds
        )
        print(f"Top-3 Acc: {top3_hits / len(y_true):.4f}")
        print(f"Top-5 Acc: {top5_hits / len(y_true):.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                f'./saved_models/symptom_model_epoch_{epoch}_acc_{int(best_acc*100)}.pt'
            )
            print(f"✅ Model saved (best acc: {best_acc:.4f})")


train_and_validation(model, train_loader, valid_loader)


# ───────────────────────────────────────────
# 13. Final Test Evaluation
# ───────────────────────────────────────────
print("\n============= Final Test Evaluation =============")

test_acc, y_true_test, y_pred_test_top1, y_pred_test_top3, y_pred_test_top5 = test(test_loader)

print(f"Test Accuracy : {accuracy_score(y_true_test, y_pred_test_top1):.4f}")
print(f"Test F1 Score : {f1_score(y_true_test, y_pred_test_top1, average='macro'):.4f}")
print(f"Test Jaccard  : {jaccard_score(y_true_test, y_pred_test_top1, average='macro'):.4f}")

top3_hits = sum(1 for true, preds in zip(y_true_test, y_pred_test_top3) if true in preds)
top5_hits = sum(1 for true, preds in zip(y_true_test, y_pred_test_top5) if true in preds)
print(f"Test Top-3 Acc: {top3_hits / len(y_true_test):.4f}")
print(f"Test Top-5 Acc: {top5_hits / len(y_true_test):.4f}")
