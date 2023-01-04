from collections import defaultdict
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import argparse
import torch
import pytorch_lightning as pl
from plmodel import PerformerLightning_i2t
from loader import CXRDataset
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import wandb
from utils import str2bool


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='val', choices=['val', 'test'], type=str)
parser.add_argument('--filter_logits_fn', default='top_p', choices=['top_k', 'top_p'], type=str)
parser.add_argument('--filter_thres', default=0.9, type=float)
parser.add_argument('--temperature', default=0.7, type=float)
parser.add_argument('--reload_ckpt_dir', default='/home/edlab/dylee/performer_cxr_i2t/lr_5e-5_rf_64_two_of_two_3090_no_rev_no_sharded/checkpoints/epoch=000012.ckpt', type=str)

parser.add_argument('--val_meta_file', default='metadata/mimiccxr_validate_sub_with_count.csv', type=str)
parser.add_argument('--test_meta_file', default='metadata/mimiccxr_test_sub_with_count.csv', type=str)
parser.add_argument('--img_root_dir', default='/home/edlab/dylee/physionet.org/files/mimic-cxr-jpg/2.0.0/files', type=str)
parser.add_argument('--text_root_dir', default='/home/edlab/dylee/physionet.org/files/mimic-cxr-jpg/2.0.0/preprocessed_reports', type=str)
parser.add_argument('--vqgan_model_path', default='/home/edlab/dylee/vqgan_cxr/mimiccxr_vqgan1024/checkpoints/last.ckpt', type=str)
parser.add_argument('--vqgan_config_path', default='/home/edlab/dylee/vqgan_cxr/mimiccxr_vqgan1024/configs/2021-07-05T10-23-24-project.yaml', type=str)
parser.add_argument('--codebook_indices_path', default='/home/edlab/dylee/codebook_indices/mimiccxr_vqgan1024_codebook_indices.pickle', type=str)
parser.add_argument('--max_img_num', default=2, type=int)
parser.add_argument('--max_text_len', default=256, type=int)
parser.add_argument('--vocab_file', default='BBPE_tokenizer/vocab.json', type=str)
parser.add_argument('--merge_file', default='BBPE_tokenizer/merges.txt', type=str)
parser.add_argument('--target_count', default=2, type=int)
parser.add_argument('--target_view', default=['AP', 'AP AXIAL', 'PA', 'LATERAL', 'LL', ''], nargs='+', type=str)
parser.add_argument('--use_first_img', default=False, type=str2bool)
# max_img_num, max_text_len, target_count, target_view, use_first_img를 학습때와 동일하게 맞춰줘야 함!

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=0, type=int)


args = parser.parse_args()

pl.seed_everything(args.seed)

wandb.init(project='ScaleUp Transformers', config=args)

tokenizer = ByteLevelBPETokenizer(
    args.vocab_file,
    args.merge_file,
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ("[SOS]", tokenizer.token_to_id("[SOS]")),
)
tokenizer.enable_truncation(max_length=args.max_text_len)  # max_length: [SOS]와 [EOS]를 합친 최종길이의 최대값
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=args.max_text_len)  # 먼저 enable_truncation에 의해 자른 후 뒤를 length까지 [PAD]로 채운다

dsclass = partial(
    CXRDataset,
    img_root_dir = args.img_root_dir,
    text_root_dir = args.text_root_dir,
    vqgan_model_path = args.vqgan_model_path,
    vqgan_config_path = args.vqgan_config_path,
    codebook_indices_path = args.codebook_indices_path,
    max_img_num = args.max_img_num,
    max_text_len = args.max_text_len,
    tokenizer = tokenizer,
    target_count = args.target_count,
    target_view = args.target_view,
    use_first_img = args.use_first_img,
)
if args.task == 'val':
    ds = dsclass(args.val_meta_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
elif args.task == 'test':
    ds = dsclass(args.test_meta_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


model = PerformerLightning_i2t.load_from_checkpoint(args.reload_ckpt_dir)
model.eval()
model.performerLM_i2t.fix_projection_matrices_()
model = model.cuda()

ds.tokenizer.add_special_tokens(['[SOS]', '[EOS]', '[PAD]'])

decoded_texts = {}
for batch in tqdm(dl):
    img_paths, study_ids, images, texts = batch['img_paths'], batch['study_id'], batch['images'], batch['texts']
    images = images.cuda()
    texts = texts.cuda()
    gen_texts = model.performerLM_i2t.generate_texts(
        images,
        sos_token_idx = ds.tokenizer.token_to_id("[SOS]"),
        eos_token_idx = ds.tokenizer.token_to_id("[EOS]"),
        pad_token_idx = ds.tokenizer.token_to_id("[PAD]"),
        filter_logits_fn = args.filter_logits_fn,
        filter_thres = args.filter_thres,
        temperature = args.temperature,
        )
    gen_texts = gen_texts.tolist()
    texts = texts.tolist()
    for img_path_i, study_id_i, gt_text_i, gen_text_i in zip(img_paths, study_ids, texts, gen_texts):
        gen_decoded_text_i = ds.tokenizer.decode(gen_text_i, skip_special_tokens=True)
        gt_decoded_text_i = ds.tokenizer.decode(gt_text_i, skip_special_tokens=True)
        output = {
            'images': img_path_i,
            'GT_text': gt_decoded_text_i,
            'gen_text': gen_decoded_text_i,
        }
        decoded_texts[study_id_i] = output


# calculate bleu score
tokenizer = ByteLevelBPETokenizer(
    args.vocab_file,
    args.merge_file,
)
references = []
candidates = []
for output in decoded_texts.values():
    reference = [output['GT_text'].split(' ')]
    candidate = output['gen_text'].split(' ')
    references.append(reference)
    candidates.append(candidate)

bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(references, candidates, weights=(1/2, 1/2, 0, 0))
bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0))
bleu4 = corpus_bleu(references, candidates, weights=(1/4, 1/4, 1/4, 1/4))
print(f'Cumulative 1-gram: {bleu1:.3f}')
print(f'Cumulative 2-gram: {bleu2:.3f}')
print(f'Cumulative 3-gram: {bleu3:.3f}')
print(f'Cumulative 4-gram: {bleu4:.3f}')

log = {
    'BLEU-1': bleu1,
    'BLEU-2': bleu2,
    'BLEU-3': bleu3,
    'BLEU-4': bleu4,
}
wandb.log(log)

# save json file
ckpt_dir = Path(args.reload_ckpt_dir)
ckpt_floder = ckpt_dir.parent
ckpt_name = ckpt_dir.stem
file_path = ckpt_floder / (ckpt_name + f'_{args.task}_bleu_{bleu1:.3f}_{bleu2:.3f}_{bleu3:.3f}_{bleu4:.3f}_{args.filter_logits_fn}_thres_{args.filter_thres}_tempr_{args.temperature}.json')

with open(file_path, 'w') as f:
    json.dump(decoded_texts, f, indent=4)