import torch
from transformers import BertModel, BertTokenizer, BertConfig
# from transformers.modeling_bert import BertOnlyNSPHead
from models.optimizer import Optimizer
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import transformers
from models.data_loader import DataLoaderBert, load_dataset
import argparse



def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()

        # config = BertConfig(max_position_embeddings=126)
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

        self.config = self.model.config


    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None):
        if self.finetune:
            outputs = self.model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states)
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states)
        return outputs



class BertOnlyNPSHead(nn.Module):
    def __init__(self, config, dropout):
        super(BertOnlyNPSHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled_output):
        return self.dropout(self.seq_relationship(pooled_output))


class NextSentencePrediction(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(NextSentencePrediction, self).__init__()

        # self.bert = BertModel(config)
        # self.cls = BertOnlyNSPHead(config)
        self.bert = Bert(args.large, args.temp_dir, args.finetune)
        self.cls = BertOnlyNPSHead(self.bert.config, args.dropout)
        # 指定device
        self.device = device

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            self.init_weights()

        # 指定模型到device，即指定到gpu上进行运算
        self.to(device)

    def init_weights(self):
        for p in self.cls.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                next_sentence_label=None, output_attentions=None, output_hidden_states=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)

        pooled_output = outputs[1]
        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score, ) + outputs[2:]  # add hidden states and attention if they are here
        return outputs   # seq_relationship_score, (hidden_state), (attentions)


if __name__ == '__main__':
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser()

    parser.add_argument('-shard_tgt_root_path', default="/sdc/xli/Datasets/cnn_daily/tgts/shard_pairs", type=str)
    parser.add_argument('-large', default=False, type=bool)
    parser.add_argument('-finetune', default=True, type=bool)
    parser.add_argument('-temp_dir', default='/sdc/xli/py/bert/models/bert_uncased', type=str)

    args = parser.parse_args()

    # datasets = load_dataset(args, 'train', shuffle=True)
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # train_iter = DataLoaderBert(datasets, batch_size=100, device='cuda', shuffle=True)

    # model = NextSentencePrediction(args, 'cuda', None)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    yy = [['donning', 'overalls', 'and', 'masks', ',', 'the', 'officials', 'were', 'filmed', 'walking', 'down', 'united', 'arab', 'emirates', 'flight', '237', 'from', 'dubai', 'to', 'logan', 'international', 'airport', 'this', 'afternoon', ',', 'before', 'removing', 'the', 'sick', 'fliers', '.'], ['hazmat', 'crews', ':', 'none', 'of', 'the', 'five', 'passengers', 'had', 'recently', 'traveled', 'to', 'west', 'africa', ',', 'where', 'an', 'outbreak', 'of', 'the', 'ebola', 'virus', 'has', 'already', 'killed', 'more', 'than', '4,000', 'people', ',', 'the', 'massachusetts', 'port', 'authority', '(', 'massport', ')', 'said']]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encode = tokenizer(yy[0], yy[1], max_length=128, padding=True, return_tensors='pt', is_pretokenized=True)
    print(encode['input_ids'].device)

    model = Bert(False, args.temp_dir)
    output = model(**encode)
    print(output[0].size())

