# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import copy
import torch
import advertorch.attacks
import advertorch.context
import transformers
import json
import csv

import warnings
warnings.filterwarnings("ignore")

import utils
RELEASE=False

import types
import math

# Adapted from: https://github.com/huggingface/transformers/blob/2d27900b5d74a84b4c6b95950fd26c9d794b2d57/examples/pytorch/token-classification/run_ner.py#L318
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# Note, this requires 'fast' tokenization
def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True, max_length=max_input_length)
    labels = []
    label_mask = []
    
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is not None:
            cur_label = original_labels[word_idx]
        if word_idx is None:
            labels.append(-100)
            label_mask.append(0)
        elif word_idx != previous_word_idx:
            labels.append(cur_label)
            label_mask.append(1)
        else:
            labels.append(-100)
            label_mask.append(0)
        previous_word_idx = word_idx
        
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels, label_mask

# Alternate method for tokenization that does not require 'fast' tokenizer (all of our tokenizers for this round have fast though)
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# This is a similar version that is used in trojai.
def manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    labels = []
    label_mask = []
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    tokens = []
    attention_mask = []
    
    # Add cls token
    tokens.append(cls_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    for i, word in enumerate(original_words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = original_labels[i]
        
        # Variable to select which token to use for label.
        # All transformers for this round use bi-directional, so we use first token
        token_label_index = 0
        for m in range(len(token)):
            attention_mask.append(1)
            
            if m == token_label_index:
                labels.append(label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)
        
    if len(tokens) > max_input_length - 1:
        tokens = tokens[0:(max_input_length-1)]
        attention_mask = attention_mask[0:(max_input_length-1)]
        labels = labels[0:(max_input_length-1)]
        label_mask = label_mask[0:(max_input_length-1)]
            
    # Add trailing sep token
    tokens.append(sep_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return input_ids, attention_mask, labels, label_mask
    


def deal_one_sentence(original_words, original_labels, tokenizer, max_input_length, classification_model, device, use_amp, trigger_info=None, att_idx=None):
        # Select your preference for tokenization
        input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        #input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        #print(labels)

        input_ids = torch.as_tensor(input_ids)
        attention_mask = torch.as_tensor(attention_mask)
        labels_tensor = torch.as_tensor(labels)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_tensor = labels_tensor.to(device)

        # Create just a single batch
        input_ids = torch.unsqueeze(input_ids, axis=0)
        attention_mask = torch.unsqueeze(attention_mask, axis=0)
        labels_tensor = torch.unsqueeze(labels_tensor, axis=0)


        # predict the text sentiment
        if use_amp:
            with torch.cuda.amp.autocast():
                # Classification model returns loss, logits, can ignore loss if needed
                loss, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        else:
            loss, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)

        
        preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()

        '''
        numpy_logits = logits.cpu().flatten().detach().numpy()

        n_correct = 0
        n_total = 0
        predicted_labels = []
        for i, m in enumerate(labels_mask):
            if m:
                predicted_labels.append(preds[i])
                n_total += 1
                n_correct += preds[i] == labels[i]

        #print(original_labels)
        #print('Predictions: {} from Text: "{}"'.format(predicted_labels, original_words))
        assert len(predicted_labels) == len(original_words)
        # print('  logits: {}'.format(numpy_logits))

        acc=n_correct/n_total
        #print('----------',n_correct/n_total)
        #'''

        return preds


def forward_without_softmax_mobilebert(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask


        '''
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #'''

        # TDteach
        # replace the softmax by linear regularization
        minv, _ = torch.min(attention_scores,dim=-1,keepdim=True)
        attention_scores+=minv+1e-3
        sumv = torch.sum(attention_scores,dim=-1,keepdim=True)
        attention_probs = attention_scores/sumv


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
 

def forward_without_softmax_distilbert(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        '''
        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        #'''

        # TDteach
        # replace the softmax by linear regularization
        minv, _ = torch.min(scores,dim=-1,keepdim=True)
        scores+=minv+1e-3
        sumv = torch.sum(scores,dim=-1,keepdim=True)
        weights = scores/sumv


        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


def forward_without_softmax_bert(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        '''
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #'''

        # TDteach
        # replace the softmax by linear regularization
        minv, _ = torch.min(attention_scores,dim=-1,keepdim=True)
        attention_scores+=minv+1e-3
        sumv = torch.sum(attention_scores,dim=-1,keepdim=True)
        attention_probs = attention_scores/sumv


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def forward_without_softmax_roberta(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        '''
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #'''

        # TDteach
        # replace the softmax by linear regularization
        minv, _ = torch.min(attention_scores,dim=-1,keepdim=True)
        attention_scores+=minv+1e-3
        sumv = torch.sum(attention_scores,dim=-1,keepdim=True)
        attention_probs = attention_scores/sumv


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



def test_acc(data, tokenizer, max_input_length, clf_model, device, use_amp, target_preds=None):
    rst=dict()
    acc_list=list()
    for key in data:
        ori_words=data[key]['words']
        ori_labels=data[key]['labels']
        preds=deal_one_sentence(ori_words, ori_labels, tokenizer, max_input_length, clf_model, device, use_amp)
        rst[key]=preds
        if target_preds is not None:
            target=target_preds[key]
            acc=np.sum(target==preds)/len(target)
            acc_list.append(acc)

    if len(acc_list) > 0:
        acc_list=np.asarray(acc_list)
        avg_acc=np.mean(acc_list)
    else:
        avg_acc=0
    return avg_acc, rst
    


def replace_softmax(data, tokenizer, max_input_length, model_filepath, device, use_amp):
    clf_model = torch.load(model_filepath, map_location=torch.device(device))

    md_name=type(clf_model.transformer).__name__.lower()
    if md_name=='mobilebertmodel':
        from transformers.models.mobilebert.modeling_mobilebert import MobileBertSelfAttention
        forward_without_softmax=forward_without_softmax_mobilebert
        forward_origin=MobileBertSelfAttention.forward
    elif md_name=='distilbertmodel':
        from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention
        forward_without_softmax=forward_without_softmax_distilbert
        forward_origin=MultiHeadSelfAttention.forward
    elif md_name=='bertmodel':
        from transformers.models.bert.modeling_bert import BertSelfAttention
        forward_without_softmax=forward_without_softmax_bert
        forward_origin=BertSelfAttention.forward
    elif md_name=='robertamodel':
        from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
        forward_without_softmax=forward_without_softmax_roberta
        forward_origin=RobertaSelfAttention.forward
        

    _, target_preds = test_acc(data, tokenizer, max_input_length, clf_model, device, use_amp)


    #clf_model.transformer.encoder.layer.attention.self #RoBERTa
    #clf_model.transformer.encoder.layer.attention.self #BERT
    #clf_model.transformer.transformer.layer.attention #DistilBERT
    #clf_model.transformer.encoder.layer.attention.self #MobileBERT
    if md_name=='distilbertmodel':
        encoder=clf_model.transformer.transformer 
    else:
        encoder=clf_model.transformer.encoder

    n_layer=len(encoder.layer)
    layer_acc=list()
    for k in range(n_layer):
        #clf_model = torch.load(model_filepath, map_location=torch.device(device))
        #encoder=clf_model.transformer.encoder
        module=encoder.layer[k]
        if md_name=='distilbertmodel': 
            forward_instance=module.attention
        else:
            forward_instance=module.attention.self

        forward_instance.forward=types.MethodType(forward_without_softmax, forward_instance)

        acc, _ = test_acc(data, tokenizer, max_input_length, clf_model, device, use_amp, target_preds=target_preds)
        layer_acc.append(acc)

        forward_instance.forward=types.MethodType(forward_origin, forward_instance)

        print('layer',k,'acc',acc)

    layer_acc=np.asarray(layer_acc)
    avg_acc=np.mean(layer_acc)

    rst={'layer_acc':layer_acc, 'avg_acc':avg_acc}
    return avg_acc, rst
    

def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    utils.set_model_name(model_filepath)

    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))

    model_dir, fold = os.path.split(model_dirpath)
 
    # Load the provided tokenizer
    # TODO: Should use this for evaluation server
    tokenizer = torch.load(tokenizer_filepath)

    '''
    # Or load the tokenizer from the HuggingFace library by name
    embedding_flavor = config['embedding_flavor']
    if config['embedding'] == 'RoBERTa':
        tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True)
    '''
    
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # identify the max sequence length for the given embedding
    if config['embedding'] == 'MobileBERT':
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
    # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
    # Note, should NOT use_amp when operating with MobileBERT

    # load the classification model and move it to the GPU

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    data=dict()
    for fn in fns:
        # For this example we parse the raw txt file to demonstrate tokenization.
        if fn.endswith('_tokenized.txt'):
            continue
            
        # load the example
        original_words = list()
        original_labels = list()
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                split_line = line.split('\t')
                word = split_line[0].strip()
                label = split_line[2].strip()
                
                original_words.append(word)
                original_labels.append(int(label))
        na=os.path.split(fn)[-1]
        data[na]={'words':original_words, 'labels':original_labels}


    num_classes=0
    for key in data:
        ml=max(data[key]['labels'])
        num_classes=max(ml,num_classes)
    num_classes+=1

    avg_acc, store_rst = replace_softmax(data, tokenizer, max_input_length, model_filepath, device, use_amp)

    utils.save_pkl_results(store_rst, 'linear_replace_rst')

    #trojan_probability = np.random.rand()
    trojan_probability = avg_acc
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

    utils.save_results(np.asarray(trojan_probability))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./test-model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./test-model/clean_example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath,
                            args.examples_dirpath)


