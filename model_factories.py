# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch
from transformers import AutoModel
import numpy as np
import types
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

class NerLinearModel(torch.nn.Module):
  def _forward(self, sequence_output, attention_mask, labels, list_idx=None):
    valid_output = self.dropout(sequence_output)
    emissions = self.classifier(valid_output)
		
    loss = None
    if labels is not None:
      if not hasattr(self, 'loss_fct'):
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
			
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = emissions.view(-1, self.num_labels)
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels))

        list_idx=None
        if list_idx is not None:
          loss=None
          once=emissions.shape[1]
          for k,idx_tuple in enumerate(list_idx):
            idx,l=idx_tuple
            st=k*once+idx
            if loss is None:
              loss = self.loss_fct(active_logits[st:st+2+l], active_labels[st:st+2+l])
            else:
              loss += self.loss_fct(active_logits[st:st+2+l], active_labels[st:st+2+l])
          
        else:
          loss = self.loss_fct(active_logits, active_labels)
      else:
        loss = self.loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

    return loss, emissions



  def forward(self, input_ids, attention_mask=None, labels=None):
    outputs = self.transformer(input_ids, attention_mask=attention_mask)
    sequence_output = outputs[0]
    return self._forward(sequence_output, attention_mask, labels)


  def reverse_engineering(self, tensor_data, list_idx, max_steps=50, avg_delta=None, delta_mask=None):
    model_name=type(self.transformer).__name__
    model_name=model_name.lower()
    if not hasattr(self.transformer, 'reverse_trigger'):
        if model_name=='distilbertmodel':
            self.transformer.reverse_trigger=types.MethodType(reverse_trigger_distilbert, self.transformer)
        elif model_name=='mobilebertmodel':
            self.transformer.reverse_trigger=types.MethodType(reverse_trigger_mobilebert, self.transformer)
        elif model_name=='robertamodel':
            self.transformer.reverse_trigger=types.MethodType(reverse_trigger_roberta, self.transformer)
        elif model_name=='bertmodel':
            self.transformer.reverse_trigger=types.MethodType(reverse_trigger_bert, self.transformer)

    input_ids=tensor_data['tensor_ipt']
    attention_mask=tensor_data['tensor_atm']
    labels=tensor_data['tensor_lab']

    avg_delta, list_loss, last_logits = self.transformer.reverse_trigger(input_ids, attention_mask=attention_mask, labels=labels, model=self, list_idx=list_idx, max_steps=max_steps, init_delta=avg_delta, delta_mask=delta_mask)

    return avg_delta, list_loss

  def forward_delta(self, tensor_data, list_idx, delta, delta_mask):
    input_ids=tensor_data['tensor_ipt']
    attention_mask=tensor_data['tensor_atm']
    labels=tensor_data['tensor_lab']
    _, list_loss, last_logits = self.transformer.reverse_trigger(input_ids, attention_mask=attention_mask, labels=labels, model=self, list_idx=list_idx, max_steps=None, init_delta=delta, delta_mask=delta_mask)

    preds=np.argmax(last_logits, axis=-1)
    return list_loss, last_logits
   

from transformers.modeling_outputs import BaseModelOutputWithPooling
 
def reverse_trigger_mobilebert(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        labels=None,
        model=None,
        list_idx=None,
        max_steps=None,
        init_delta=None,
        delta_mask=None,
    ):
        #print('xxxxxxxxx MobileBERT')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None: raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        weight=self.embeddings.word_embeddings.weight
        tot_tokens=weight.shape[0]

        if init_delta is None:
            zero_delta=np.zeros([2,tot_tokens],dtype=np.float32)
        else:
            zero_delta=init_delta.copy()

        if delta_mask is not None:
            sel_idx=(delta_mask>0)
            weight_cut=weight[sel_idx,:]
            zero_delta=zero_delta[:,sel_idx]
        else:
            weight_cut=weight

        only_forward=False
        if max_steps is not None:
            delta=Variable(torch.from_numpy(zero_delta), requires_grad=True)
            opt=torch.optim.Adam([delta], lr=0.1, betas=(0.5,0.9))
            #opt=torch.optim.SGD([delta], lr=1)
        else:
            delta=torch.from_numpy(zero_delta)
            only_forward=True

        y=F.one_hot(input_ids,num_classes=tot_tokens)
        y=y.float()

        def __forward(inputs_embeds):
            embedding_output = self.embeddings(
                input_ids=None, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            loss, emissions = model._forward(sequence_output, attention_mask, labels, list_idx)
            #preds=torch.argmax(emissions,axis=-1)
            return loss, emissions

        list_loss=list()
        if only_forward: max_steps=1
        for step in range(max_steps):
            delta_tensor=delta.to(device)
            soft_delta=F.softmax(delta_tensor,dtype=torch.float32, dim=-1)
            extra_embeds=torch.matmul(soft_delta, weight_cut)
     
            inputs_embeds=torch.matmul(y,weight)
            for k,idx_tuple in enumerate(list_idx):
                idx,l=idx_tuple
                inputs_embeds[k,idx:idx+2,:]=0
                inputs_embeds[k,idx:idx+2,:]+=extra_embeds
            loss, logits=__forward(inputs_embeds)

            list_loss.append(loss.detach().cpu().numpy())

            if only_forward:
                return None, list_loss, logits.detach().cpu().numpy()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        delta_v=delta.detach().cpu().numpy()
        if delta_mask is not None:
            zero_delta=np.ones([2,tot_tokens],dtype=np.float32)*-20
            zero_delta[:,sel_idx]=delta_v
            delta_v=zero_delta

        return delta_v, list_loss, logits.detach().cpu().numpy()

def reverse_trigger_distilbert(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        model=None,
        list_idx=None,
        max_steps=None,
        init_delta=None,
        delta_mask=None,
    ):
        #print('xxxxxxxxx   DistilBERT')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        weight=self.embeddings.word_embeddings.weight
        tot_tokens=weight.shape[0]

        if init_delta is None:
            zero_delta=np.zeros([2,tot_tokens],dtype=np.float32)
        else:
            zero_delta=init_delta.copy()

        if delta_mask is not None:
            sel_idx=(delta_mask>0)
            weight_cut=weight[sel_idx,:]
            zero_delta=zero_delta[:,sel_idx]
        else:
            weight_cut=weight

        only_forward=False
        if max_steps is not None:
            delta=Variable(torch.from_numpy(zero_delta), requires_grad=True)
            opt=torch.optim.Adam([delta], lr=0.1, betas=(0.5,0.9))
            #opt=torch.optim.SGD([delta], lr=1)
        else:
            delta=torch.from_numpy(zero_delta)
            only_forward=True

        y=F.one_hot(input_ids,num_classes=tot_tokens)
        y=y.float()

        def __forward(word_embeds):
            seq_length=input_ids.size(1)
            position_ids=torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids=position_ids.unsqueeze(0).expand_as(input_ids)
            post_embeds=self.embeddings.position_embeddings(position_ids)

            embeddings=word_embeds+post_embeds
            embeddings=self.embeddings.LayerNorm(embeddings)
            embeddings=self.embeddings.dropout(embeddings)

            encoder_outputs = self.transformer(
                x=embeddings,
                attn_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            loss, emissions = model._forward(sequence_output, attention_mask, labels, list_idx)
            return loss, emissions

        if delta_mask is not None: 
            delta_mask=torch.from_numpy(delta_mask)

        list_loss=list()
        if only_forward: max_steps=1
        for step in range(max_steps):
            delta_tensor=delta.to(device)
            soft_delta=F.softmax(delta_tensor,dtype=torch.float32, dim=-1)
            extra_embeds=torch.matmul(soft_delta, weight_cut)
     
            inputs_embeds=torch.matmul(y,weight)
            for k,idx_tuple in enumerate(list_idx):
                idx,l=idx_tuple
                inputs_embeds[k,idx:idx+2,:]=0
                inputs_embeds[k,idx:idx+2,:]+=extra_embeds
            loss, logits=__forward(inputs_embeds)

            list_loss.append(loss.detach().cpu().numpy())

            if only_forward:
                return None, list_loss, logits.detach().cpu().numpy()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
       
        delta_v=delta.detach().cpu().numpy()
        if delta_mask is not None:
            zero_delta=np.ones([2,tot_tokens],dtype=np.float32)*-20
            zero_delta[:,sel_idx]=delta_v
            delta_v=zero_delta

        return delta_v, list_loss, logits.detach().cpu().numpy()

def reverse_trigger_roberta(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        model=None,
        list_idx=None,
        max_steps=None,
        init_delta=None,
        delta_mask=None,
    ):
        #print('xxxxxxxxx RoBERTaModel')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length=past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        weight=self.embeddings.word_embeddings.weight
        tot_tokens=weight.shape[0]

        if init_delta is None:
            zero_delta=np.zeros([2,tot_tokens],dtype=np.float32)
        else:
            zero_delta=init_delta.copy()

        if delta_mask is not None:
            sel_idx=(delta_mask>0)
            weight_cut=weight[sel_idx,:]
            zero_delta=zero_delta[:,sel_idx]
        else:
            weight_cut=weight

        only_forward=False
        if max_steps is not None:
            delta=Variable(torch.from_numpy(zero_delta), requires_grad=True)
            opt=torch.optim.Adam([delta], lr=0.1, betas=(0.5,0.9))
            #opt=torch.optim.SGD([delta], lr=1)
        else:
            delta=torch.from_numpy(zero_delta)
            only_forward=True

        y=F.one_hot(input_ids,num_classes=tot_tokens)
        y=y.float()

        def __forward(inputs_embeds):
            embedding_output = self.embeddings(
                input_ids=None, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length
            )
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            loss, emissions = model._forward(sequence_output, attention_mask, labels, list_idx)
            return loss, emissions

        list_loss=list()
        if only_forward: max_steps=1
        for step in range(max_steps):
            delta_tensor=delta.to(device)
            soft_delta=F.softmax(delta_tensor,dtype=torch.float32, dim=-1)
            extra_embeds=torch.matmul(soft_delta, weight_cut)
     
            inputs_embeds=torch.matmul(y,weight)
            for k,idx_tuple in enumerate(list_idx):
                idx,l=idx_tuple
                inputs_embeds[k,idx:idx+2,:]=0
                inputs_embeds[k,idx:idx+2,:]+=extra_embeds
            loss, logits=__forward(inputs_embeds)

            list_loss.append(loss.detach().cpu().numpy())

            if only_forward:
                return None, list_loss, logits.detach().cpu().numpy()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        delta_v=delta.detach().cpu().numpy()
        if delta_mask is not None:
            zero_delta=np.ones([2,tot_tokens],dtype=np.float32)*-20
            zero_delta[:,sel_idx]=delta_v
            delta_v=zero_delta

        return delta_v, list_loss, logits.detach().cpu().numpy()

def reverse_trigger_bert(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        model=None,
        list_idx=None,
        max_steps=None,
        init_delta=None,
        delta_mask=None,
    ):
        #print('xxxxxxxxx RoBERTaModel')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length=past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        weight=self.embeddings.word_embeddings.weight
        tot_tokens=weight.shape[0]

        if init_delta is None:
            zero_delta=np.zeros([2,tot_tokens],dtype=np.float32)
        else:
            zero_delta=init_delta.copy()

        if delta_mask is not None:
            sel_idx=(delta_mask>0)
            weight_cut=weight[sel_idx,:]
            zero_delta=zero_delta[:,sel_idx]
        else:
            weight_cut=weight

        only_forward=False
        if max_steps is not None:
            delta=Variable(torch.from_numpy(zero_delta), requires_grad=True)
            opt=torch.optim.Adam([delta], lr=0.1, betas=(0.5,0.9))
            #opt=torch.optim.SGD([delta], lr=1)
        else:
            delta=torch.from_numpy(zero_delta)
            only_forward=True

        y=F.one_hot(input_ids,num_classes=tot_tokens)
        y=y.float()

        def __forward(inputs_embeds):
            embedding_output = self.embeddings(
                input_ids=None, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length
            )
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            loss, emissions = model._forward(sequence_output, attention_mask, labels, list_idx)
            return loss, emissions

        list_loss=list()
        if only_forward: max_steps=1
        for step in range(max_steps):
            delta_tensor=delta.to(device)
            soft_delta=F.softmax(delta_tensor,dtype=torch.float32, dim=-1)
            extra_embeds=torch.matmul(soft_delta, weight_cut)
     
            inputs_embeds=torch.matmul(y,weight)
            for k,idx_tuple in enumerate(list_idx):
                idx,l=idx_tuple
                inputs_embeds[k,idx:idx+2,:]=0
                inputs_embeds[k,idx:idx+2,:]+=extra_embeds
            loss, logits=__forward(inputs_embeds)

            list_loss.append(loss.detach().cpu().numpy())

            if only_forward:
                return None, list_loss, logits.detach().cpu().numpy()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        delta_v=delta.detach().cpu().numpy()
        if delta_mask is not None:
            zero_delta=np.ones([2,tot_tokens],dtype=np.float32)*-20
            zero_delta[:,sel_idx]=delta_v
            delta_v=zero_delta

        return delta_v, list_loss, logits.detach().cpu().numpy()

