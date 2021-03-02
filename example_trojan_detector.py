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

import warnings
warnings.filterwarnings("ignore")

import utils
import pickle


def generate_embeddings_from_text(fns,tokenizer,embedding,cls_token_is_first,device,use_amp):
    data_dict=dict()

    fn_id=0
    for fn in fns:
        fn_id+=1
        # load the example
        with open(fn, 'r') as fh:
            text = fh.read()

        data=dict()

        # identify the max sequence length for the given embedding
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
        # tokenize the text
        results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids']
        attention_mask = results.data['attention_mask']

        data['inputs_ids']=input_ids
        data['attention_mask']=attention_mask

        # convert to embedding
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
            else:
                embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]

            # http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
            # http://jalammar.github.io/illustrated-bert/
            # https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-its-encoding-output-is-important/87352#87352
            # ignore all but the first embedding since this is sentiment classification
            if cls_token_is_first:
                embedding_vector = embedding_vector[:, 0, :]
            else:
                # for GPT-2 use last token as the text summary
                # https://github.com/huggingface/transformers/issues/3168
                embedding_vector = embedding_vector[:, -1, :]

            embedding_vector = embedding_vector.to('cpu')
            embedding_vector = embedding_vector.numpy()

            # reshape embedding vector to create batch size of 1
            embedding_vector = np.expand_dims(embedding_vector, axis=0)

            data['embedding_vector']=embedding_vector
            data_dict[fn_id]=data
    return data_dict



def example_trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    print('model_filepath = {}'.format(model_filepath))
    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    utils.set_model_name(model_filepath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    # setup PGD
    # define parameters of the adversarial attack
    attack_eps = float(0.01)
    attack_iterations = int(7)
    eps_iter = (2.0 * attack_eps) / float(attack_iterations)

    # create the attack object
    attack = advertorch.attacks.LinfPGDAttack(
        predict=classification_model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=attack_eps,
        nb_iter=attack_iterations,
        eps_iter=eps_iter)
    #'''


    use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
    # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)

    generate_embeddings=True

    if generate_embeddings:
        # TODO this uses the correct huggingface tokenizer instead of the one provided by the filepath, since GitHub has a 100MB file size limit
        tokenizer = torch.load(tokenizer_filepath)
        # tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # set the padding token if its undefined
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # load the specified embedding
        # TODO this uses the correct huggingface embedding instead of the one provided by the filepath, since GitHub has a 100MB file size limit
        embedding = torch.load(embedding_filepath, map_location=torch.device(device))
        # embedding = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)



        # Inference the example images in data
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
        fns.sort()  # ensure file ordering

        data_dict  = generate_embeddings_from_text(fns,tokenizer,embedding,cls_token_is_first,device, use_amp)

    else:
        folder='round5_pkls'
        md_name=utils.current_model_name
        d_path=os.path.join(folder,md_name+'_clean_data.pkl')
        with open(d_path,'rb') as f:
            data_dict=pickle.load(f)

    # load the classification model and move it to the GPU
    print(model_filepath)
    classification_model = torch.load(model_filepath, map_location=torch.device(device))

    #utils.save_pkl_results(data_dict,save_name='clean_data',folder='round5_pkls')

    #rst_dict = batch_reverse(data_dict, classification_model, device)
    #utils.save_pkl_results(rst_dict, save_name='clean_probs',folder='round5_rsts')

    rst_dict = pca_analysis(data_dict, classification_model, device)
    #utils.save_pkl_results(rst_dict, save_name='clean_pca',folder='round5_rsts')

    sc=rst_dict['variance_ratio']



    '''
        # create a prediction tensor without graph connections by copying it to a numpy array
        pred_tensor = torch.from_numpy(np.asarray(sentiment_pred)).reshape(-1).to(device)
        # predicted sentiment stands if for the ground truth label
        y_truth = pred_tensor
        adv_embedding_vector = torch.from_numpy(adv_embedding_vector).to(device)

        # get predictions based on input & weights learned so far
        if use_amp:
            with torch.cuda.amp.autocast():
                # add adversarial noise via l_inf PGD attack
                # only apply attack to attack_prob of the batches
               with advertorch.context.ctx_noparamgrad_and_eval(classification_model):
                   classification_model.train()  # RNN needs to be in train model to enable gradients
                   adv_embedding_vector = attack.perturb(adv_embedding_vector, y_truth).cpu().detach().numpy()
               adv_logits = classification_model(torch.from_numpy(adv_embedding_vector).to(device)).cpu().detach().numpy()
        else:
            # add adversarial noise vis lin PGD attack
            with advertorch.context.ctx_noparamgrad_and_eval(classification_model):
                classification_model.train()  # RNN needs to be in train model to enable gradients
                adv_embedding_vector = attack.perturb(adv_embedding_vector, y_truth).cpu().detach().numpy()
            adv_logits = classification_model(torch.from_numpy(adv_embedding_vector).to(device)).cpu().detach().numpy()

        adv_sentiment_pred = np.argmax(adv_logits)
        print('  adversarial sentiment: {}'.format(adv_sentiment_pred))

    #'''

    # Test scratch space
    # with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
    #    fh.write('this is a test')

    trojan_probability = final_adjust(sc)
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))




class DataLoader():
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.n=data.shape[0]
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.cur_i=0
        if shuffle: self.do_shuffle()

    def do_shuffle(self):
        a = list(range(self.n))
        a = np.asarray(a)
        np.random.shuffle(a)
        self.data=self.data[a,:,:]

    def next(self):
        next_i=self.cur_i+self.batch_size
        d=self.data[self.cur_i:min(self.n,next_i),:,:]
        if next_i > self.n:
            dr=self.data[0:next_i-self.n,:,:]
            d = np.concatenate([d,dr],axis=0)

        if next_i >= self.n:
            self.do_shuffle()
        self.cur_i=next_i%self.n
        return d



def batch_reverse(data_dict,classification_model,device):

    embeddings=list()
    for num in data_dict:
        embeddings.append(data_dict[num]['embedding_vector'])
    embeddings=np.concatenate(embeddings,axis=0)

    embedding_all_tensor=torch.from_numpy(embeddings).to(device)
    logits_all = classification_model(embedding_all_tensor).detach().cpu().numpy()
    preds_all = np.argmax(logits_all,axis=1)
    print(preds_all)

    batch_size=16

    rst_dict=dict()

    from torch.autograd import Variable
    import torch.nn.functional as F
    import copy

    classification_model.train()

    for tgt_lb in range(2):

        idx=preds_all!=tgt_lb
        z = np.asarray(list(range(len(preds_all))))
        w = z[idx]; v = z[~idx]
        v = np.concatenate([v,w[5:]])
        w = w[:5]
        test_data=copy.deepcopy(embeddings[w])
        train_data=copy.deepcopy(embeddings[v])

        dl = DataLoader(train_data, batch_size,shuffle=True)
        tgt_lb_numpy=np.ones([batch_size],dtype=np.int64)*tgt_lb
        tgt_lb_tensor=torch.from_numpy(tgt_lb_numpy).to(device)

        embedding_vector_numpy=dl.next()
        embedding_vector_tensor=torch.from_numpy(embedding_vector_numpy).to(device)
        zero_tensor = torch.zeros_like(embedding_vector_tensor)
        delta_shape=list(embedding_vector_numpy.shape)
        delta_shape[0]=1
        delta = np.zeros(delta_shape, dtype=np.float32)
        delta_tensor = Variable(torch.from_numpy(delta), requires_grad=True)

        opt = torch.optim.Adam([delta_tensor],  lr=0.01)

        max_step=100
        for step in range(max_step):
            if step>0:
                embedding_vector_numpy=dl.next()
                embeddgin_vector_tensor=torch.from_numpy(embedding_vector_numpy).to(device)


            altered_tensor = embedding_vector_tensor+delta_tensor.to(device)
            logits = classification_model(altered_tensor)

            ce_loss = F.cross_entropy(logits,  tgt_lb_tensor)
            #l2_loss = F.mse_loss(delta_tensor, zero_tensor)
            l2_loss = torch.norm(delta_tensor)
            loss = ce_loss + l2_loss

            #print('step %d:'%step, loss, ce_loss, l2_loss)
            #print(torch.argmax(logits,axis=1))

            opt.zero_grad()
            loss.backward()
            opt.step()

        embedding_test_tensor=torch.from_numpy(test_data).to(device)
        altered_tensor = embedding_test_tensor+delta_tensor.to(device)
        logits = classification_model(altered_tensor)
        preds=torch.argmax(logits,axis=1).detach().cpu().numpy()

        prob=np.sum(preds==tgt_lb)/len(preds)
        delta_norm=torch.norm(delta_tensor).detach().cpu().numpy()
        print(tgt_lb, prob, delta_norm)

        rst_dict[tgt_lb]={'test_prob':prob,'delta_norm':delta_norm}

    return rst_dict

def pca_analysis(data_dict,classification_model,device):

    embeddings=list()
    for num in data_dict:
        embeddings.append(data_dict[num]['embedding_vector'])
    embeddings=np.concatenate(embeddings,axis=0)

    input_rds=list()
    def hook(model, input, output):
        input_rds.append(input[0].detach().cpu().numpy())

    model=classification_model
    for ch in model.children():
        ch_name = type(ch).__name__
        if ch_name=='Linear':
            ch.register_forward_hook(hook)
            break

    embedding_all_tensor=torch.from_numpy(embeddings).to(device)
    logits_all = classification_model(embedding_all_tensor).detach().cpu().numpy()

    input_rds=np.concatenate(input_rds)
    print(input_rds.shape)

    from sklearn.decomposition import PCA
    pca=PCA()
    pca.fit(input_rds)
    ratio=pca.explained_variance_ratio_

    rst_dict={'representations':input_rds,'variance_ratio':ratio}

    return rst_dict


def final_adjust(sc):
    print(sc)
    sc=np.sum(sc[:2])
    sc=sc*2-1
    sc=max(sc,-1+1e-12)
    sc=min(sc,+1-1e-12)
    sc=np.arctanh(sc)
    va=-0.9296
    vb=0.9051
    sc=sc*va+vb
    sc=np.tanh(sc)
    sc=sc/2+0.5
    return sc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', action='store_true', default=False)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct embedding to be used with the model_filepath.', default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)



