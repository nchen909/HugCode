from transformers import PLBartConfig
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import os
from models import CODE_MODEL_CLASSES
from models import TOKENIZER_CLASSES
import torch
from torch import nn
from transformers import PLBartConfig
torch.manual_seed(1234)

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CloneModel(nn.Module):
    def __init__(self, encoder, tokenizer, model_type):
        super(CloneModel, self).__init__()
        # checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
        # config = AutoConfig.from_pretrained(checkpoint)
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = ClassificationHead(self.encoder.config)
        self.model_type = model_type


    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)

        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.tokenizer.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        # position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        # position_ids = position_ids*attention_mask
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.tokenizer.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        # attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def get_unixcoder_vec(self, source_ids):
        attention_mask = source_ids.ne(1)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask

        outputs = self.encoder(source_ids,attention_mask=attention_mask)[0]#shape:batch_size*max_len512*hidden_size768

        outputs = (outputs * source_ids.ne(1)[:,:,None]).sum(1)/source_ids.ne(1).sum(1)[:,None]#shape:batch_size*hidden_size
        outputs = outputs.reshape(-1,2,outputs.size(-1))#shape:batch_size/2 *2*hidden_size
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        return cos_sim #cos_sim, labels

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, 512)#[batch*2,512]

        if self.model_type in ['t5','codet5']:
            vec = self.get_t5_vec(source_ids)#[batch*2,768]
            logits = self.classifier(vec)#[batch,2]
            prob = nn.functional.softmax(logits)
        elif self.model_type in ['bart','plbart']:
            vec = self.get_bart_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.model_type in ['roberta', 'codebert', 'graphcodebert']:
            vec = self.get_roberta_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.model_type in ['unixcoder']:
            logits = self.get_unixcoder_vec(source_ids)
            prob = logits #=cos_sim

        if labels is not None:
            if self.model_type not in ['unixcoder']:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                loss = ((logits-labels.float())**2).mean()
                return loss, prob
        else:
            return prob

class HugCloneAPI:
    def __init__(self, model_type, hugcode_model_name_or_path) -> None:
        if model_type not in CODE_MODEL_CLASSES["code_generation"].keys():
            raise KeyError(
                "You must choose one of the following model: {}".format(
                    ", ".join(
                        list(CODE_MODEL_CLASSES["code_generation"].
                             keys()))))
        self.model_type = model_type
        self.config =PLBartConfig.from_pretrained(hugcode_model_name_or_path)
        self.tokenizer = TOKENIZER_CLASSES[self.model_type].from_pretrained(
            hugcode_model_name_or_path)
        model = CODE_MODEL_CLASSES["code_generation"][
            self.model_type](self.config)#.from_pretrained(hugcode_model_name_or_path)
        model = CloneModel(model, self.tokenizer, self.model_type)
        model = model.module if hasattr(model, 'module') else model
        file = os.path.join(hugcode_model_name_or_path, 'pytorch_model.bin')
        model.load_state_dict(torch.load(file))
        self.model = model
        self.max_source_length = 512
        self.max_target_length = 512

    def convert_clone_examples_to_features(self,source,target):
        source = ' '.join(source.split())
        target = ' '.join(target.split())
        if self.model_type in ['t5', 'codet5']:
            source_str = "{}: {}".format('clone', source)
            target_str = "{}: {}".format('clone', target)
        elif self.model_type in ['unixcoder']:
            source_str = self.tokenizer.tokenize(source[:self.max_source_length-4])#format_special_chars(self.tokenizer.tokenize(source[:args.max_source_length-4]))
            source_str =[self.tokenizer.cls_token,"<encoder-only>",self.tokenizer.sep_token]+source_str+[self.tokenizer.sep_token]
            target_str = self.tokenizer.tokenize(target[:self.max_target_length-4])#format_special_chars(self.tokenizer.tokenize(target[:args.max_target_length-4]))
            target_str =[self.tokenizer.cls_token,"<encoder-only>",self.tokenizer.sep_token]+target_str+[self.tokenizer.sep_token]
            example_index = source_str + target_str
        else:

            source_str = source
            target_str = target
        if self.model_type in ['unixcoder']:
            code1 = self.tokenizer.convert_tokens_to_ids(source_str)
            padding_length = self.max_source_length - len(code1)
            code1 += [self.tokenizer.pad_token_id]*padding_length

            code2 = self.tokenizer.convert_tokens_to_ids(target_str)
            padding_length = self.max_target_length - len(code2)
            code2 += [self.tokenizer.pad_token_id]*padding_length
            source_ids = code1 + code2
        else:
            code1 = self.tokenizer.encode(
                source_str, max_length=self.max_source_length, padding='max_length', truncation=True)
            code2 = self.tokenizer.encode(
                target_str, max_length=self.max_target_length, padding='max_length', truncation=True)
            source_ids = code1 + code2
        return torch.tensor(
                [source_ids], dtype=torch.long)

    def request(self, func1: str, func2: str):

        inputs = self.convert_clone_examples_to_features(func1,func2)     
        outputs = self.model(inputs)
        probs = outputs#torch.softmax(outputs.logits, dim=-1)
        clone_probability = probs[0, 1].item()

        return clone_probability
# 从本地路径加载模型和配置

# model = CODE_MODEL_CLASSES["code_cls"]["plbart"].from_pretrained("/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1")#("/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1/pytorch_model.bin")
# tokenizer = TOKENIZER_CLASSES["plbart"].from_pretrained("/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1")#("/root/autodl-tmp/CodePrompt/data/huggingface_models/plbart-base/sentencepiece.bpe.model")


if __name__ == "__main__":
    from scripts.HugClone.clone_api import HugCloneAPI
    model_type = "plbart"
    hugie_model_name_or_path = "nchen909/plbart-base-finetuned-clone-detection"#"/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/ckpt_test/"
    hugie = HugCloneAPI(model_type, hugie_model_name_or_path)

    ## JAVA code clone detection
    func1="""
    public String getData(DefaultHttpClient httpclient) {
        try {
            HttpGet get = new HttpGet("http://3dforandroid.appspot.com/api/v1/note");
            get.setHeader("Content-Type", "application/json");
            get.setHeader("Accept", "*/*");
            HttpResponse response = httpclient.execute(get);
            HttpEntity entity = response.getEntity();
            InputStream instream = entity.getContent();
            responseMessage = read(instream);
            if (instream != null) instream.close();
        } catch (ClientProtocolException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return responseMessage;
    }
    """
    func2="""
    public static void copyFile(File in, File out) throws Exception {
        FileChannel sourceChannel = new FileInputStream(in).getChannel();
        FileChannel destinationChannel = new FileOutputStream(out).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
    """
    clone_probability = hugie.request(func1, func2)
    print("clone_probability:{}".format(clone_probability))
    print("\n\n")

    ## JAVA code clone detection
    func1="""
    public static void copyFile(File source, File dest) throws IOException {
        FileChannel in = null, out = null;
        try {
            in = new FileInputStream(source).getChannel();
            out = new FileOutputStream(dest).getChannel();
            in.transferTo(0, in.size(), out);
        } catch (FileNotFoundException fnfe) {
            Log.debug(fnfe);
        } finally {
            if (in != null) in.close();
            if (out != null) out.close();
        }
    }
    """

    func2="""
    public static void copyFile(File from, File to) throws IOException {
        if (from.isDirectory()) {
            if (!to.exists()) {
                to.mkdir();
            }
            File[] children = from.listFiles();
            for (int i = 0; i < children.length; i++) {
                if (children[i].getName().equals(".") || children[i].getName().equals("..")) {
                    continue;
                }
                if (children[i].isDirectory()) {
                    File f = new File(to, children[i].getName());
                    copyFile(children[i], f);
                } else {
                    copyFile(children[i], to);
                }
            }
        } else if (from.isFile() && (to.isDirectory() || to.isFile())) {
            if (to.isDirectory()) {
                to = new File(to, from.getName());
            }
            FileInputStream in = new FileInputStream(from);
            FileOutputStream out = new FileOutputStream(to);
            byte[] buf = new byte[32678];
            int read;
            while ((read = in.read(buf)) > -1) {
                out.write(buf, 0, read);
            }
            closeStream(in);
            closeStream(out);
        }
    }
    """
    clone_probability = hugie.request(func1, func2)
    print("clone_probability:{}".format(clone_probability))

"""
clone_probability:2.0006775685033062e-06
clone_probability:0.9999953508377075
    """
