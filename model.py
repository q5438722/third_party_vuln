import torch
import torch.nn as nn

from transformers import AutoModel, GPTBigCodePreTrainedModel, GPTBigCodeModel
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput


class BinaryClassifier(nn.Module):
    def __init__(self, config):
        super(BinaryClassifier, self).__init__()
        self.num_labels = 1

        self.codebert = AutoModel.from_pretrained(**config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 1)

        # self.init_weights()
        self.alpha = [1, 10]
        self.gamma = 2

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.codebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print('wow')
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = BCEWithLogitsLoss()
                # print(logits.view(-1), labels.view(-1).float())
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
                alpha = self.alpha[0] * (1 - labels.view(-1).float()) + self.alpha[1] * labels.view(-1).float()
                loss = (alpha * loss).mean()
                # print('type', type(loss))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print('loss: ', loss)
            # outputs = (loss,) + outputs
        # print(loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits)  # (loss), logits, (hidden_states), (attentions)
