import torch
from transformers import AutoTokenizer, AutoModelForPreTraining


path = '/home/halamri/summer2020/avsd-transofrmers/mlmAVSD/'

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForPreTraining.from_pretrained(path)

ques = 'Hey..how are you?'
ans = 'I will be just fine !'

input_ids = tokenizer.encode_plus(text=[ques,ans],
                                  add_special_tokens=True,
                                  max_length=200,
                                  pad_to_max_length=True,
                                  return_tensors='pt')

model.eval()
output = model(input_ids=input_ids.data['input_ids'] ,
               attention_mask=input_ids.data['attention_mask'],
               token_type_ids=input_ids.data['token_type_ids'])

print('Test..')