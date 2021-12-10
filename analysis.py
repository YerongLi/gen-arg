# with open('/scratch/yerong/gen-arg/data/wikievents/test.jsonl', 'r') as json_file:
#     test_list = list(json_file)

# print(eval(test_list[0]).keys())
# print(eval(test_list[0])['doc_id'])
# print(eval(test_list[0])['entity_mentions'])

with open('/scratch/yerong/gen-arg/checkpoints/gen-KAIROS-pred/predictions.jsonl', 'r') as json_file:
    predict_list = list(json_file)
acc = 0.0
for line in predict_list:
    predicted = eval(line)['predicted']
    print(predicted)
    print(predicted.count('<arg>'))