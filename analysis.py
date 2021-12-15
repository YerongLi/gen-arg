with open('/scratch/yerong/gen-arg/data/wikievents/test.jsonl', 'r') as json_file:
    test_list = list(json_file)
print('test_list')
print(test_list[0].keys())
print(test_list[0])
# print(eval(test_list[0]).keys())
# print(eval(test_list[0])['doc_id'])
# print(eval(test_list[0])['entity_mentions'])

# with open('/scratch/yerong/gen-arg/checkpoints/gen-KAIROS-pred/predictions.jsonl', 'r') as json_file:
#     predict_list = list(json_file)
# acc = []
# for line in predict_list:
#     gold = eval(line)['gold']
#     predicted = eval(line)['predicted']
#     pcount = predicted.count('<arg>')
#     gcount = gold.count('<arg>')
#     if gcount != 0.0:
#         print(abs(pcount- gcount) /gcount)
#         acc.append(1-abs(pcount- gcount) /gcount)
#     else:
#         acc.append(1.0)
# print(sum(acc)/len(acc))