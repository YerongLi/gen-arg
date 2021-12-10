with open('/scratch/yerong/gen-arg/data/wikievents/test.jsonl', 'r') as json_file:
    test_list = list(json_file)

print(type(test_list[0]))
with open('/scratch/yerong/gen-arg/checkpoints/gen-KAIROS-pred/predictions.jsonl', 'r') as json_file:
    test_list = list(json_file)