import re
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
train_path = 'train.crash' #need to change
test_path = 'test.crash'#need to change

train_id, train_text, train_label = [], [], []
test_id, test_text = [], []

with open(train_path, 'r', encoding = "utf8") as f_r:
    data = f_r.read().strip()

    data = re.findall('train_[\s\S]+?\"\n[01]\n\n', data)

    for sample in data:
        splits = sample.strip().split('\n')

        id = splits[0]
        label = int(splits[-1])
        text = ' '.join(splits[1:-1])[1:-1]
        text = rdrsegmenter.tokenize(text)
        text = ' '.join([' '.join(x) for x in text])

        train_id.append(id)
        train_text.append(text)
        train_label.append(label)


with open(test_path, 'r', encoding = 'utf8') as f_r:
    data = f_r.read().strip()
    data = re.findall('train_[\s\S]+?\"\n[01]\n\n', data)

    for sample in data:
        splits = sample.strip().split('\n')

        id = splits[0]
        text = ' '.join(splits[1:])[1:-1]
        text = rdrsegmenter.tokenize(text)
        text = ' '.join([' '.join(x) for x in text])

        test_id.append(id)
        test_text.append(text)