<h1> install vncorenlp <h1>
pip install vncorenlp

# Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
<h1> install transformers </h1>
wget https://public.vinai.io/PhoBERT_base_fairseq.tar.gz
tar -xzvf PhoBERT_base_fairseq.tar.gz
<h1> or </h1>
wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
tar -xzvf PhoBERT_base_transformers.tar.gz

