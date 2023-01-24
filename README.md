# RAMScrapersAttentionDetectors 

This repo contains:

 - Transaction network traffic pcap format in "TransactionTracks" folder;
 - Malware network traffic in pcap format "POSmalwareTracks" folder;
 - Models implementation in "models" folder
 
 Models implemented:
 
 - __LSTM Autoencoder__
 - __LSTM Autoencoder with attention__
 
 	- __Bahdanau__ attention [[1]](#1)
 	- __Luong__ attention with several score function [[2]](#2)
 		- __Dot__ Score
 		- __General__ Score
 		- __Concat__ Score
 		- __LATTE__ Score [[3]](#3)
 - __Transformer__ [[4]](#4)

## References

[1]
Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).

[2]
Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective approaches to attention-based neural machine translation." arXiv preprint arXiv:1508.04025 (2015).

[3]
Kukkala, Vipin Kumar, Sooryaa Vignesh Thiruloga, and Sudeep Pasricha. "LATTE: L STM Self-Att ention based Anomaly Detection in E mbedded Automotive Platforms." ACM Transactions on Embedded Computing Systems (TECS) 20.5s (2021): 1-23.

[4]
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
 
