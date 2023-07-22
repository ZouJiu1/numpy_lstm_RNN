# RNN and LSTM in numpy 
I write a cnn network in numpy fully, including forward and backpropagation.<br>
including those layers, **Fullconnect**, **RNNCell**, **LSTMCell**, **Embedding**, **Cross Entropy loss** and **MSE loss**<br>
In training, it use cpu，it can train with Chinse poetry<br>

## Train and predict
### character
rnn English 26 character
```
python rnn_1layer\train_rnn_1layer.py
```
```
python rnn_1layer\predict_rnn_1layer.py
```
lstm English 26 character
```
python lstm_1layer\train_lstm_1layer.py
```
```
python lstm_1layer\predict_lstm_1layer.py
```

### Poetry
rnn in character
```
python rnn_3layer_chars\train_rnn_2layerV2_Embedding_highfreq.py
```
```
python rnn_3layer_chars\predict_rnn_2layerV2_Embedding_highfreq.py
```

#### rnn in [hanlp](https://github.com/hankcs/HanLP) words segmentation<br>
*_dynamic is the dynamic rnn, dynamic input sequence and dynamic output sequence<br>
```
python rnn_3layer\train_rnn_2layer_Embedding_dynamic.py
python rnn_3layer\predict_rnn_2layer_Embedding_dynamic.py
```
*_V2_Embedding.py's labels is moving one word by input words, input sequence==output sequence<br>
```
python rnn_3layer\train_rnn_2layerV2_Embedding.py
python rnn_3layer\predict_rnn_2layerV2_Embedding.py
```
output sequence = 1<br>
```
python rnn_3layer\train_rnn_2layer_Embedding.py
python rnn_3layer\predict_rnn_2layer_Embedding.py
```

#### lstm in [hanlp](https://github.com/hankcs/HanLP) words segmentation<br>
*_dynamic is the dynamic lstm, dynamic input sequence and dynamic output sequence<br>
```
python lstm_3layer\train_lstm_2layer_Embedding_dynamic.py
python lstm_3layer\predict_lstm_2layer_Embedding_dynamic.py
```
*_V2_Embedding.py's labels is moving one word by input words, input sequence==output sequence<br>
```
python lstm_3layer\train_lstm_2layerV2_Embedding.py
python lstm_3layer\predict_lstm_2layerV2_Embedding.py
```
output sequence = 1<br>
```
python lstm_3layer\train_lstm_2layer_Embedding.py
python lstm_3layer\predict_lstm_2layer_Embedding.py
```
bidirectional lstm<br>
```
python lstm_3layer\train_bidirectional_lstm_2layerV2_Embedding.py
python lstm_3layer\predict_bidirectional_lstm_2layerV2_Embedding.py
```
### FORSHOW
```
输入：暮云千山雪
暮云千山雪，
春行复深上。
风无流鸟树，
清客鸟归还。

输入：朝送山僧去
朝送山僧去，
莫君在梦何。
不知山不知，
山中我欲幸。

输入：携杖溪边听
携杖溪边听，
抱我树月知。
故山中常更，
鸟中应上鬓。

输入：楼高秋易寒
楼高秋易寒,
凭谁暮云云,
添我下来衣,
知一别来云,

输入：残星落檐外
残星落檐外,
馀月罢窗来,
水白先成秋,
霞暗未成不,

输入：月在画楼西
月在画楼西，
烛故是愁来。
何转知此山，
花常更花中。
```
## blogs
[numpy实现RNN层的前向传播和反向传播-https://zhuanlan.zhihu.com/p/645190373](https://zhuanlan.zhihu.com/p/645190373)<br>
[numpy实现LSTM层的前向传播和反向传播-https://zhuanlan.zhihu.com/p/645261658](https://zhuanlan.zhihu.com/p/645261658)<br>
[numpy实现embedding层的前向传播和反向传播-https://zhuanlan.zhihu.com/p/642997702](https://zhuanlan.zhihu.com/p/642997702)<br>
[全连接层的前向传播和反向传播-https://zhuanlan.zhihu.com/p/642043155](https://zhuanlan.zhihu.com/p/642043155)<br>
[损失函数的前向传播和反向传播-https://zhuanlan.zhihu.com/p/642025009](https://zhuanlan.zhihu.com/p/642025009)<br>

### Reference
[https://blog.csdn.net/SHU15121856/article/details/104387209](https://blog.csdn.net/SHU15121856/article/details/104387209)<br>
[https://hanlp.hankcs.com/docs/api/hanlp/pretrained/tok.html](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/tok.html)<br>
[https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)<br>
[https://github.com/Werneror/Poetry](https://github.com/Werneror/Poetry)<br>
[https://discuss.pytorch.org/t/how-nn-embedding-trained/32533/11](https://discuss.pytorch.org/t/how-nn-embedding-trained/32533/11)<br>
[https://zhuanlan.zhihu.com/p/247970862](https://zhuanlan.zhihu.com/p/247970862)<br>
[https://zhuanlan.zhihu.com/p/147685918](https://zhuanlan.zhihu.com/p/147685918)<br>
[https://zhuanlan.zhihu.com/p/28054589](https://zhuanlan.zhihu.com/p/28054589)<br>
[https://zhuanlan.zhihu.com/p/371849556](https://zhuanlan.zhihu.com/p/371849556)<br>
[https://zhuanlan.zhihu.com/p/54868269](https://zhuanlan.zhihu.com/p/54868269)<br>
[https://zhuanlan.zhihu.com/p/488710218](https://zhuanlan.zhihu.com/p/488710218)<br>
[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)<br>
[https://blog.csdn.net/zhaojc1995/article/details/80572098](https://blog.csdn.net/zhaojc1995/article/details/80572098)<br>
[https://github.com/wzyonggege/RNN_poetry_generator](https://github.com/wzyonggege/RNN_poetry_generator)<br>
[https://github.com/stardut/Text-Generate-RNN](https://github.com/stardut/Text-Generate-RNN)<br>
[https://github.com/youyuge34/Poems_generator_Keras](https://github.com/youyuge34/Poems_generator_Keras)<br>
[https://github.com/justdark/pytorch-poetry-gen](https://github.com/justdark/pytorch-poetry-gen)<br>