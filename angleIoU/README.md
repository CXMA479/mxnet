this project aims to provide a *numerical* estimation method for `iou`(intersection over union )
calculation between two arbitrarily given boxes

there are two implementations: `CPU` & `GPU`

*CPU* version (*cytool.pyx*) uses `cython` to accelerate processing
*GPU* version (*gpu_IoU.py*) exploits `MXNet` to obtain a faster performance

# Usage
```sh
make
python iou-test.py
```


