# Logging and Monitoring Package
### **Under development!**

Lamp offers a simple custom python logging class and Handlers for visualization via Visdom and Matplotlib. You can use additional logger functions to log your data on top of the standard python logging functionality. We support:

- [x] Scalar
- [x] Image
- [x] Histogram
- [x] Video
- [x] Hyperparameter
- [ ] Scatter
  
In addition to logging Lamp includes Visdom and Matplotlib handlers. Which provide online and offline visualization options for logged data.

### Example:
``` Python
logging.setLoggerClass(lamp.DataLogger)
my_logger = logging.getLogger(__name__)

vizscalarhandler = lamp.VisdomScalarHandler(logging.INFO,
                                        overwrite_window=True,
                                        lock=None)
vizhisthandler = lamp.VisdomHistHandler(logging.DEBUG,
                                        overwrite_window=True)

my_logger.addHandler(vizscalarhandler)
my_logger.addHandler(vizhisthandler)
my_logger.setLevel(logging.DEBUG)


for i in range(10):
    my_logger.scalar(i**2, win="polynomial", trace="x^2")
    my_logger.scalar(i**2-5*i + 2, win="polynomial", trace="x^2 - 5x + 2")

gauss_vector = torch.randn(500)
my_logger.histogram(gauss_vector, win="Normal distribution")

```
![image](https://github.com/TolgaOk/LAMP/blob/master/doc/viz-examle.png)

Lamp also provides offline plotting via Matplotlib. It automatically plots the logged data from the buffer just before the shutdown.

A general use case of Lamp can be seen below.

![image](https://github.com/TolgaOk/LAMP/blob/master/doc/doc-image-1.png)


- - -
## Requirements
- Visdom
- Matplotlib
- Pytorch

- - -
## Install
``` Python
 pip install -e
```