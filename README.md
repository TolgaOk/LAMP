# Logging and Monitoring Package

Offers a simple custom python logging class and Handlers for visualization via Visdom and Matplotlib. You can use additional logger functions to log your data. We support:
- [ ] Scalar
- [ ] Image
- [ ] Histogram
- [ ] Video
- [ ] Hyperparameter
loggings. In addition to logging Lamp includes Visdom and Matplotlib handlers.

### Example:
``` Python
logging.setLoggerClass(DataLogger)
my_logger = logging.getLogger("main")

handlerscalar = MatplotScalarHandler()
handlerimage = MatplotImageHandler()

my_logger.addHandler(handlerscalar)
my_logger.addHandler(handlerimage)

my_logger.setLevel(logging.DEBUG)

my_logger.image(np.random.uniform(size=(32, 32)), cmap="gray")
for i in range(50):
    my_logger.scalar(i**2)
```
- - -
## Install
``` Python
 pip install -e
```