import hydra
from omegaconf import DictConfig
from pyvirtualdisplay import Display

from hw3.hw3 import my_app

@hydra.main(config_path="conf", config_name="config_hw3")
def my_main(cfg: DictConfig):
    my_app(cfg)

if __name__ == "__main__":
    # Initialize PyVirtualDisplay here
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Now, run your main function
    my_main()

    # Stop the display after your application is done
    display.stop()