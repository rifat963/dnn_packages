
class visulaize_model(object):

    def plot_loss(self, history):
        """
        This function plot
        Arg:
          1) history --> predictive model results containing losses

        """
        plt.figure(figsize=(12, 8))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.legend(['train', 'val'], loc='upper right', fontsize=18)
        plt.show()

