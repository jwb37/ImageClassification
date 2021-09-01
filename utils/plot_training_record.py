import itertools
import os.path as path
import matplotlib.pyplot as plt

colours = {
    'train_loss': 'b',
    'valid_loss': 'm',
    'top_1_accuracy': 'g',
    'top_5_accuracy': 'r'
}


def plot_training_record(model_name, TRAIN_SET_SIZE=26226, BATCH_SIZE=16, interactive=False, xlabels='iterations'):

    def loss_record_to_iters(e, i):
        if i==(TRAIN_SET_SIZE // BATCH_SIZE):
            return e*TRAIN_SET_SIZE
        else:
            return (e-1)*TRAIN_SET_SIZE + (i+1)*BATCH_SIZE

    records_path = path.join('../checkpoints/', model_name)
    record_files = {
        'train': path.join( records_path, 'training.log' ),
        'valid': path.join( records_path, 'validation.log' )
    }

    with open(record_files['train']) as in_file:
        train_loss_x, train_loss_y = zip(
            *[
                (
                    loss_record_to_iters(int(line.split()[1]),int(line.split()[3])),
                    float(line.split()[-1])
                )
                # Use a slice to skip lines so we only read 8 values per epoch
                # (otherwise training loss graph is VERY dense!)
                for line in itertools.islice(in_file,0,None, TRAIN_SET_SIZE // (BATCH_SIZE * 8))
            ]
        )

    with open(record_files['valid']) as in_file:
        valid_x, valid_loss_y, valid_top1, valid_top5 = zip(
            *[
                (
                    int(line.split()[1]) * TRAIN_SET_SIZE,
                    float(line.split()[5]),
                    float(line.split()[7]),
                    float(line.split()[9])
                )
                for line in in_file
            ]
        )

    fig, ax = plt.subplots()
    ax.plot( train_loss_x, train_loss_y, colours['train_loss'], label='Training loss' )
    ax.plot( valid_x, valid_loss_y, colours['valid_loss'], label='Validation loss' )
    if xlabels == 'iterations':
        ax.set_xlabel( 'Iterations' )
    elif xlabels == 'epochs':
        ax.set_xlabel( 'Epochs' )
        tick_positions = [e*TRAIN_SET_SIZE for e in range(max(valid_x)//TRAIN_SET_SIZE + 1)]
        tick_labels = [str(e) for e in range(max(valid_x)//TRAIN_SET_SIZE + 1)]
        ax.set_xticks( tick_positions )
        ax.set_xticklabels( tick_labels )
        ax.grid(axis='x')

    ax.set_ylabel( 'Loss' )

    ax2 = ax.twinx()
    ax2.set_ylabel( 'Accuracy' )
    ax2.set_yticks( list(range(0,100,10)) )
    ax2.plot( valid_x, valid_top1, colours['top_1_accuracy'], label='Validation Accuracy (top 1)' )
    ax2.plot( valid_x, valid_top5, colours['top_5_accuracy'], label='Validation Accuracy (top 5)' )

    fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5) )

    if interactive:
        plt.show()

    # Save both to model folder and one large folder (for easier comparison)
    plt.savefig( path.join(records_path, 'training_record.png') )
    plt.savefig( path.join('./training_plots/', f"{model_name}.png") )



if __name__ == "__main__":
    # Iterable of 'ModelName', 'BatchSize' pairs (see params file)
    plots = (
        ('VGG16_Sketchy', 68053, 16),
    )
    for model_name, training_set_size, batch_size in plots:
        plot_training_record(model_name, TRAIN_SET_SIZE=training_set_size, BATCH_SIZE=batch_size, xlabels='epochs')
