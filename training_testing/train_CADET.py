# BASIC LIBRARIES
import io, os, sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

# TENSORFLOW
import tensorflow as tf
from tensorflow.config.experimental import list_physical_devices, set_memory_growth, set_virtual_device_configuration, VirtualDeviceConfiguration
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorboard.plugins.hparams import api as hp

# GPU initialization
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = list_physical_devices('GPU')
set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=5000)])
print(len(gpus), "Physical GPUs")

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.35'

# # GPU initialization
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# gpus = list_physical_devices('GPU')
# set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=2600)])
# print(len(gpus), "Physical GPUs")
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'

# CUSTOM LIBRARIES
from functions import *
from testing import *
from img_generator import *

#################### PARSING ARGUMENTS ####################

try: network = sys.argv[3]
except: network = ""
try: data = sys.argv[4]
except: data = ""
try: drop = float(sys.argv[5])
except: drop = 0.0

if "50" in network: cavities = 50
elif "90" in network: cavities = 90
elif "100" in network: cavities = 100
else: cavities = 100
active = "prelu" if "prelu" in network else "relu"

shape_image = (128, 128, 1)

###################### PICKING MODEL ######################

if "UNET" in network: 
    from unet import UNET
    model = UNET(shape_image, num_classes=1)
elif "VGG" in network: 
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=1, input_height=shape_image[0], input_width=shape_image[1])
elif "customunet" in network: 
    from keras_unet.models import custom_unet
    model = custom_unet(input_shape=shape_image, use_batch_norm=True,
                                        num_classes=1, dropout=0.0, # filters=128,
                                        #dropout_change_per_layer=0.0, activation="relu",
                                        output_activation='sigmoid')
elif "vanilla" in network: 
    from stannet_vanilla import Stannet
    model = Stannet_vanilla(shape_image)
else:
    from stannet import Stannet
    model = Stannet(shape_image, active, drop)

# # TEST ON DATA SAME AS GENERATED
# params = read_csv(f"simulated_params/sim{data}.csv")
# X_test, y_test, _ = read_data(0, 24, network, params)

# for i in range(16):
#     plt.subplot(1,2,1)
#     plt.imshow(X_test[i,:,:,0], cmap="gray")
#     plt.axis("off")
#     plt.subplot(1,2,2)
#     plt.imshow(y_test[i,:,:,0], cmap="gray")
#     plt.axis("off")
#     plt.savefig(f"img_{i}.png")

######################## TRAINING ########################

# basic params
epochs = 4
n = 0
lr = float(sys.argv[1])
batch = int(sys.argv[2])
images_per_epoch = 6144 * 2
steps_per_epoch = images_per_epoch // batch
N_train = images_per_epoch * epochs
N_test = 10000

N_test_nocav = 2000
N_val = 2000

hparams = {"dropout" : drop,
           "lr" : lr,
           "cavities" : cavities}

string = f"\nTraining \"{network}\" network with \"{data}\" data."
string += f"\nLearning rate: {lr}"
string += f"\nDropout: {drop}"
string += f"\nFraction of cavities: {cavities}"
string += f"\nEpochs: {epochs}"
string += f"\nBatch size: {batch}"
string += f"\nSteps per epoch: {steps_per_epoch}"
string += f"\nImages per epoch: {images_per_epoch}"
string += f"\nTrain data: {N_train}"
string += f"\nTest data: {N_test}"
string += f"\nValidation data: {N_val}"
print(string)

# VALIDATION DATA
X_val, y_val, _, _ = load_test_data(N_val, N0=N_test)

# LOAD SIMULATED PARAMETERS
params = pd.read_csv(f"simulated_params/sim{data}.csv")
params_test = pd.read_csv(f"simulated_params/sim{data}_test.csv")
betas = np.array(params_test["beta"])[:N_test]
betas_nocav = np.array(params_test["beta"])[-2000:]

# FILENAME AND LOGDIR
timestamp = f'b{batch}_lr{lr}_d{drop}{network}'
log_dir = f"logs/{timestamp}"
test_dir = f"{log_dir}/test"
os.makedirs(test_dir, exist_ok=True)
file_writer = tf.summary.create_file_writer(log_dir + "/train")
with open(f"{log_dir}/info.txt", "w") as f: f.write(string)

# Tensorflow callbacks
callbacks = [TensorBoard(log_dir=log_dir, update_freq="epoch", write_graph=True),
             ModelCheckpoint(filepath=f"{log_dir}/best.hdf5", save_best_only=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
             hp.KerasCallback(log_dir+"/train", hparams, trial_id=timestamp)]

# COMPILE MODEL
adam_opt = Adam(learning_rate=lr, decay=0)
model.compile(optimizer=adam_opt, loss=BinaryCrossentropy(), metrics=["binary_accuracy"])

# SAVE ARCHITECTURE SPECIFICATIONS
with open(f"{log_dir}/{timestamp}.txt", "w") as f: 
    model.summary(print_fn=lambda x: f.write(x + "\n"))
plot_model(model, to_file=f"{log_dir}/{timestamp}.png", show_shapes=True) #, rankdir="LR")

# FIT MODEL
# model = load_model(f"{log_dir}/{timestamp}.hdf5")
model = load_model(f"{log_dir}/best.hdf5")

# model.fit(img_generator(batch, network, params=params),
#           validation_data=(X_val, y_val),
#           epochs=0,
#           initial_epoch=32,
#           steps_per_epoch=steps_per_epoch,
#           callbacks=callbacks, verbose=1)

# model.save(f"{log_dir}/{timestamp}.hdf5")

######################## TEST DATA ########################

with file_writer.as_default():
    # IMAGES WITH CAVITIES
    X_test, y_test, v_test, c_test = load_test_data(N_test, N0=0)
    y_pred = model.predict(X_test)

    X_test = jnp.asarray(X_test).reshape(N_test, 128, 128)
    y_test = jnp.asarray(y_test).reshape(N_test, 128, 128)
    y_pred = jnp.asarray(y_pred).reshape(N_test, 128, 128)

    b = (c_test > 5000) & (betas < 1.5) & (c_test < 1e7)
    X_test, y_test, y_pred, v_test, c_test, betas = X_test[b], y_test[b], y_pred[b], v_test[b], c_test[b], betas[b]

    # PLOT TEST IMAGES
    for i in range(10):    
        fig = plot_testgal_prediction("", X_test[i], y_pred[i], y_test[i], betas[i])
        fig.savefig(f"{test_dir}/test_{i}.png", bbox_inches='tight')
        tf.summary.image("6) Testing images", plot_to_image(fig), step=i)

    # BINARY CROSSENTROPY LOSS FOR TESTING DATA
    score1 = BinaryCrossentropy()(y_test[:N_test//2], y_pred[:N_test//2]).numpy()
    score2 = BinaryCrossentropy()(y_test[N_test//2:], y_pred[N_test//2:]).numpy()
    score = (score1 + score2) / 2
    tf.summary.scalar('test_loss', data=score, step=0)

    # THRESHOLDS
    print("\nCalculating errors for test data with cavities for various thresholds.")
    skip = 2
    thresholds = np.linspace(0.1,0.9,9)
    Ae_th, Ve_th, TP_th = get_threshold_error(y_pred[::skip], y_test[::skip], v_test[::skip], thresholds)

    # IMAGE WITHOUT CAVITIES
    print("\nCalculating false-positive rate vs counts for various thresholds.")
    X_test_nocav, _, _, c_test_nocav = load_test_data(N_test_nocav, N0=0, nocav=True)
    y_pred_nocav = model.predict(X_test_nocav)

    X_test_nocav = jnp.asarray(X_test_nocav).reshape(N_test_nocav, 128, 128)
    y_pred_nocav = jnp.asarray(y_pred_nocav).reshape(N_test_nocav, 128, 128)

    b = (c_test_nocav > 5000) & (betas_nocav < 1.5) & (c_test_nocav < 1e7)
    X_test_nocav, y_pred_nocav, betas_nocav, c_test_nocav = X_test_nocav[b], y_pred_nocav[b], betas_nocav[b], c_test_nocav[b]

    # TEST NOCAV IMAGES
    for i in range(10):    
        fig = plot_testgal_prediction("", X_test_nocav[i], y_pred_nocav[i], np.zeros((128,128)), betas_nocav[i])
        fig.savefig(f"{test_dir}/test_nocav_{i}.png", bbox_inches='tight')
        tf.summary.image("7) Testing nocav images", plot_to_image(fig), step=i)

    # GET FALSE POSITIVE FOR THRESHOLDS
    FP_th = []
    for th in tqdm.tqdm(thresholds):
        FP_th.append(get_false_positives(y_pred_nocav, th))

    # AREA ERROR VS THRESHOLD BINNED BY BETAS
    fig = plot_discrimination_threshold_by_betas(Ae_th, TP_th, betas[::skip], thresholds, N_bins=3, name="area")
    fig.savefig(f"{test_dir}/area_threshold_betas.png", bbox_inches='tight')
    tf.summary.image("4) Discrimination threshold", plot_to_image(fig), step=0)

    # VOLUME ERROR VS THRESHOLD BINNED BY BETAS
    fig = plot_discrimination_threshold_by_betas(Ve_th, TP_th, betas[::skip], thresholds, N_bins=3, name="volume")
    fig.savefig(f"{test_dir}/volume_threshold_betas.png", bbox_inches='tight')
    tf.summary.image("4) Discrimination threshold", plot_to_image(fig), step=1)

    # AREA ERROR VS THRESHOLD BINNED BY COUNTS
    fig = plot_discrimination_threshold_by_counts(Ae_th, TP_th, c_test[::skip], thresholds, N_bins=3, name="area")
    fig.savefig(f"{test_dir}/area_threshold_counts.png", bbox_inches='tight')
    tf.summary.image("4) Discrimination threshold", plot_to_image(fig), step=2)

    # VOLUME ERROR VS THRESHOLD BINNED BY COUNTS
    fig = plot_discrimination_threshold_by_counts(Ve_th, TP_th, c_test[::skip], thresholds, N_bins=3, name="volume")
    fig.savefig(f"{test_dir}/volume_threshold_counts.png", bbox_inches='tight')
    tf.summary.image("4) Discrimination threshold", plot_to_image(fig), step=3)

   # FPRATE VS THRESHOLD BINNED BY COUNTS
    fig = plot_fprate_vs_discrimination_threshold_by_counts(FP_th, c_test_nocav, TP_th, c_test[::skip], thresholds, N_bins=3)
    fig.savefig(f"{test_dir}/fp_tp_rate_by_counts.png", bbox_inches='tight')
    tf.summary.image("4) Discrimination threshold", plot_to_image(fig), step=4)

    # ERROR, FP, TP VS THRESHOLD
    fig = plot_error_fprate_vs_discrimination_threshold(Ae_th, Ve_th, FP_th, TP_th, thresholds)
    fig.savefig(f"{test_dir}/discrinimation_threshold.pdf", bbox_inches='tight')
    tf.summary.image("4) Discrimination threshold", plot_to_image(fig), step=5)

    N_bins = 3

    # OPTIMAL THRESHOLD VS COUNTS
    fig, bins, optimal, TP_optimal, FP_optimal = plot_optimal_threshold_vs_counts(Ae_th, Ve_th, TP_th, FP_th, c_test[::skip], c_test_nocav, thresholds, N_bins=N_bins, name="volume")
    fig.savefig(f"{test_dir}/optimal_threshold_vs_counts.pdf", bbox_inches='tight')
    tf.summary.image("5) Optimal threshold vs counts", plot_to_image(fig), step=0)

    # ERRORS & TRUE-POSITIVES
    print("\nCalculating errors for test data with cavities.")
    FP_optimal = np.max(np.vstack((optimal, FP_optimal)), axis=0)
    A, Ae, V, Ve, TP = get_error(y_pred, y_test, v_test, c_test, bins, optimal, FP_optimal)

    # FALSE-POSITIVES
    print("\nCalculating false-positive rate for test data without cavities.")
    FP = get_optimal_false_positives(y_pred_nocav, c_test_nocav, bins, optimal, FP_optimal)

    # AREA & VOLUME ERROR VS COUNTS
    fig = plot_error_vs_counts(Ae, Ve, TP, c_test, FP, c_test_nocav, N_bins=7)
    fig.savefig(f"{test_dir}/error_vs_counts.pdf", bbox_inches='tight')
    tf.summary.image("3) Error vs counts", plot_to_image(fig), step=0)

    # AREA ERROR MATRIX
    fig = plot_error_matrix(A[TP == 1], log=False, name="area (pixels$^{\\text{2}}$)")
    fig.savefig(f"{test_dir}/area_error_matrix.png", bbox_inches='tight')
    tf.summary.image("2) Confusion matrix", plot_to_image(fig), step=0)

    # VOLUME ERROR MATRIX
    fig = plot_error_matrix(V[TP == 1], log=False, name="volume (pixels$^{\\text{3}}$)")
    fig.savefig(f"{test_dir}/volume_error_matrix.pdf", bbox_inches='tight')
    tf.summary.image("2) Confusion matrix", plot_to_image(fig), step=1)

    # SCALARS: ERROR, FP, TP
    error = lambda x: abs(x - 1)*100
    tf.summary.scalar('test_volume_error', data=np.median(error(Ve[TP == 1])), step=n-1)
    tf.summary.scalar('test_TP', data=np.mean(TP), step=n-1)
    tf.summary.scalar('test_FP', data=np.mean(FP), step=n-1)

    # MAKE LATEX TABLE
    with open("test.txt", "a") as f:
        median = np.median(error(Ve[TP == 1]))
        volume = "${{{0:.1f}}}^{{+{1:.1f}}}_{{-{2:.1f}}}$".format(median, *abs(np.quantile(error(Ve[TP == 1]), (0.75, 0.25))-median))
        f.write(f"{cavities:.0f}\% & {lr} & {drop:.1f} & {score:.2g} & {optimal[0]:.2f} & {FP_optimal[0]:.2f} & {volume} & {np.mean(TP):.2f} & {np.mean(FP):.2f} & 4/6\\\\\n")

######################## REAL DATA ########################

print("\nTesting CADET on real Chandra images.")

real_gals = sorted(glob.glob("real_data/*.fits"))

real_imgs = read_realgals(real_gals)

for i, img, gal in zip(range(len(real_gals)), real_imgs, real_gals):    
    y_pred = np.zeros((128, 128))
    for j in [1,2,3,0]:
        rotated = np.rot90(img, j)
        pred = model.predict(rotated.reshape(1, 128, 128, 1)).reshape(128 ,128)
        pred = np.rot90(pred, -j)
        y_pred += pred / 4

    cavs, x, y, cluster = decompose(np.where(y_pred > optimal[0], y_pred, 0), threshold2=FP_optimal[0])
    # cavs, x, y, cluster = decompose(np.where(y_pred > 0.3, y_pred, 0), threshold2=0.3)

    with file_writer.as_default():
        gal = gal.split("/")[-1].split("_")[0]
        # fig = plot_realgal_prediction(gal, real_imgs[i], real_gals_pred[i])
        fig = plot_realgal_prediction(gal, img, cavs, x, y, cluster)
        fig.savefig(f"{test_dir}/{gal}.png", bbox_inches='tight')
        tf.summary.image("1) Real galaxies", plot_to_image(fig), step=i)

####################### CUSTOM DATA #######################

with file_writer.as_default():
    # LINE-OF-SIDE ANGLE
    print("\nCalculating error vs line-of-side angle")
    N_angles, N, N2 = 10, 1200, 50
    Xs, ys, As, Vs = get_los_angle_error(N_angles=N_angles, N=N, params=params_test)

    fig = plot_los_angle(model, N_angles, N, ys, Xs, As, Vs) #, bins, optimal)
    fig.savefig(f"{test_dir}/los_angle.pdf", bbox_inches="tight")
    tf.summary.image("8) Error vs line-of-sight angle", plot_to_image(fig), step=0)


