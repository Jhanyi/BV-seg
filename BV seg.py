"""
Implementation of U-Net with RESNET50 architecture for segmentation of blood vessels in mouse cortex acquire with SEM
"""

import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from skimage import io
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
%env SM_FRAMEWORK = tf.keras
import segmentation_models as sm

tf.test.is_gpu_available(
  cuda_only=False, min_cuda_compute_capability=None
)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

######################################### read images into array ############################################
image_file = r'V:\Hanyi\Utils\BV seg\images'
mask_file = r'V:\Hanyi\Utils\BV seg\masks'


images_file_list = next(os.walk(image_file))[2]
images_sorted = sorted(images_file_list, key=lambda x:x[:4])
image_array = []
for file in images_sorted:
  image = io.imread(image_file+'\\{}'.format(file))
  image_array.append(image)
image_array = np.asarray(image_array)

masks_file_list = next(os.walk(mask_file))[2]
masks_sorted = sorted(masks_file_list, key=lambda x:x[:4])
mask_array = []
for mask in masks_sorted:
  mask = io.imread(mask_file+'\\{}'.format(mask))
  mask_array.append(mask)
mask_array = np.asarray(mask_array)


######################################### generate patches ############################################
HEIGHT = 1024
WIDTH = 1024

image_patches = []
for image in image_array:
    image_patch=patchify(image,(HEIGHT,WIDTH), step=HEIGHT)
    for i in range(image_patch.shape[0]):
        for j in range(image_patch.shape[1]):
            single_image_patch=image_patch[i,j,:,:]

            image_patches.append(single_image_patch)
image_patches = np.array(image_patches)
image_patches = np.expand_dims(image_patches, -1)
# image_patches=np.stack((image_patches,)*3, axis=-1)

mask_patches = []
for mask in mask_array:
    mask_patch=patchify(mask,(HEIGHT,WIDTH), step=HEIGHT)
    for i in range(mask_patch.shape[0]):
        for j in range(mask_patch.shape[1]):
            single_mask_patch = mask_patch[i, j, :, :]

            mask_patches.append(single_mask_patch)
mask_patches = np.array(mask_patches, dtype='bool')
mask_patches = np.expand_dims(mask_patches, -1)

##################################### train test split #############################################
BACKBONE = 'resnet34'
X_train, X_test, y_train, y_test = train_test_split(image_patches, mask_patches, test_size=0.25, random_state=42)

preprocess_input1 = sm.get_preprocessing(BACKBONE)
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

randnum = np.random.randint(0,len(X_train1))
fig, (ax1, ax2) = plt.subplots(1,2, sharey='row',figsize=(12,6))
ax1.imshow(np.reshape(X_train1[randnum,:,:,0], (1024,1024)), cmap='gray')
ax2.imshow(np.reshape(y_train[randnum,:,:,0], (1024,1024)), cmap='gray')


##################################### data augmentation #############################
from tensorflow.keras.preprocessing.image import ImageDataGenerator
seed = 24
image_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect',
                          preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(
                              x.dtype))

batch_size = 8

image_data_gen = ImageDataGenerator(**image_data_gen_args)
image_gen = image_data_gen.flow(X_train1, seed=seed, batch_size=batch_size)
valid_image_gen = image_data_gen.flow(X_test1, seed=seed, batch_size=batch_size)

mask_data_gen = ImageDataGenerator(**mask_data_gen_args)
mask_gen = mask_data_gen.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_gen = mask_data_gen.flow(y_test, seed=seed, batch_size=batch_size)

def my_image_mask_gen(image_gen, mask_gen):
    train_generator = zip(image_gen, mask_gen)
    for (image, mask) in train_generator:
        yield (image, mask)

my_generator = my_image_mask_gen(image_gen, mask_gen)
validation_datagen = my_image_mask_gen(valid_image_gen, valid_mask_gen)


x = image_gen.next()
y = mask_gen.next()
for i in range(0, 1):
    image = x[i]
    mask = y[i]
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0])
    plt.show()




############################## set checkpoints and early stopping ########################################
import shutil
shutil.rmtree('logs\\fit')

log_dir='logs\\fit'

steps_per_epoch = 3 * (len(X_train1)) // batch_size

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=8)
board = TensorBoard(log_dir='logs-new\\fit',
                    histogram_freq=1,
                    write_images='True',
                    write_graph='True',
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=1)
saved_model_filepath = '7-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = saved_model_filepath, monitor='loss', save_best_only=True, verbose=1)

############################## training with segmentation models ########################################
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

sm_model = sm.Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=False)

inp = Input(shape=(None,None,1))
l1 = Conv2D(3,(1,1))(inp)
outp = sm_model(l1)

BVmodel = Model(inp, outp, name='BVmodel')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
BVmodel.compile(optimizer=optimizer, loss=total_loss, metrics=[sm.metrics.iou_score])

initialise = BVmodel.fit(my_generator,
                         validation_data=validation_datagen,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=steps_per_epoch,
                         callbacks=checkpoint,
                         epochs=2)
BVmodel.save('BV_model5_2epochs.hdf5')

BVmodel = tf.keras.models.load_model(r'BV_model5_2epochs.hdf5')

sm.utils.set_trainable(BVmodel, recompile=False) #does not work if recompile=True - follow raised issue on github
BVmodel.compile(optimizer=optimizer, loss=total_loss, metrics=[sm.metrics.iou_score])

history = BVmodel.fit(my_generator,
                       validation_data=validation_datagen,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=steps_per_epoch,
                       epochs=20,
                       callbacks=[early_stop, board, checkpoint])

BVmodel.save('BVmodel5_2+20epochs.hdf5')


########################################### OR from load weights ##########################################
# TODO: change model
del BVmodel
sm_model = sm.Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=False)

inp = Input(shape=(None,None,1))
l1 = Conv2D(3,(1,1))(inp)
outp = sm_model(l1)

BVmodel = Model(inp, outp, name='BVmodel')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# new_batch_size = 16
# new_steps_per_epoch = 3 * (len(X_train1)) // new_batch_size

BVmodel.load_weights('5-04-0.1413.hdf5')
for layer in BVmodel.layers:
    layer.trainable = True

BVmodel.compile(optimizer=optimizer, loss=total_loss, metrics=[sm.metrics.iou_score])
history = BVmodel.fit(my_generator,
                       validation_data=validation_datagen,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=steps_per_epoch,
                       epochs=5,
                       callbacks=[early_stop, board, checkpoint])

BVmodel.save('BV_model6_20epochs.hdf5')

########################################### OR from load model ##########################################
# TODO: train from loaded hdf5 model, get_config

BVmodel = tf.keras.models.load_model('BV_model6_20epochs', compile=True)

BVmodel = tf.keras.models.load_model('BV_model6_20epochs.hdf5',
                                            compile=True,
                                            custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                                            'iou_score': sm.metrics.iou_score}
                                            )

history = BVmodel.fit(my_generator,
                       validation_data=validation_datagen,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=steps_per_epoch,
                       epochs=25,
                       callbacks=[early_stop, board, checkpoint]
                      )

BVmodel.save('BV_model7_25+25+19epochs.hdf5')


################################################## analyse #####################################

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
iou = history.history['iou_score']  #accuracy
val_iou = history.history['val_iou_score']   #val_accuracy
plt.plot(epochs, iou, 'y', label='Training iou')
plt.plot(epochs, val_iou, 'r', label='Validation iou')



test_img_number = np.random.randint(0, len(X_test))
# test_img_number = 3031
test_img = X_test1[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = BVmodel.predict(test_img_input)
prediction = (prediction[0,:,:,0]>0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()


############################################# on unseen images ####################################################
BVmodel = tf.keras.models.load_model('BV_model_33+25epochs.hdf5', compile=False)

# unseen = io.imread('V:\Hanyi\Quantification\hm(not)_WT_3mo.tif')


from glob import glob
pred_file = r'V:\Hanyi\Quantification\WT_6mo'
predimg_list = []
for imgfile in glob(pred_file+'\*'):
    img = io.imread(imgfile)
    predimg_list.append(img)
unseen = np.asarray(predimg_list)

unseen_p = preprocess_input1(unseen)

unseen0 = unseen_p[0]
unseen0 = np.pad(unseen0, ((0,864), (0,192)), mode='constant')     #use 256xn-H%0

unseen0_patches = patchify(unseen0, (HEIGHT, WIDTH), step=HEIGHT)

randr, randc = np.random.randint(0, [unseen0_patches.shape[0], unseen0_patches.shape[1]])
unseen0_patches0 = unseen0_patches[randr,randc,:,:] # 6,1
# unseen0_patches0 = np.stack((unseen0_patches0,)*3, axis=-1)
unseen0_patches0 = np.expand_dims(unseen0_patches0, axis=(0,-1))

unseen0_pred = BVmodel.predict(unseen0_patches0)
unseen0_pred = (unseen0_pred[0,:,:,0]>0.01).astype(bool)

fig, ax = plt.subplots(1,2)
ax[0].imshow(unseen0_patches0[0,:,:,0], cmap='gray')
ax[1].imshow(unseen0_pred, cmap='gray')


unseen_preds = np.zeros(unseen0_patches.shape)
for r in range(unseen0_patches.shape[0]):
    for c in range(unseen0_patches.shape[1]):
        unseen0_patch = unseen0_patches[r,c,:,:]
        # unseen0_patch = np.stack((unseen0_patch,) * 3, axis=-1)
        unseen0_patch = np.expand_dims(unseen0_patch, axis=0)

        unseen_pred = BVmodel.predict(unseen0_patch)
        unseen_preds[r,c] = (unseen_pred[0,:,:,0]>0).astype(np.uint8)

unseen_preds = unpatchify(unseen_preds, unseen0.shape)
plt.imshow(unseen_preds)


###### on stack #######

def pred_BV(preprcSlice):
    padded = np.pad(preprcSlice, ((0,864), (0,192)), mode='constant')
    patches = patchify(padded, (HEIGHT, WIDTH), step=HEIGHT)
    predSlice = np.zeros(patches.shape)

    for r in range(patches.shape[0]):
        for c in range(patches.shape[1]):
            patch = patches[r,c,:,:]
            patch = np.expand_dims(patch, axis=0)

            predPatch = BVmodel.predict(patch)
            predSlice[r,c] = (predPatch[0,:,:,0]>0.2).astype(bool)

    predSlice = unpatchify(predSlice, padded.shape)
    predSlice = predSlice[:preprcSlice.shape[0], :preprcSlice.shape[1]]
    return predSlice

# test_slice0 = pred_BV(unseen0)
# plt.imshow(test_slice0, cmap='gray')

from joblib import Parallel, delayed
import tifffile
predStack = Parallel(n_jobs=10, verbose=10, backend='threading')(delayed(pred_BV)(sl) for sl in unseen_p)
predStack = np.array(predStack)
tifffile.imsave('WT_6mo_BVpredseg.tiff', predStack)

import napari
viewer = napari.Viewer()
viewer.add_image(predStack)

#######################################################################################################

tensorboard --logdir logs\fit







###################################### some post processing ####################################################
predStackPart = predStack[5:]
empty = np.zeros(image.shape)
predStackUse = np.concatenate(((empty,)*5,predStackPart), axis=0)
tifffile.imsave('WT_6mo_BVpredseg_use.tiff', predStackUse)



# major credits to: https://github.com/bnsreenu/python_for_microscopists/blob/76ff821bed35f931dd01f7a4204d71cb1ce16bbd/216_mito_unet__xferlearn_12_training_images.py
